import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data, SquadDataset, MultitaskDataset

from evaluation import model_eval_sst, test_model_multitask, model_eval_multitask


TQDM_DISABLE=False

# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        
        ### TODO
        self.sentiment_classifier = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        self.paraphrase_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        self.similarity_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        self.self_supervised_attention = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE)
        self.qa_classifier = nn.Linear(BERT_HIDDEN_SIZE, 2)
        self.no_answer_classifier = nn.Linear(BERT_HIDDEN_SIZE, 1)

    def forward(self, input_ids, attention_mask, masked_positions=None, return_sequence=False):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = bert_output["last_hidden_state"]
        if return_sequence:
            cls_embedding = last_hidden_state
        else:
            cls_embedding = last_hidden_state[:, 0, :]
        
        if masked_positions is not None:
            batch_size, num_masked_positions = masked_positions.size()
            masked_positions_flat = masked_positions.view(-1).long()
            input_indices = torch.arange(batch_size).view(-1, 1).repeat(1, num_masked_positions).view(-1)
            
            masked_hidden_states = last_hidden_state[input_indices, masked_positions_flat]
            attention_weights = self.self_supervised_attention(masked_hidden_states)
            attention_weights = attention_weights.float()
            return cls_embedding, attention_weights
        else:
            return cls_embedding


    def predict_sentiment(self, input_ids, attention_mask, masked_positions=None):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        cls_embeddings, attention_weights = self.forward(input_ids, attention_mask, masked_positions)
        sentiment_logits = self.sentiment_classifier(cls_embeddings)
        return sentiment_logits, attention_weights


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2, 
                           masked_positions_1=None, masked_positions_2=None):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        cls_embeddings_1, attention_weights1 = self.forward(input_ids_1, attention_mask_1, masked_positions_1)
        cls_embeddings_2, attention_weights2 = self.forward(input_ids_2, attention_mask_2, masked_positions_2)
        concatenated_embeddings = torch.cat((cls_embeddings_1, cls_embeddings_2), dim=1)
        paraphrase_logit = self.paraphrase_classifier(concatenated_embeddings)
        return paraphrase_logit, attention_weights1, attention_weights2


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2,
                           masked_positions_1=None, masked_positions_2=None):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        cls_embeddings_1, attention_weights1 = self.forward(input_ids_1, attention_mask_1, masked_positions_1)
        cls_embeddings_2, attention_weights2 = self.forward(input_ids_2, attention_mask_2, masked_positions_2)
        concatenated_embeddings = torch.cat((cls_embeddings_1, cls_embeddings_2), dim=1)
        similarity_logit = self.similarity_classifier(concatenated_embeddings)
        return similarity_logit, attention_weights1, attention_weights2
    
    def predict_qa(self, input_ids, attention_mask, masked_positions=None):
        '''Given a batch of context-question pairs, outputs logits for the start and end positions of the answer span.
        The output should be a tuple containing two tensors:
        - start_logits: a tensor of shape (batch_size, sequence_length)
        - end_logits: a tensor of shape (batch_size, sequence_length)
        '''
        bert_output, attention_weights = self.forward(input_ids, attention_mask, masked_positions, return_sequence=True)
        qa_logits = self.qa_classifier(bert_output)  # Shape: (batch_size, sequence_length, 2)
        start_logits, end_logits = qa_logits.split(1, dim=-1)  # Split along the last dimension
        start_logits = start_logits.squeeze(-1)  # Shape: (batch_size, sequence_length)
        end_logits = end_logits.squeeze(-1)  # Shape: (batch_size, sequence_length)
        
        no_answer_logits = self.no_answer_classifier(bert_output[:, 0, :]).squeeze(-1)  # Shape: (batch_size,)

        # Normalize the start_logits, end_logits, and no_answer_logits
        start_logits = torch.cat([start_logits, no_answer_logits.unsqueeze(-1)], dim=-1)
        end_logits = torch.cat([end_logits, no_answer_logits.unsqueeze(-1)], dim=-1)

        start_logits = F.softmax(start_logits, dim=-1)
        end_logits = F.softmax(end_logits, dim=-1)
        
        return start_logits, end_logits, attention_weights




def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    # Load data
    sst_train_data, para_train_data, sts_train_data, squad_train_data, num_labels = load_multitask_data(args.sst_train,args.para_train,args.sts_train,args.squad_train, split ='train')
    sst_dev_data, para_dev_data, sts_dev_data, squad_dev_data, num_labels = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,args.squad_dev, split ='test')
    
    # Create the datasets
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)
    
    squad_train_data = SquadDataset(squad_train_data, args)
    squad_dev_data = SquadDataset(squad_dev_data, args)

    multitask_train_data = MultitaskDataset(sst_train_data, para_train_data, sts_train_data, squad_train_data)
    # multitask_dev_data = MultitaskDataset(sst_dev_data, para_dev_data, sts_dev_data, squad_dev_data)
    
    # Create the dataloaders
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)
    
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)
    
    multitask_train_dataloader = DataLoader(multitask_train_data, shuffle=True, batch_size=args.batch_size,
                                            collate_fn=multitask_train_data.collate_fn)
    
    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0
    
    # Define the loss functions for the tasks
    sst_loss_fn = nn.CrossEntropyLoss()
    para_loss_fn = nn.BCEWithLogitsLoss()
    sts_loss_fn = nn.MSELoss()
    qa_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch in tqdm(multitask_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            batch = {task: {key: tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor for key, tensor in task_batch.items()} for task, task_batch in batch.items()}
            sst_batch = batch["sst"]
            para_batch = batch["para"]
            sts_batch = batch["sts"]
            squad_batch = batch["squad"]
            optimizer.zero_grad()

            # Process the SST batch
            sst_logits, sst_attention_weights = model.predict_sentiment(sst_batch["token_ids"], sst_batch["attention_mask"], sst_batch["masked_positions"])
            sst_loss = sst_loss_fn(sst_logits.squeeze(-1), sst_batch["labels"])
            sst_ssa_loss = F.cross_entropy(sst_attention_weights, sst_batch["masked_positions"].view(-1).long(), reduction='sum', ignore_index=-1) / (args.batch_size * sst_batch["masked_positions"].size(1))

            # Process the paraphrase batch
            para_logits, para_attention_weights1, para_attention_weights2 = model.predict_paraphrase(para_batch["token_ids_1"], para_batch["attention_mask_1"],
                                                                            para_batch["token_ids_2"], para_batch["attention_mask_2"], 
                                                                            para_batch["masked_positions_1"], para_batch["masked_positions_2"])
            para_loss = para_loss_fn(para_logits.squeeze(-1).float(), para_batch["labels"].float())
            para_ssa_loss = F.cross_entropy(para_attention_weights1, para_batch["masked_positions_1"].view(-1).long(), reduction='sum', ignore_index=-1) / (args.batch_size * para_batch["masked_positions_1"].size(1))
            para_ssa_loss += F.cross_entropy(para_attention_weights2, para_batch["masked_positions_2"].view(-1).long(), reduction='sum', ignore_index=-1) / (args.batch_size * para_batch["masked_positions_2"].size(1))

            # Process the STS batch
            sts_logits, sts_attention_weights1, sts_attention_weights2 = model.predict_similarity(sts_batch["token_ids_1"], sts_batch["attention_mask_1"],
                                                                            sts_batch["token_ids_2"], sts_batch["attention_mask_2"], 
                                                                            sts_batch["masked_positions_1"], sts_batch["masked_positions_2"])
            sts_loss = sts_loss_fn(sts_logits.squeeze(-1), sts_batch["labels"].float())
            sts_ssa_loss = F.cross_entropy(sts_attention_weights1, sts_batch["masked_positions_1"].view(-1).long(), reduction='sum', ignore_index=-1) / (args.batch_size * sts_batch["masked_positions_1"].size(1))
            sts_ssa_loss += F.cross_entropy(sts_attention_weights2, sts_batch["masked_positions_2"].view(-1).long(), reduction='sum', ignore_index=-1) / (args.batch_size * sts_batch["masked_positions_2"].size(1))

            # Process the SQuAD batch
            squad_start_logits, squad_end_logits, squad_attention_weights = model.predict_qa(squad_batch["input_ids"], squad_batch["attention_mask"], squad_batch["masked_positions"])
            
            qa_start_loss = qa_loss_fn(squad_start_logits, squad_batch["start_positions"])
            qa_end_loss = qa_loss_fn(squad_end_logits, squad_batch["end_positions"])
            qa_loss = (qa_start_loss + qa_end_loss) / 2
            squad_ssa_loss = F.cross_entropy(squad_attention_weights, squad_batch["masked_positions"].view(-1).long(), reduction='sum', ignore_index=-1) / (args.batch_size * squad_batch["masked_positions"].size(1))

            # Combine the losses
            total_loss = sst_loss  + args.ssa_loss_weight * sst_ssa_loss \
                       + para_loss + args.ssa_loss_weight * para_ssa_loss \
                       + sts_loss  + args.ssa_loss_weight * sts_ssa_loss \
                       + qa_loss   + args.ssa_loss_weight * squad_ssa_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches

        # Evaluate the model on each task
        para_train_acc, _, _, sst_train_acc, _, _, sts_train_ac, _, _ = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        para_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_ac, _, _ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

        # Save the model if the dev performance improves
        avg_dev_acc = (para_dev_acc + sst_dev_acc + sts_dev_ac) / 3
        if avg_dev_acc > best_dev_acc:
            best_dev_acc = avg_dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}")
        print(f"SST: train acc :: {sst_train_acc :.3f}, dev acc :: {sst_dev_acc :.3f}")
        print(f"Paraphrase: train acc :: {para_train_acc :.3f}, dev acc :: {para_dev_acc :.3f}")
        print(f"STS: train acc :: {sts_train_ac :.3f}, dev acc :: {sts_dev_ac :.3f}")

def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")
    
    parser.add_argument("--squad_dev_out", type=str, default="predictions/squad-dev-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    
    parser.add_argument("--ssa_loss_weight", type=float, default=1.0, help="Weight of self-supervised attention loss")
    
    parser.add_argument("--finetune_squad", action='store_true', help="Fine-tune on the SQuAD dataset")
    parser.add_argument("--squad_train", type=str, default="data/train-v2.0.json", help="Path to the SQuAD train dataset")
    parser.add_argument("--squad_dev", type=str, default="data/dev-v2.0.json", help="Path to the SQuAD dev dataset")



    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
