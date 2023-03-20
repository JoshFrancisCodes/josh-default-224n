from multitask_classifier import *
from bertviz import head_view
from IPython.display import display
from tokenizer import BertTokenizer

def get_tokens(input_ids, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    return tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pretrain-10-1e-05-multitask.pt")
    parser.add_argument("--dest", type=str, default="visualizations/")
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--squad_train", type=str, default="data/train-v2.0.json")
    parser.add_argument("--use_gpu", action='store_true')

    args = parser.parse_args()
    
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    with torch.no_grad():
        saved = torch.load(args.model)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        print(f"Loaded model to test from {args.model}")
        
        sst_train_data, para_train_data, sts_train_data, squad_train_data, num_labels = load_multitask_data(args.sst_train,args.para_train,args.sts_train,args.squad_train, split ='train')
        sst_train_data = SentenceClassificationDataset(sst_train_data, args)
        sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=1,
                                      collate_fn=sst_train_data.collate_fn)
        
        for batch in sst_train_dataloader:
            token_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            _, _, attention_weights = model(token_ids, attention_mask, return_attention_weights=True)
            
            tokens = get_tokens(token_ids.squeeze().tolist(), tokenizer)
            
            display(head_view(attention_weights, tokens))
            break

if __name__ == "__main__":
    main()