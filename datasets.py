#!/usr/bin/env python3

'''
This module contains our Dataset classes and functions to load the 3 datasets we're using.

You should only need to call load_multitask_data to get the training and dev examples
to train your model.
'''


import csv

import torch
from torch.utils.data import Dataset
from tokenizer import BertTokenizer
import random
import json


def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())
    
class MultitaskDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, index):
        return tuple(dataset[index % len(dataset)] for dataset in self.datasets)

    def __len__(self):
        return max(len(dataset) for dataset in self.datasets)
    
    def collate_fn(self, data):
        sst_batch, para_batch, sts_batch, squad_batch = zip(*data)

        # Collate each task's data
        sst_collated = self.datasets[0].collate_fn(sst_batch)
        para_collated = self.datasets[1].collate_fn(para_batch)
        sts_collated = self.datasets[2].collate_fn(sts_batch)
        squad_collated = self.datasets[3].collate_fn(squad_batch)

        # Combine the collated data into a single dictionary
        multitask_collated = {
            "sst": sst_collated,
            "para": para_collated,
            "sts": sts_collated,
            "squad": squad_collated
        }

        return multitask_collated



class SquadDataset(Dataset):
    def __init__(self, dataset, args, mask_probability=0.15):
        self.dataset = dataset
        self.p = args
        self.mask_probability = mask_probability
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        context, question, answer, example_id = self.dataset[idx]

        encoding = self.tokenizer(question, context, max_length=512, truncation=True, padding='max_length')
        input_ids = encoding['input_ids']
        token_type_ids = encoding['token_type_ids']
        attention_mask = encoding['attention_mask']
        
        answer_start_token_idx = -1
        answer_end_token_idx = -1
        
        if not answer:
            has_answer = 0
        else:
            has_answer = 1
            answer = answer[0]
            start_position = ...
            end_position = ...
            answer_text = answer['text']
            answer_start = answer['answer_start']
            
            # Get the encoded version of the answer text
            answer_encoding = self.tokenizer(answer_text)
            answer_input_ids = answer_encoding['input_ids'][1:-1]  # remove [CLS] and [SEP] tokens

            

            # Loop through the input_ids to find the matching sequence of tokens
            for i in range(len(input_ids) - len(answer_input_ids) + 1):
                if input_ids[i:i+len(answer_input_ids)] == answer_input_ids:
                    answer_start_token_idx = i
                    answer_end_token_idx = i + len(answer_input_ids) - 1
                    break
            
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'start_positions': answer_start_token_idx,
            'end_positions': answer_end_token_idx,
            'example_id': example_id,
            'has_answer': has_answer
        }

    def pad_data(self, data):
        input_ids = [torch.tensor(d['input_ids']) for d in data]
        token_type_ids = [torch.tensor(d['token_type_ids']) for d in data]
        attention_mask = [torch.tensor(d['attention_mask']) for d in data]
        start_positions = [d['start_positions'] for d in data]
        end_positions = [d['end_positions'] for d in data]
        example_ids = [d['example_id'] for d in data]
        has_answers = [d['has_answer'] for d in data]

        # Add code to mask tokens
        masked_positions = []
        for sent in input_ids:
            positions = [i for i, token in enumerate(sent) if self.tokenizer.convert_ids_to_tokens(token.item()) not in ["[CLS]", "[SEP]", "[PAD]"]]
            masked_count = int(len(positions) * self.mask_probability)
            random.shuffle(positions)
            mask_indices = positions[:masked_count]
            sent[mask_indices] = self.tokenizer.mask_token_id
            masked_positions.append(torch.tensor(mask_indices))

        masked_positions = torch.nn.utils.rnn.pad_sequence(masked_positions, batch_first=True, padding_value=-1)

        padded = self.tokenizer.pad({
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }, return_tensors='pt')

        return (
            padded['input_ids'],
            padded['token_type_ids'],
            padded['attention_mask'],
            torch.tensor(start_positions),
            torch.tensor(end_positions),
            example_ids,
            masked_positions,
            torch.tensor(has_answers)
        )

    def collate_fn(self, data):
        input_ids, token_type_ids, attention_mask, start_positions, end_positions, example_ids, masked_positions, has_answers = self.pad_data(data)

        batched_data = {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'start_positions': start_positions,
            'end_positions': end_positions,
            'example_ids': example_ids,
            'masked_positions': masked_positions,
            'has_answers' : has_answers
        }

        return batched_data



class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args, mask_probability=0.15):
        self.dataset = dataset
        self.p = args
        self.mask_probability = mask_probability
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):

        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)
        
        # Add code to mask tokens
        masked_positions = []
        for sent in token_ids:
            positions = [i for i, token in enumerate(sent) if self.tokenizer.convert_ids_to_tokens(token.item()) not in ["[CLS]", "[SEP]", "[PAD]"]]
            masked_count = int(len(positions) * self.mask_probability)
            random.shuffle(positions)
            mask_indices = positions[:masked_count]
            sent[mask_indices] = self.tokenizer.mask_token_id
            masked_positions.append(torch.tensor(mask_indices))

        masked_positions = torch.nn.utils.rnn.pad_sequence(masked_positions, batch_first=True, padding_value=-1)

        return token_ids, attention_mask, labels, sents, sent_ids, masked_positions

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids, masked_positions = self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids,
                'masked_positions': masked_positions
            }

        return batched_data


class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression =False, mask_probability=0.15):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.mask_probability = mask_probability
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])
        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)
            
        # Add code to mask tokens for both sent1 and sent2
        masked_positions = []
        for sent in [token_ids, token_ids2]:
            sent_masked_positions = []
            for s in sent:
                positions = [i for i, token in enumerate(s) if self.tokenizer.convert_ids_to_tokens(token.item()) not in ["[CLS]", "[SEP]", "[PAD]"]]
                masked_count = int(len(positions) * self.mask_probability)
                random.shuffle(positions)
                mask_indices = positions[:masked_count]
                s[mask_indices] = self.tokenizer.mask_token_id
                sent_masked_positions.append(torch.tensor(mask_indices))

            masked_positions.append(torch.nn.utils.rnn.pad_sequence(sent_masked_positions, batch_first=True, padding_value=-1))  
            

        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
                labels,sent_ids, *masked_positions)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         labels, sent_ids, masked_positions1, masked_positions2) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'labels': labels,
                'sent_ids': sent_ids,
                'masked_positions_1': masked_positions1,
                'masked_positions_2': masked_positions2
            }

        return batched_data


class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])


        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
               sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'sent_ids': sent_ids
            }

        return batched_data


def load_multitask_test_data():
    paraphrase_filename = f'data/quora-test.csv'
    sentiment_filename = f'data/ids-sst-test.txt'
    similarity_filename = f'data/sts-test.csv'
    squad_filename = f'data/squad-test.csv'

    sentiment_data = []
    with open(sentiment_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            sent = record['sentence'].lower().strip()
            sentiment_data.append(sent)

    print(f"Loaded {len(sentiment_data)} test examples from {sentiment_filename}")

    paraphrase_data = []
    with open(paraphrase_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            #if record['split'] != split:
            #    continue
            paraphrase_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(paraphrase_data)} test examples from {paraphrase_filename}")

    similarity_data = []
    with open(similarity_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            similarity_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(similarity_data)} test examples from {similarity_filename}")
    
    squad_data = []
    with open(squad_filename, 'r') as fp:
            squad_raw_data = json.load(fp)
            for article in squad_raw_data['data']:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa['question']
                        squad_data.append((context, question))

    return sentiment_data, paraphrase_data, similarity_data, squad_data



def load_multitask_data(sentiment_filename,paraphrase_filename,similarity_filename,squad_filename,split='train'):
    sentiment_data = []
    num_labels = {}
    if split == 'test':
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                sentiment_data.append((sent,sent_id))
    else:
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sentiment_data.append((sent, label,sent_id))

    print(f"Loaded {len(sentiment_data)} {split} examples from {sentiment_filename}")

    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))

    else:
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    paraphrase_data.append((preprocess_string(record['sentence1']),
                                            preprocess_string(record['sentence2']),
                                            int(float(record['is_duplicate'])),sent_id))
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == 'test':
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2'])
                                        ,sent_id))
    else:
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        float(record['similarity']),sent_id))

    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")
    
    squad_data = []
    if split == 'test':
        with open(squad_filename, 'r') as fp:
            squad_raw_data = json.load(fp)
            for article in squad_raw_data['data']:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa['question']
                        id = qa['id']
                        squad_data.append((context, question, id))
    else:
        with open(squad_filename, 'r') as fp:
            squad_raw_data = json.load(fp)
            for article in squad_raw_data['data']:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa['question']
                        id = qa['id']
                        answers = qa['answers']
                        squad_data.append((context, question, answers, id))

    print(f"Loaded {len(squad_data)} {split} examples from {squad_filename}")

    return sentiment_data, paraphrase_data, similarity_data, squad_data, num_labels
