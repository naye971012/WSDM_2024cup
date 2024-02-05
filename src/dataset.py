import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, AutoTokenizer
import json
from typing import Tuple, List, Dict

from src.bm25 import sort_and_remain_5_documents
from src.utils import *


def get_loaders(args, tokenizer):
    """
    get raw data, processing, and return train/valid/test dataloader

    Args:
        args
    """
    raw_train, raw_test = get_data(args)
    raw_train_data, raw_valid_data = split_train_validation_data(args,raw_train)
    ## -> Datatype = List[Dict]
    
    ##### need to preprocessing #####
    train_data = preprocess(raw_train_data,tokenizer)
    valid_data = preprocess(raw_valid_data,tokenizer)
    test_data = preprocess(raw_test,tokenizer)
    #################################
    
    train_dataset = baseDataset_with_history(args, train_data, tokenizer=tokenizer)
    valid_dataset = baseDataset_with_history(args, valid_data, tokenizer=tokenizer)
    test_dataset = baseTestDataset(args, test_data, tokenizer=tokenizer)
    

    train_loader = DataLoader(train_dataset,args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=4)
    valid_loader = DataLoader(valid_dataset,args.batch_size * 16, shuffle=False, collate_fn=valid_dataset.collate_fn, num_workers=4)
    test_loader = DataLoader(test_dataset,args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=4)
    
    return train_loader, valid_loader, test_loader


class baseDataset(Dataset):
    """
    just concat Question and Document
    do not use history data
    
    implemented based on T5
    """
    def __init__(self, 
                 args,
                 data, 
                 tokenizer:T5Tokenizer):
        
        self.args = args
        self.data = data
        self.tokenizer = tokenizer

        self.myvocab=dict()
        self.dealing_special_tokens()
        
    def dealing_special_tokens(self):
        #self.tokenizer에 있는 extra_id를 이용하여 q_start, q_end 등의 token을 만듦.
                
        self.myvocab = {
            "<question>" : "<extra_id_0>",
            "</question>" : "<extra_id_1>",
            "<document>" : "<extra_id_2>",
            "</document>" : "<extra_id_3>",
            "<answer>" : "<extra_id_4>",
            "</answer>" : "<extra_id_5>",
        }
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        
        if random.random() < 0.2:
            random.shuffle(entry['documents'])        
        
        input_text = self.myvocab['<question>'] + entry['question'] + self.myvocab['</question>'] \
                + self.myvocab['<document>'] + ' '.join(entry['documents']) + self.myvocab['</document>'] \

        target_text = self.myvocab['<answer>'] + entry["answer"] + self.myvocab['</answer>']

        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
        targets = self.tokenizer(target_text, return_tensors="pt", max_length=1024, truncation=True)
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }

    def collate_fn(self,batch):

        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        # Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class baseDataset_with_history(Dataset):
    """
    just concat Question and Document
    do not use history data
    
    implemented based on T5
    """
    def __init__(self, 
                 args,
                 data, 
                 tokenizer:T5Tokenizer):
        
        self.args = args
        self.data = data
        self.tokenizer = tokenizer

        self.myvocab=dict()
        self.dealing_special_tokens()
        
    def dealing_special_tokens(self):
        #self.tokenizer에 있는 extra_id를 이용하여 q_start, q_end 등의 token을 만듦.
                
        self.myvocab = {
            "<question>" : "<extra_id_0>",
            "</question>" : "<extra_id_1>",
            "<document>" : "<extra_id_2>",
            "</document>" : "<extra_id_3>",
            "<answer>" : "<extra_id_4>",
            "</answer>" : "<extra_id_5>",
            "<history>" : "<extra_id_6>",
            "</history>" : "<extra_id_7>",
            "<Q>" : "<extra_id_8>",
            "</Q>" : "<extra_id_9>",
            "<A>" : "<extra_id_10>",
            "</A>" : "<extra_id_11>",
        }
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        
        if random.random() < 0.2:
            random.shuffle(entry['documents'])        
        
        history_sum = ""
        for his in entry['history']:
            history_sum += f" {self.myvocab['<Q>']} {his['question']} {self.myvocab['</Q>']}" + \
                f" {self.myvocab['<A>']} {his['answer']} {self.myvocab['</A>']}"
        
        document_sum = ""
        for doc in entry['documents']:
            document_sum += f" {self.myvocab['<document>']} {doc} {self.myvocab['</document>']}"
        
        input_text = self.myvocab['<question>'] + entry['question'] + self.myvocab['</question>'] \
            + document_sum + history_sum

        target_text = self.myvocab['<answer>'] + entry["answer"] + self.myvocab['</answer>']

        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=2048, truncation=True)
        targets = self.tokenizer(target_text, return_tensors="pt", max_length=2048, truncation=True)
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }

    def collate_fn(self,batch):

        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        # Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }




class baseTestDataset(Dataset):
    """
    just concat Question and Document
    do not use history data
    
    implemented based on T5
    """
    def __init__(self, 
                 args,
                 data, 
                 tokenizer:T5Tokenizer):
        
        self.args = args
        self.data = data
        self.tokenizer = tokenizer

        self.myvocab=dict()
        self.dealing_special_tokens()
        
    def dealing_special_tokens(self):
        #self.tokenizer에 있는 extra_id를 이용하여 q_start, q_end 등의 token을 만듦.
                
        self.myvocab = {
            "<question>" : "<extra_id_0>",
            "</question>" : "<extra_id_1>",
            "<document>" : "<extra_id_2>",
            "</document>" : "<extra_id_3>",
            "<answer>" : "<extra_id_4>",
            "</answer>" : "<extra_id_5>",
        }
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        
        input_text = self.myvocab['<question>'] + entry['question'] + self.myvocab['</question>'] \
                + self.myvocab['<document>'] + ' '.join(entry['documents']) + self.myvocab['</document>'] \

        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

        uuid = entry['uuid']
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "uuid": uuid
        }

    def collate_fn(self,batch):

        input_ids = [item["input_ids"] for item in batch]
        uuid = [item["uuid"] for item in batch]
        
        # Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        return {
            "uuid": uuid,
            "input_ids": input_ids,
        }


def preprocess(data:List[Dict], tokenizer):
    """
    remove emoji
    sort by bm25 and remove documents over than 5
    """
    
    new_data = []
    
    for temp in data:
                
        temp['documents'] = [remove_emoji(doc) for doc in temp["documents"]]
        
        """
        try:
            temp['documents'] = sort_and_remain_5_documents(temp['documents'],temp['question'])
        except:
            pass
            #when there is no documents
        """
        
        new_data.append(temp)
    
    return new_data