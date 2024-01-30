import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, AutoTokenizer
import json
from typing import Tuple, List

from src.utils import get_data, split_train_validation_data


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
    train_data = raw_train_data
    valid_data = raw_valid_data
    test_data = raw_test
    #################################
    
    train_dataset = CustomDataset(args, train_data, tokenizer=tokenizer)
    valid_dataset = CustomDataset(args, valid_data, tokenizer=tokenizer)
    test_dataset = CustomDataset(args, test_data, tokenizer=tokenizer)


    train_loader = DataLoader(train_dataset,args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset,args.batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn)
    test_loader = DataLoader(test_dataset,args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
    
    return train_loader, valid_loader, test_loader


class CustomDataset(Dataset):
    def __init__(self, 
                 args,
                 data, 
                 tokenizer:AutoTokenizer):
        
        self.args = args
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        input_text = f"question: {entry['question']} context: {' '.join(entry['documents'])}"
        target_text = entry["answer"]

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