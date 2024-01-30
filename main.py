import torch
import torch.nn as nn
import transformers
import numpy as np
import pandas as pd
import wandb
import argparse
import random
import os
import sys
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import logging

### Remove warning message ###
logging.getLogger().setLevel(logging.ERROR)

############## this block is just for import moudles ######
current_path = os.path.dirname(os.path.realpath(__file__))
print(current_path)
#parent_path = os.path.dirname(current_path)
#grand_path = os.path.dirname(parent_path)
#sys.path.append(grand_path)
###########################################################

from src.utils import set_global_seed, check_gpu
from src.dataset import get_loaders
from src.trainer import train, validation
from src.model import baseModel

def get_args():
    # ArgumentParser 객체를 생성합니다.
    parser = argparse.ArgumentParser(description='wsdm cup 2024')

    #dealing datapath
    parser.add_argument('--raw_train_path', type=str, default="data/release_train_data.json", help='original data path')
    parser.add_argument('--raw_test_path', type=str, default="data/phase_1_test.json", help='original data path')


    #dealing training arguments
    parser.add_argument('--model_name', type=str, default="t5-base", help='dataloader batch size')
    parser.add_argument('--batch_size', type=int, default=2, help='dataloader batch size')
    parser.add_argument('--epoch', type=int, default=50, help='train epoch')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')   
    parser.add_argument('--validation_ratio', type=float, default=0.1, help='dataloader validation length')

    parser.add_argument('--patience', type=int, default=5, help='patience (can be used when LR decline, earlystop, etc...)')
    parser.add_argument('--early_stopping', type=bool, default=False, help='early stop')


    #dealing boolean objects
    parser.add_argument('--is_train', type=bool, default=True, help='train or not')
    parser.add_argument('--is_test', type=bool, default=False, help='inference or not')
    parser.add_argument('--is_logging', type=bool, default=False, help='using wandb')
    
    
    #save model
    parser.add_argument('--save_best_model', type=bool, default=False, help='is save model')
    parser.add_argument('--best_model_metric', type=str, default="loss", help='loss,rouge1,rouge2,rougeL,rougeLsum,')    
    parser.add_argument('--save_name', type=str, default="baseline_model", help='save name')



    #wandb logging arguments
    parser.add_argument('--project_name', type=str, default="wsdm2024", help='project name of wandb')


    #arguemtns for reconstruction
    parser.add_argument('--seed', type=int, default=42, help='random seed')


    # 파싱된 인수들을 반환합니다.
    args = parser.parse_args()
    return args


def main(args):
    set_global_seed(args.seed)

    model, tokenizer = baseModel(args.model_name).get_model_and_tokenizer()

    print(f"model size: {model.num_parameters(only_trainable=False) / 1e6}M ")
    print(f"trainable params: {model.num_parameters(only_trainable=True) / 1e6}M ")    
    
    train_loader, valid_loader, test_loader = get_loaders(args, tokenizer)
    
    #train(args,model,tokenizer,train_loader,valid_loader)
    validation(args,0,model,tokenizer,valid_loader)

if __name__=="__main__":
    
    print(f"checking gpu... {torch.cuda.is_available()}")
    check_gpu()
    
    args = get_args()
    
    
    if args.is_logging:
        wandb.init(project=f"{args.project_name}",
                   name="1")
    
    
    main(args)