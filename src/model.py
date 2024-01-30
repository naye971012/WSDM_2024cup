import torch.nn as nn
import torch
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class baseModel:
    """
    Base Model
    Just Concat Question and Document as model Input
    Do not use History data
    Return Answer
    """
    def __init__(self, model_name:str):

        self.tokenizer = T5Tokenizer.from_pretrained(model_name,
                                            model_max_length = 1024, #이렇게 하는게 맞나? 오류줄이려고 이렇게함
                                            legacy = False,
                                              device_map = "balanced",  
                                              max_memory={0: "20GB", 1: "20GB"})
    
    
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, 
                                      device_map = "balanced",  
                                      max_memory={0: "20GB", 1: "20GB"})    
    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer