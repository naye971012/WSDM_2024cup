import torch.nn as nn
import torch
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#t5 / 미스트랄(별로) /  3bilion models 위주
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        #retrival 분야 / 앙상블 등등
        
        tokenizer = AutoTokenizer.from_pretrained()        
        
    
    def forward(self,x):
        return x
    
    