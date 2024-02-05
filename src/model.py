import torch.nn as nn
import torch
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from peft import get_peft_model, LoraConfig, TaskType, PeftConfig, PeftModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class baseModel:
    """
    Base Model
    Just Concat Question and Document as model Input
    Do not use History data
    Return Answer
    """
    def __init__(self, model_name:str,
                        save_name:str,
                        is_test:bool):
        
        #if it is training step, load finetuned model(save_name)
        if is_test:
            model_name = save_name
             
        self.tokenizer = T5Tokenizer.from_pretrained(model_name,
                                                model_max_length = 1024, #이렇게 하는게 맞나? 오류줄이려고 이렇게함
                                                legacy = False,
                                                device_map = "balanced",  
                                                max_memory={0: "20GB"})
        
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, 
                                        device_map = "balanced",  
                                        max_memory={0: "20GB"}) 
        
    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer

class baseModel_large_with_lora:
    """
    Base Model with lora
    """
    def __init__(self, model_name:str,
                        save_name:str,
                        is_test:bool):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                model_max_length = 1024, #이렇게 하는게 맞나? 오류줄이려고 이렇게함
                                                legacy = False,
                                                device_map = "balanced",  
                                        max_memory={0: "22GB"}) 
        
        
        peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1,
        )   
        
        #if it is test step, load finetuned model(save_name)
        if is_test:
            model_name = save_name
            config = PeftConfig.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
            self.model = PeftModel.from_pretrained(self.model, model_name)
        
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, 
                                        device_map = "balanced",  
                                        max_memory={0: "23GB"}) 
        
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        
        
        
    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer

class LongModel_base_with_lora:
    """
    Base Model with lora
    """
    def __init__(self, model_name:str,
                        save_name:str,
                        is_test:bool,
                        device_map:str='auto'):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                model_max_length = 2048, #이렇게 하는게 맞나? 오류줄이려고 이렇게함
                                                legacy = False,
                                                device_map = device_map,  
                                                max_memory={0: "22GB", 1:"22GB"}
                                                )  
        
        
        peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q","v"]
        )   
        
        #if it is test step, load finetuned model(save_name)
        if is_test:
            model_name = save_name
            config = PeftConfig.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
            self.model = PeftModel.from_pretrained(self.model, model_name)
        
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, 
                                        device_map = device_map,  
                                        max_memory={0: "19GB", 1:"19GB"}
                                        ) 

            #print(self.model)
            
            #self.model = get_peft_model(self.model, peft_config)
            
            #self.model.base_model.model.encoder.enable_input_require_grads()
            #self.model.base_model.model.decoder.enable_input_require_grads()
            
            #self.model.print_trainable_parameters()
        
        
        
    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer