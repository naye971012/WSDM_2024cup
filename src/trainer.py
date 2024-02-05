import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from tqdm import tqdm
from torch.optim import *
from torch.utils.data import DataLoader
import wandb
import evaluate
import random
import json
from rouge_score import rouge_scorer

from model import *
from src.measure import * #공통 평가지표
from torch.optim.lr_scheduler import StepLR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args, 
          model:AutoModel,
          tokenizer: AutoTokenizer,
          train_loader:DataLoader, 
          valid_loader:DataLoader):
    
    model.train()
    
    ### Can be changed
    optimizer = SGD(model.parameters(), args.learning_rate, momentum=0.9, nesterov=True)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.25)
    
    optimizer.zero_grad()
    
    best_loss = np.inf
    for epoch in range(args.epoch):
        
        tokenizer = tokenizer.from_pretrained(f"model/{args.save_name}",
                                                model_max_length = 2048, #이렇게 하는게 맞나? 오류줄이려고 이렇게함
                                                legacy = False,
                                                device_map = "balanced",  
                                                max_memory={0: "22GB", 1:"22GB"}
                                                )
        model = model.from_pretrained(f"model/{args.save_name}",
                                        device_map = "balanced",  
                                        max_memory={0: "19GB", 1:"19GB"}
                                        )

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True)
        for i, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = torch.sum(outputs.loss)
            
            loss.backward()
            
            if i%args.accumulation_step==0:
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({"loss": loss.item()})
            if i%20==19 and args.is_logging:
                wandb.log({"train_loss": loss})

            if i==6000: ########early stopping for validation
                break
            
        
        model.save_pretrained(f"model/{args.save_name}")
        tokenizer.save_pretrained(f"model/{args.save_name}")
        
        valid_results = validation(args,epoch,model,tokenizer,valid_loader)
        
        
        if args.save_best_model:
            # Early Stopping 및 모델 저장 / 원하는 metric으로
            
            #loss 제외 metric은 클수록 좋은 것 이므로 -1을 곱해 작을수록 좋게 변형
            if args.best_model_metric !="loss":
                valid_results[args.best_model_metric] *= -1
            
            
            if valid_results[args.best_model_metric] < best_loss:
                best_loss = valid_results[args.best_model_metric]
                counter = 0
                # Save the model
                model.save_pretrained(f"model/{args.save_name}_best")
                tokenizer.save_pretrained(f"model/{args.save_name}_best")
            else:
                counter += 1
                print(f"not improvement for {counter} steps...")

            if counter >= args.patience and args.early_stopping:
                print(f"Early stopping after {epoch+1} epochs without improvement.")
                break
            
        scheduler.step()

def validation(args,
               epoch:int,
               model:AutoModelForSeq2SeqLM, 
               tokenizer:T5Tokenizer,
               valid_loader:DataLoader):
    """
    validation step
    return validation loss(type:float)
    """
    
    #print(model.device)
    #print(tokenizer.device)
    
    
    tokenizer = tokenizer.from_pretrained(f"model/{args.save_name}",
                                                model_max_length = 2048, #이렇게 하는게 맞나? 오류줄이려고 이렇게함
                                                legacy = False,
                                                device_map = "sequential",  
                                                max_memory={0: "22GB", 1:"22GB"}
                                                )
    model = model.from_pretrained(f"model/{args.save_name}",
                                        device_map = "sequential",  
                                        max_memory={0: "19GB", 1:"19GB"}
                                        )
    
    model.to(DEVICE)
    model.eval()
    
    rouge = evaluate.load('rouge')
     
    # Validation
    answer_list = []
    pred_list = []
    
    #validation_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(valid_loader, desc=f"Validation - Epoch {epoch+1}")):
            input_ids = batch["input_ids"].to(DEVICE)
            #attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE) 
            
            #below output is for checking validation loss
            #outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            #validation_loss += outputs.loss.sum().item()
            
            #below output is for checking rouge-L score, etc...
            
            generated_output = model.generate(input_ids=input_ids,max_length=200) #num_beams=10, length_penalty=2.0, early_stopping=True
                        
            pred_list.extend(tokenizer.batch_decode(generated_output, skip_special_tokens=True))
            answer_list.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))  
                      
            
    print(pred_list[0])
    print(answer_list[0])      
    
    #validation_loss /= len(valid_loader)
    
    #only used when see validation in txt file (pipe)
    #print(validation_loss)
    #for step, (i,j) in enumerate(zip(answer_list,pred_list)):
    #    print(f"{step}: =========================")
    #    print(i)
    #    print(j)
    #    print("==================================")
    
    
    results = rouge.compute(predictions=pred_list,
                             references=answer_list) #-> return Dict
    #results['loss'] = validation_loss
    
    score = []
    for ans, pred in zip(answer_list, pred_list):
        score.append(calculate_rouge_l_score(ans,pred))
    average_rouge_l = sum(score) / len(score)
    print(f"common rougeL: {average_rouge_l}")
    
    #common_results = common_measure(pred_list, answer_list)
    
    print(results)
    
    if args.is_logging:
        wandb.log({"Rouge-L_common": average_rouge_l})
        wandb.log(results)
        #wandb.log(common_results) #공통지표
                
        selected_indices = random.sample(range(len(pred_list)), k=10)
        selected_items_pred = [pred_list[i] for i in selected_indices]
        selected_items_answer = [answer_list[i] for i in selected_indices]
        
        data = {"Predicted": selected_items_pred, "Actual": selected_items_answer}
        table = wandb.Table(data=pd.DataFrame(data))
        wandb.log({"valid_prediction_example": table})
        
    return results
            
            

def inference(args, model, tokenizer, test_loader):
    
    model.to(DEVICE)
    
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Inference..."):
            input_ids = batch["input_ids"].to(DEVICE)
            uuid = batch['uuid']
            
            outputs = model.generate(input_ids, max_length=200) #num_beams=10, length_penalty=2.0, early_stopping=True

            # 생성된 결과를 토큰에서 텍스트로 디코딩하고, 각 배치 결과를 저장
            for i in range(input_ids.size(0)):
                generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
                #print( tokenizer.decode(outputs[i], skip_special_tokens=False))
                predictions.append({"uuid": uuid[i], "prediction": generated_text})
            
            #break
        
    with open("data/submission.json", "w", encoding="utf-8") as output_file:
        json.dump(predictions, output_file, ensure_ascii=False, indent=2)