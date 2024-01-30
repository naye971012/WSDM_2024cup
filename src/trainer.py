import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb
import evaluate
import random
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args, 
          model:AutoModel,
          tokenizer: AutoTokenizer,
          train_loader:DataLoader, 
          valid_loader:DataLoader):
    
    model.train()
    
    ### Can be changed
    optimizer = AdamW(model.parameters(), args.learning_rate)
    
    
    best_loss = np.inf
    for epoch in range(args.epoch):
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", dynamic_ncols=True)

        for i, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = torch.sum(outputs.loss)
            
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({"loss": loss.item()})
            if i%20==19 and args.is_logging:
                wandb.log({"train_loss": loss})

        valid_results = validation(args,epoch,model,tokenizer,valid_loader)

        
        if args.save_best_model:
            # Early Stopping 및 모델 저장 / 원하는 metric으로
            if valid_results[args.best_model_metric] < best_loss:
                best_loss = valid_results[args.best_model_metric]
                counter = 0
                # Save the model
                model.save_pretrained(f"model/{args.save_name}")
                tokenizer.save_pretrained(f"model/{args.save_name}")
            else:
                counter += 1
                print(f"not improvement for {counter} steps...")

            if counter >= args.patience and args.early_stopping:
                print(f"Early stopping after {epoch+1} epochs without improvement.")
                break


def validation(args,
               epoch:int,
               model:T5ForConditionalGeneration, 
               tokenizer:T5Tokenizer,
               valid_loader:DataLoader):
    """
    validation step
    return validation loss(type:float)
    """
    model.to(DEVICE)
    rouge = evaluate.load('rouge')
     
    # Validation
    answer_list = []
    pred_list = []
    
    validation_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc=f"Validation - Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE) 

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            validation_loss += outputs.loss.sum().item()
            
            answer_list.extend(tokenizer.batch_decode(torch.argmax(outputs.logits,dim=-1), skip_special_tokens=True))
            pred_list.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))
         
    validation_loss /= len(valid_loader)
    
    """ #only used when see validation in txt file (pipe)
    for step, (i,j) in enumerate(zip(answer_list,pred_list)):
        print(f"{step}: =========================")
        print(i)
        print(j)
        print("==================================")
    """
    
    #compute rouge score and merge loss in dictionary
    results = rouge.compute(predictions=pred_list,
                             references=answer_list) #-> return Dict
    results['loss'] = validation_loss
        
    if args.is_logging:
        wandb.log(results)
                
        selected_indices = random.sample(range(len(pred_list)), k=10)
        selected_items_pred = [pred_list[i] for i in selected_indices]
        selected_items_answer = [answer_list[i] for i in selected_indices]
        
        for i in range(10):
            wandb.log({"valid_prediction_example": selected_items_pred[i]})
            wandb.log({"valid_label_example": selected_items_answer[i]})
    
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