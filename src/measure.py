import numpy as np
import torch
import torch.nn as nn
from rouge_score import rouge_scorer
from keybert import KeyBERT
from tqdm import tqdm


def calculate_rouge_l_score(target, prediction):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(target, prediction)
    return scores['rougeL'].fmeasure

def extract_keywords(model, text, top_k=5):
    keywords = model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=top_k)
    return [keyword[0] for keyword in keywords]


def common_measure(pred_list, answer_list): #seed = 42
    # Rouge-L 및 Keyword Recall 계산 결과 저장 변수
    rouge_l_scores = []
    keyword_recall_at_1_scores = []
    keyword_recall_at_3_scores = []
    keyword_recall_at_5_scores = []

    model = KeyBERT('distilbert-base-nli-mean-tokens')
    # 각 데이터에 대해 Rouge-L 및 Keyword Recall을 계산하고 결과 저장
    
    for answer, prediction in tqdm(zip(answer_list, pred_list)):
        # Rouge-L 계산
        rouge_l_score = calculate_rouge_l_score(answer, prediction)
        rouge_l_scores.append(rouge_l_score)

        # Keyword 추출 및 recall 계산
        answer_keywords = extract_keywords(model,answer)
        prediction_keywords = extract_keywords(model,prediction)

        keyword_recall_at_1 = len(set(answer_keywords[:1]).intersection(prediction_keywords)) / len(set(answer_keywords[:1]))
        keyword_recall_at_1_scores.append(keyword_recall_at_1)

        keyword_recall_at_3 = len(set(answer_keywords[:3]).intersection(prediction_keywords)) / len(set(answer_keywords[:3]))
        keyword_recall_at_3_scores.append(keyword_recall_at_3)

        keyword_recall_at_5 = len(set(answer_keywords[:5]).intersection(prediction_keywords)) / len(set(answer_keywords[:5]))
        keyword_recall_at_5_scores.append(keyword_recall_at_5)

    # 결과 출력
    average_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
    average_keyword_recall_at_1 = sum(keyword_recall_at_1_scores) / len(keyword_recall_at_1_scores)
    average_keyword_recall_at_3 = sum(keyword_recall_at_3_scores) / len(keyword_recall_at_3_scores)
    average_keyword_recall_at_5 = sum(keyword_recall_at_5_scores) / len(keyword_recall_at_5_scores)

    
    #print(f"Average Rouge-L Score: {average_rouge_l}")
    #print(f"Average Keyword Recall@1: {average_keyword_recall_at_1}")
    #print(f"Average Keyword Recall@3: {average_keyword_recall_at_3}")
    #print(f"Average Keyword Recall@5: {average_keyword_recall_at_5}")
    
    
    return_value = {
        "valid_avg_rouge_L" : average_rouge_l,
        "valid_key_recall@1" : average_keyword_recall_at_1,
        "valid_key_recall@3" : average_keyword_recall_at_3,
        "valid_key_recall@5" : average_keyword_recall_at_5,        
    }
    
    return return_value