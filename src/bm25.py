from rank_bm25 import BM25Okapi
from typing import List

def sort_and_remain_5_documents(documents:List[str],
                                question:str):
	tokenized_documents = [doc.split() for doc in documents]
    
	bm25 = BM25Okapi(tokenized_documents)

    # Perform BM25 search and sort
	tokenized_question = question.split()
    
	doc_scores = bm25.get_scores(tokenized_question)

    # Get sorted document indices
	sorted_indices = sorted(range(len(doc_scores)), key=lambda k: doc_scores[k], reverse=True)

    #sort document and remain 5
	sorted_documents = [documents[i] for cnt, i in enumerate(sorted_indices) if cnt<5]
 
	return sorted_documents