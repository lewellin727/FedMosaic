from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import copy

from src.utils import *
from src.train import *

def init_silos(silo_data, aug_data, augment_model):
    silos = []
    for silo_id, doc_list in silo_data.items():
        docs = []
        for doc_id in doc_list:
            docs.append(Document(id2psg(doc_id, aug_data), id2aug(doc_id, aug_data), augment_model))
        silos.append(Silo(silo_id, docs))
    return silos


class Silo:
    def __init__(self, silo_id, Documents):
        self.id = silo_id
        self.Documents = Documents
        self._build_index()
    
    def _build_index(self):
        self.corpus = [doc.passage for doc in self.Documents] 
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        
        # from sentence_transformers import SentenceTransformer
        # self.emb_model = SentenceTransformer(embed_model_path)
        # self.corpus_embeddings = self.emb_model.encode(self.corpus, convert_to_numpy=True, normalize_embeddings=True)

    def retrieve(self, question, k):
        question_vec = self.vectorizer.transform([question])
        cosine_similarities = cosine_similarity(question_vec, self.tfidf_matrix).flatten()
        topk_indices = np.argsort(cosine_similarities)[::-1][:k]
        docs = [self.Documents[i] for i in topk_indices]

        return docs
    
    # def retrieve_emb(self, question, k):
    #     q_emb = self.emb_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    #     similarities = np.dot(self.corpus_embeddings, q_emb)  
    #     top_k_idx = np.argsort(similarities)[-k:][::-1]
    #     docs = [self.Documents[i] for i in top_k_idx]
    #     return docs

    def retrieve_with_scores(self, question, k):
        question_vec = self.vectorizer.transform([question])
        cosine_similarities = cosine_similarity(question_vec, self.tfidf_matrix).flatten()
        topk_indices = np.argsort(cosine_similarities)[::-1][:k]
        docs = [self.Documents[i] for i in topk_indices]
        scores = [cosine_similarities[i] for i in topk_indices]

        return docs, scores


def overlap(merged_mask_1, merged_mask_2):
    assert len(merged_mask_1) == len(merged_mask_2)
    overlap_radio = sum([1 for i, j in zip(merged_mask_1, merged_mask_2) if i == j]) / len(merged_mask_1)

    return overlap_radio

def merge_mask(mask_dict):
    merged_mask = []
    for name, mask in mask_dict.items():
        mask = mask.view(-1).cpu().numpy()
        merged_mask.extend(mask)
    merged_mask = [int(i) for i in merged_mask]
    return merged_mask


def fedranking(question, documents, mask_save_dir, config, reranker):
    passages = [doc.passage for doc in documents]
    doc_ids = [doc.id for doc in documents]
    doc_cnt = len(passages)


    print(f'Silo Ranking...')
    query_doc_pairs = [(question, doc) for doc in passages]
    scores = reranker.compute_score(query_doc_pairs, normalize=True)
    scores = {doc_id: score for doc_id, score in zip(doc_ids, scores)}

    mask_paths = [os.path.join(mask_save_dir, f"doc_id={doc_id}.pt") for doc_id in doc_ids]
    merged_masks = [merge_mask(load_bitpacked_mask(mask_path)) for mask_path in mask_paths]
    masks = np.array(merged_masks) 
    N, D = masks.shape
    equal_matrix = (masks[:, None, :] == masks[None, :, :]) 
    overlap_matrix = equal_matrix.sum(axis=2) / D 

    overlap_radios = {}
    for i in range(doc_cnt):
        for j in range(i + 1, doc_cnt):
            overlap_radios[(doc_ids[i], doc_ids[j])] = overlap_matrix[i, j]
            overlap_radios[(doc_ids[j], doc_ids[i])] = overlap_matrix[i, j]


    print(f'Server Selecting...')
    selected_ids = []
    selected_scores = []
    lambda_ol = config['rank']['lambda_ol']
    threshold = config['rank']['threshold']
    for _ in range(config['rank']['rank_K']):
        max_score = -999
        tar_id = -1
        for idx, doc_id in enumerate(doc_ids):
            if select(selected_ids, doc_id, scores, threshold):
                tmp = [doc_id]
                for i in selected_ids:
                    tmp.append(i)
                tmp_score = ranking_objective(tmp, scores, overlap_radios, lambda_ol)
                if tmp_score > max_score:
                    max_score = tmp_score
                    tar_id = doc_id
        if tar_id != -1:
            selected_ids.append(tar_id)
            selected_scores.append(scores[tar_id])
        else:
            break
    
    selected_scores = [s / sum(selected_scores) for s in selected_scores]
    return selected_ids, selected_scores



def select(selected_ids, tar_id, scores, threshold):
    if tar_id in selected_ids:
        return False
    if scores[tar_id] < threshold and len(selected_ids) > 0:
        return False
    return True


def ranking_objective(selected_ids, scores, overlap_radios, lambda_ol):

    total_length = len(selected_ids)
    ranking_socre = sum([scores[i] for i in selected_ids]) / total_length
    overlapping = 0
    for i in range(total_length):
        for j in range(i + 1, total_length):
            overlapping += overlap_radios[(selected_ids[i], selected_ids[j])]
    if total_length > 1:
        overlapping = overlapping / (total_length * (total_length - 1) / 2)

    return ranking_socre - lambda_ol * overlapping
    



        

    
