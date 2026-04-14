import os
import time
import json
import random
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
from tqdm import tqdm
from src.utils import *
    
def clustering(root_dir, dataset_name, dataset_type, silo_data, config):
    C = config['offline']['clustering']['C']
    seed = config['offline']['clustering']['random_seed']
    k = config['offline']['clustering']['eval_topk']

    data_apth = os.path.join(root_dir, f'dataset/{dataset_name}/{dataset_type}/aug.json')
    aug_data = json.load(open(data_apth, 'r', encoding='utf-8'))
    questions = [d['question'] for d in aug_data]

    output_path_1 = os.path.join(root_dir, f'dataset/{dataset_name}/{dataset_type}/C={C}/clu.json')
    output_path_2 = os.path.join(root_dir, f'dataset/{dataset_name}/{dataset_type}/C={C}/test_dir/clu_kmeans.json')
    output_path_3 = os.path.join(root_dir, f'dataset/{dataset_name}/{dataset_type}/C={C}/test_dir/clu_random.json')
    os.makedirs(os.path.dirname(output_path_2), exist_ok=True)


    constrained_results = {}
    simple_results = {}
    random_results = {}

    print("Running capacity-constrained clustering...")
    for silo_id, silo in tqdm(silo_data.items(), desc="Constrained Clustering"):
        K = int(len(silo) / C)
        clustered = ConstrainedKmeans(silo, aug_data, C=C, seed=seed)
        constrained_results[silo_id] = clustered

    print("Running simple KMeans clustering...")
    for silo_id, silo in tqdm(silo_data.items(), desc="Simple KMeans Clustering"):
        clustered = simple_kmeans_cluster(silo, aug_data, C=C, seed=seed)
        simple_results[silo_id] = clustered

    print("Running random clustering...")
    for silo_id, silo in tqdm(silo_data.items(), desc="Random Clustering"):
        clustered = random_cluster(silo, aug_data, C=C, seed=seed)
        random_results[silo_id] = clustered


    json.dump(constrained_results, open(output_path_1, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    json.dump(simple_results, open(output_path_2, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    json.dump(random_results, open(output_path_3, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
    print(f"Saved constrained clustering to {output_path_1}")
    print(f"Saved simple clustering to {output_path_2}")
    print(f"Saved random clustering to {output_path_3}")

    print("--- Clustering Evaluation (Per Silo) ---")
    for silo_id, silo in silo_data.items():
        result1 = evaluate_clustering(silo, aug_data, constrained_results[silo_id], questions, k=k)
        result2 = evaluate_clustering(silo, aug_data, simple_results[silo_id], questions, k=k)
        result3 = evaluate_clustering(silo, aug_data, random_results[silo_id], questions, k=k)

        print(f"\nSilo: {silo_id}")
        print(f"  Constrained  | Top-{k} same cluster ratio: {result1['same_cluster_ratio']:.4f} | "
            f"Question cluster ratio: {result1['question_same_cluster_ratio']:.4f} | Cluster std: {result1['std']:.2f}")
        print(f"  Simple       | Top-{k} same cluster ratio: {result2['same_cluster_ratio']:.4f} | "
            f"Question cluster ratio: {result2['question_same_cluster_ratio']:.4f} | Cluster std: {result2['std']:.2f}")
        print(f"  Random       | Top-{k} same cluster ratio: {result3['same_cluster_ratio']:.4f} | "
            f"Question cluster ratio: {result3['question_same_cluster_ratio']:.4f} | Cluster std: {result3['std']:.2f}")

    return constrained_results, simple_results 


def ConstrainedKmeans(id_list, aug_data, C, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    doc_ids = id_list
    passages = [id2psg(id, aug_data) for id in id_list]

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X = tfidf.fit_transform(passages)
    X = normalize(X, norm='l2')

    n_clusters = len(doc_ids) // C + 1

    from k_means_constrained import KMeansConstrained
    kmeans = KMeansConstrained(n_clusters=n_clusters, random_state=seed, size_min=C-2, size_max=C)
    labels = kmeans.fit_predict(X.toarray())


    clustered_result = {str(doc_ids[i]): int(labels[i]) for i in range(len(doc_ids))}
    clustered_result = dict(sorted(clustered_result.items(), key=lambda item: item[1]))
    return clustered_result


def simple_kmeans_cluster(id_list, aug_data, C, seed):
    random.seed(seed)
    np.random.seed(seed)

    doc_ids = id_list
    passages = [id2psg(id, aug_data) for id in id_list]

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X = tfidf.fit_transform(passages)

    n_clusters = len(doc_ids) // C + 1

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, max_iter=500, n_init='auto')
    labels = kmeans.fit_predict(X.toarray())

    clustered_result = {str(doc_ids[i]): int(labels[i]) for i in range(len(doc_ids))}
    clustered_result = dict(sorted(clustered_result.items(), key=lambda item: item[1]))
    return clustered_result

def random_cluster(id_list, aug_data, C, seed):
    random.seed(seed)
    np.random.seed(seed)

    doc_ids = id_list
    n_clusters = len(id_list) // C + 1

    labels = np.random.randint(0, n_clusters, size=len(id_list))
    clustered_result = {str(doc_ids[i]): int(labels[i]) for i in range(len(doc_ids))}

    clustered_result = dict(sorted(clustered_result.items(), key=lambda item: item[1]))
    return clustered_result


def evaluate_clustering(id_list, aug_data, clustered_result, questions, k=5):
    id_to_idx = id_list
    passages = [id2psg(id, aug_data) for id in id_list]
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X = tfidf.fit_transform(passages)

    sim_matrix = cosine_similarity(X)
    n = len(id_list)
    same_cluster_ratios = []

    for i in range(n):
        sims = sim_matrix[i]
        sims[i] = -1  
        topk_idx = np.argsort(sims)[-k:]
        same_cluster = sum(
            clustered_result[str(id_list[i])] == clustered_result[str(id_list[j])]
            for j in topk_idx
        )
        same_cluster_ratios.append(same_cluster / k)

    cluster_counts = defaultdict(int)
    for cid in clustered_result.values():
        cluster_counts[cid] += 1
    cluster_sizes = list(cluster_counts.values())
    avg = np.mean(cluster_sizes)
    std = np.std(cluster_sizes)

    question_same_cluster_ratios = []
    question_vectors = tfidf.transform(questions) 
    question_sim = cosine_similarity(question_vectors, X) 

    for i in range(len(questions)):
        top3_idx = np.argsort(question_sim[i])[-3:]  
        top3_cluster_ids = [clustered_result[str(id_list[j])] for j in top3_idx]
        most_common = max(set(top3_cluster_ids), key=top3_cluster_ids.count)
        count_same = top3_cluster_ids.count(most_common)
        question_same_cluster_ratios.append(count_same / 3)

    return {
        "same_cluster_ratio": sum(same_cluster_ratios) / n,
        "avg": avg,
        "std": std,
        "question_same_cluster_ratio": sum(question_same_cluster_ratios) / len(questions)
    }


