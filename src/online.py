import os
import json
import torch
import numpy as np
from src.train import *
from src.utils import *
from src.rag import *
from src.silo import *

def id2lora(doc_id, clu_data, lora_save_dir):
    for silo_id, cluster in clu_data.items():
        for doc, clu_id in cluster.items():
            if str(doc_id) == str(doc):
                return os.path.join(lora_save_dir, f'silo={silo_id}_clu={clu_id}')
    raise ValueError(f'{doc_id} not found in cluster')



def online(root_dir, dataset_name, model_name, augment_model, dataset_type, config, k=5):
    save_dir = config['train']['save_dir']
    C = config['offline']['clustering']['C']
    le = config['train']['lora_epoch']
    me = config['train']['mask_epoch']
    lambda_l1 = config['train']['lambda_l1']
    lora_save_dir = os.path.join(save_dir, f'fedmosaic/{dataset_name}/{dataset_type}/{model_name}/C={C}le={le}/lora')
    mask_save_dir = os.path.join(save_dir, f'fedmosaic/{dataset_name}/{dataset_type}/{model_name}/C={C}le={le}/mask_me={me}lambda_l1={lambda_l1}')

    data_path = os.path.join(root_dir, f'dataset/{dataset_name}/{dataset_type}')
    aug_data = json.load(open(os.path.join(data_path, 'aug.json'), 'r', encoding='utf-8'))
    clu_data = json.load(open(os.path.join(data_path, f'C={C}/clu.json'), 'r', encoding='utf-8'))
    silo_data = json.load(open(os.path.join(data_path, 'silo.json'), 'r', encoding='utf-8'))

    base_model, tokenizer, generation_config = get_model(model_name)
    from FlagEmbedding import FlagReranker
    reranker = FlagReranker(
        config['rank']['rank_model_path'], 
        use_fp16=True,
        devices="cuda:0"
    )

    questions = [d['question'] for d in aug_data]
    answers = [d['answer'] for d in aug_data]
    silos = init_silos(silo_data, aug_data, augment_model)
    # K = config['rank']['retrieval_K']
    K = k

    results = []
    for idx in range(len(questions)):
        print(f'\nInferencing {idx + 1} / {len(questions)} questions')

        print(f'Retrieving from silos...')
        documents = []
        for silo in silos:
            documents.extend(silo.retrieve(questions[idx], k=K))
        
        print(f'Ranking target docs...')
        doc_ids, doc_scores = fedranking(questions[idx], documents, mask_save_dir, config, reranker)
        
        print(f'Selected doc cnt: {len(doc_ids)}')

        lora_paths = [id2lora(doc_id, clu_data, lora_save_dir) for doc_id in doc_ids]
        mask_paths = [os.path.join(mask_save_dir, f"doc_id={doc_id}.pt") for doc_id in doc_ids]
        pred = inference_with_mask(questions[idx], doc_scores, base_model, lora_paths, mask_paths, tokenizer, generation_config)

        # pred = inference(questions[idx], base_model, lora_paths, tokenizer, generation_config)

        results.append(pred)
        print(f'Question {idx + 1} get  answer: {pred}')
        print(f'Question {idx + 1} true answer: {answers[idx]}')
    
    lambda_ol = config['rank']['lambda_ol']
    result_eval_dir = os.path.join(root_dir, f'output/fedmosaic/{dataset_name}/{dataset_type}/{model_name}', 
                                   f'C={C}le={le}me={me}K={K}lamb={lambda_ol}lambda_l1={lambda_l1}')
    os.makedirs(result_eval_dir, exist_ok=True)
    result_save_path = os.path.join(result_eval_dir, f'result.json')
    json.dump(results, open(result_save_path, 'w'), indent=4)


    # ## evaluation
    eval_result = []
    assert len(questions) == len(answers)
    assert len(questions) == len(results)
    for i in range(len(results)):
        eval_result.append(evaluate(results[i], answers[i]))
    avg = {k: sum(float(d[k]) for d in eval_result) / len(eval_result) for k in eval_result[0]}
    print(f'Eval avg: {avg}')
    eval_save_path = os.path.join(result_eval_dir, f'eval.json')
    os.makedirs(os.path.dirname(eval_save_path), exist_ok=True)
    json.dump(avg, open(eval_save_path, 'w'), indent=4)
    print(f'Eval save to {eval_save_path}')
