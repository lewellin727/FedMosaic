import os
import json
from tqdm import tqdm
import random
import torch
import time
import ast
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.utils import *

def prep_dataset(root_dir, dataset_name, augment_model, dataset_type, retriever, config):
    raw_dataset_dir = os.path.join(root_dir, f'dataset/raw_data/{dataset_name}')
    download_dataset(raw_dataset_dir, dataset_name, dataset_type)
    
    
    dataset_path = os.path.join(root_dir, f'dataset/{dataset_name}/{dataset_type}')
    os.makedirs(dataset_path, exist_ok=True)
    base_dataset_path = os.path.join(root_dir, f"{raw_dataset_dir}/{dataset_type}.json")
    base_dataset = json.load(open(base_dataset_path, 'r'))

    print(f'Retreval step...')
    src_list = retrieve_data(base_dataset, retriever, dataset_path, config)
    text_list = []
    for src_item in src_list:
        text_list.extend(src_item['passages'])
    
    print('Start data augment...')
    aug_save_path = os.path.join(root_dir, f'dataset/{dataset_name}/{dataset_type}/aug.json')
    os.makedirs(os.path.dirname(aug_save_path), exist_ok=True)
    aug_list = augment(src_list, augment_model, aug_save_path, config)

    print('Start silo data split...')    
    silo_data = split_silo(text_list, dataset_path, config)
    
    print('Prepared dataset end')


def load_dataset(root_dir, dataset_name, dataset_type, augment_model, dataset_path):
    print('Loading dataset...')
    data_path = os.path.join(root_dir, f'data_src/data_aug/{dataset_name}/{augment_model}/{dataset_type}.json')
    dataset = json.load(open(data_path, 'r'))
    src_list, text_list, aug_list = [], [], []
    for data in dataset:
        ss = data.copy()
        ss.pop('augment', None)
        src_list.append(ss)
        text_list.extend(data['passages'])
    text_list = [{"id": idx, "passage": t} for idx, t in enumerate(text_list)]
    for item in tqdm(text_list):
        aug = item.copy()
        ppp = item['passage']
        find = False
        for i in dataset:
            for idx, psg in enumerate(i['passages']):
                if ppp == psg:
                    aug[f'{augment_model}_rewrite'] = i['augment'][idx][f'{augment_model}_rewrite']
                    aug[f'{augment_model}_qa'] = i['augment'][idx][f'{augment_model}_qa']
                    find = True
                    break
            if find:
                break
        if not find:
            ValueError(f'{ppp} not found in dataset')
        aug_list.append(aug)
    json.dump(src_list, open(os.path.join(dataset_path, 'src.json'), 'w'), indent=4)
    json.dump(text_list, open(os.path.join(dataset_path, 'text.json'), 'w'), indent=4)
    json.dump(aug_list, open(os.path.join(dataset_path, 'aug.json'), 'w'), indent=4)
    print(f'Save src_list, text_list, aug_list to {dataset_path}')
    return src_list, text_list, aug_list
    
def retrieve_data(base_dataset, retriever, dataset_path, config):
    num_samples = config['prep_dataset']['retrieve']['sample_size']
    base_dataset = base_dataset[:num_samples]

    text_id = 0
    if not os.path.exists(os.path.join(dataset_path, 'src.json')):
        src_list = []
        for data in tqdm(base_dataset):
            ss = data.copy()
            ss.pop('augment', None)
            text = retriever(data['question'], topk=config['prep_dataset']['retrieve']['retrieve_size'])
            text = [{"id": idx + text_id, "passage": t} for idx, t in enumerate(text)]
            text_id += len(text)
            ss['passages'] = text
            src_list.append(ss)
        json.dump(src_list, open(os.path.join(dataset_path, 'src.json'), 'w'), indent=4)
        print(f"Save src and text at {dataset_path}")
    else:
        print("file already exists, skip")
        src_list = json.load(open(os.path.join(dataset_path, 'src.json'), 'r'))
    return src_list

def split_silo(text_list, dataset_path, config):
    silo_size = config['prep_dataset']['split']['silo_size']
    seed = config['prep_dataset']['split']['random_seed']
    alpha = config['prep_dataset']['split']['alpha']
    split_type = config['prep_dataset']['split']['split_type']
    split_model = config['prep_dataset']['split']['split_model']
    split_model_path = config['prep_dataset']['split']['split_model_path']

    random.seed(seed)
    np.random.seed(seed)

    passages = [doc["passage"] for doc in text_list]
    if split_type == 'dense':
        model = SentenceTransformer(split_model_path)
        features = model.encode(passages, normalize_embeddings=True)
    else:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        features = vectorizer.fit_transform(passages).toarray()

    print(f"Clustering into {silo_size} topics using KMeans...")
    kmeans = KMeans(n_clusters=silo_size, random_state=seed, n_init=10)
    topic_labels = kmeans.fit_predict(features)

    print("Assigning to silos via Dirichlet within each topic...")
    silo_data = {i: [] for i in range(silo_size)}
    for topic_id in range(silo_size):
        topic_indices = [i for i, label in enumerate(topic_labels) if label == topic_id]
        topic_size = len(topic_indices)
        if topic_size == 0:
            continue

        np.random.shuffle(topic_indices)
        proportions = dirichlet(alpha=np.repeat(alpha, silo_size)).rvs(1)[0]
        cut_points = (np.cumsum(proportions)[:-1] * topic_size).astype(int)
        splits = np.split(np.array(topic_indices), cut_points)
        for client_id, idx_group in enumerate(splits):
            for idx in idx_group:
                silo_data[client_id].append(text_list[int(idx)]["id"])

    for k, v in silo_data.items():
        print(f'silo {k}: get {len(v)} passages')
    
    json.dump(silo_data, open(os.path.join(dataset_path, "silo.json"), "w"), indent=4)
    print(f'Save silo data to {os.path.join(dataset_path, "silo.json")}')
    
    return silo_data

def augment(src_list, augment_model, aug_save_path, config):
    print(f"Augmenting {len(src_list)} src...")
    model, tokenizer, _ = get_model(augment_model)
    aug_generation_config = dict(
        max_new_tokens=config['prep_dataset']['augment']['max_new_tokens'],
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        temperature=config['prep_dataset']['augment']['temperature'],
        top_k=config['prep_dataset']['augment']['top_k'],
    )

    aug_list = []
    for idx, src in enumerate(src_list):
        print(f"Generating augment for {idx} / {len(src_list)} src texts...")
        aug = src.copy()
        aug['augments'] = []
        with torch.no_grad():
            passages = src['passages']
            for psg_item in passages:
                begin_time = time.time()
                aug_item = {}
                aug_item['id'] = psg_item['id']

                aug_item[f"{augment_model}_rewrite"] = get_rewrite(psg_item['passage'], model, tokenizer, aug_generation_config)
                qa = get_qa(psg_item['passage'], augment_model, model, tokenizer, aug_generation_config)
                if fix_qa(qa)[0] == False: 
                    aug_item[f"{augment_model}_qa"] = None
                else:
                    aug_item[f"{augment_model}_qa"] = qa
                    
                aug['augments'].append(aug_item)
                end_time = time.time()
                print(f'Augmentation time cost: {end_time - begin_time}s')
        aug_list.append(aug)

        if len(aug_list) % 10 == 0:
            json.dump(aug_list, open(aug_save_path, 'w'), indent=4)
            
    json.dump(aug_list, open(aug_save_path, 'w'), indent=4)
    return aug_list


HOTPOT_URLS = ["http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"]
HOTPOT_NAMES = ["hotpot_dev_distractor_v1.json"]

WIKIMULTIHOP_URL = "voidful/2WikiMultihopQA"
WIKIMULTIHOP_NAMES = ['train.json', 'dev.json']

POPQA_URL = "https://raw.githubusercontent.com/AlexTMallen/adaptive-retrieval/main/data/popQA.tsv"
POPQA_TSV = "popQA.tsv"

COMPLEXWEBQUESTIONS_URL = "https://www.dropbox.com/s/440f4096rkeo7xc/Data.zip"
COMPLEX_NAMES = ['ComplexWebQuestions_train.json', 'ComplexWebQuestions_dev.json']

def _filter_and_sample(items, dataset_name, dataset_type, type_key, sample_num):
    dataset_type = dataset_type or 'total'
    if dataset_type != 'total':
        items = [i for i in items if i[type_key] == dataset_type]
    if not items:
        raise ValueError(f"No {dataset_type} in Dataset {dataset_name}")
    return random.sample(items, sample_num)


def _load_jsons(download_dir, names):
    items = []
    for name in names:
        items.extend(json.load(open(os.path.join(download_dir, name), 'r')))
    return items


def _all_exist(download_dir, names):
    return all(os.path.exists(os.path.join(download_dir, n)) for n in names)


def _ensure_download(download_dir, names, download_cmd):
    if _all_exist(download_dir, names):
        print(f"Already downloaded in {download_dir}, skip")
        return
    print("Download the raw dataset")
    os.system(download_cmd)


def download_dataset(raw_dataset_dir, dataset_name, dataset_type, sample_num=30, seed=42):
    random.seed(seed)
    download_dir = os.path.join(raw_dataset_dir, "cache")
    os.makedirs(download_dir, exist_ok=True)

    if dataset_name == "hotpotqa":
        for url, name in zip(HOTPOT_URLS, HOTPOT_NAMES):
            _ensure_download(download_dir, [name], f"wget -c {url} -P {download_dir}")
        tmp = _filter_and_sample(
            _load_jsons(download_dir, HOTPOT_NAMES),
            dataset_name, dataset_type, "type", sample_num,
        )
        sample_dataset = [
            {"id": idx, "question": it["question"], "answer": [it["answer"]]}
            for idx, it in enumerate(tmp)
        ]

    elif dataset_name == "2wikimultihopqa":
        _ensure_download(
            download_dir, WIKIMULTIHOP_NAMES,
            f"modelscope download --dataset {WIKIMULTIHOP_URL} {' '.join(WIKIMULTIHOP_NAMES)} --local_dir {download_dir}",
        )
        tmp = _filter_and_sample(
            _load_jsons(download_dir, WIKIMULTIHOP_NAMES),
            dataset_name, dataset_type, "type", sample_num,
        )
        sample_dataset = [
            {"id": idx, "question": it["question"], "answer": [it["answer"]]}
            for idx, it in enumerate(tmp)
        ]

    elif dataset_name == "popqa":
        _ensure_download(download_dir, [POPQA_TSV], f"wget -c {POPQA_URL} -P {download_dir}")
        df = pd.read_csv(os.path.join(download_dir, POPQA_TSV), sep='\t')
        sampled = random.sample(list(zip(df['question'], df['possible_answers'])), sample_num)
        sample_dataset = [
            {"id": idx, "question": q, "answer": ast.literal_eval(a)}
            for idx, (q, a) in enumerate(sampled)
        ]

    elif dataset_name == "complexwebquestions":
        if not _all_exist(download_dir, COMPLEX_NAMES):
            print("Download the raw dataset")
            os.system(f"wget -c {COMPLEXWEBQUESTIONS_URL} -P {download_dir}")
            zip_path = os.path.join(download_dir, "Data.zip")
            os.system(f"unzip -o {zip_path} -d {download_dir}")
        else:
            print(f"Already downloaded in {download_dir}, skip")
        tmp = _filter_and_sample(
            _load_jsons(f"{download_dir}/Data/complex_web_questions", COMPLEX_NAMES),
            dataset_name, dataset_type, "compositionality_type", sample_num,
        )
        sample_dataset = []
        for idx, it in enumerate(tmp):
            answer = []
            for aa in it["answers"]:
                answer.append(aa['answer'])
                answer.extend(aa['aliases'])
            sample_dataset.append({"id": idx, "question": it["question"], "answer": answer})

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    raw_dataset_path = os.path.join(raw_dataset_dir, f"{dataset_type}.json")
    json.dump(sample_dataset, open(raw_dataset_path, 'w'), indent=4)
    print(f"Raw dataset saved to {raw_dataset_path}")