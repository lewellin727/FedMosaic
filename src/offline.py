import os
import time
import json
import random
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from src.utils import *
from src.cluster import *
from src.train import *

def offline(root_dir, dataset_name, model_name, augment_model, dataset_type, config):

   # clustering
   print('Start silo data clustering...')
   silo_data_path = os.path.join(root_dir, f'dataset/{dataset_name}/{dataset_type}/silo.json')
   silo_data = json.load(open(silo_data_path, 'r'))
   _, _ = clustering(root_dir, dataset_name, dataset_type, silo_data, config)

   # training LoRA per cluster
   print(f'Loading clustered documents...')
   C = config['offline']['clustering']['C']
   cluster_path = os.path.join(root_dir, f'dataset/{dataset_name}/{dataset_type}/C={C}/clu.json')
   aug_path = os.path.join(root_dir, f'dataset/{dataset_name}/{dataset_type}/aug.json')
   doc_clusters = load_clustered_doc(cluster_path, aug_path, augment_model)

   model, tokenizer, _ = get_model(model_name)
   save_dir = config['train']['save_dir']
   
   # training clustered LoRA
   lora_epoch = config['train']['lora_epoch']
   lora_save_dir = os.path.join(save_dir, f'fedmosaic/{dataset_name}/{dataset_type}/{model_name}/C={C}le={lora_epoch}/lora')
   print(f'\nStart LoRA params initialization...')
   for idx, doc_cluster in enumerate(doc_clusters):
      print(f'\nTraining {idx} / {len(doc_clusters)} LoRA...')
      train_lora(doc_cluster, model, tokenizer, lora_save_dir, config)
   
   # training masks
   mask_epoch = config['train']['mask_epoch']
   lambda_l1 = config['train']['lambda_l1']
   mask_save_dir = os.path.join(save_dir, f'fedmosaic/{dataset_name}/{dataset_type}/{model_name}/C={C}le={lora_epoch}/mask_me={mask_epoch}lambda_l1={lambda_l1}')
   print(f'\nStart training with masks...')
   for idx, doc_cluster in enumerate(doc_clusters):
      print(f'\nTraining {idx} / {len(doc_clusters)} LoRA with masks...')
      train_mask_with_fixed_lora_cluster(doc_cluster, model, tokenizer, lora_save_dir, mask_save_dir, config)



def load_clustered_doc(cluster_path, aug_path, augment_model):
   cluster = json.load(open(os.path.join(cluster_path), 'r', encoding='utf-8'))
   augment = json.load(open(os.path.join(aug_path), 'r', encoding='utf-8'))

   def get_cluster(c_list, s_id, c_id):
      for c in c_list:
         if c.silo_id == s_id and c.cluster_id == c_id:
            return c
      return None
   
   cluster_list = []
   for silo_id, silo_data in cluster.items():
      for doc_id, cluster_id in silo_data.items():
         passage = id2psg(doc_id, augment)
         aug_item = id2aug(doc_id, augment)
         doc = Document(passage, aug_item, augment_model)
         tar_cluster = get_cluster(cluster_list, silo_id, cluster_id)
         if tar_cluster is None:
            tar_cluster = DocCluster(silo_id, cluster_id, [doc])
            cluster_list.append(tar_cluster)
         else:
            tar_cluster.Documents.append(doc)
   return cluster_list
