# FedMosaic: Federated Retrieval-Augmented Generation via Parametric Adapters

This repository contains the official implementation of our paper:

> **FedMosaic: Federated Retrieval-Augmented Generation via Parametric Adapters**
> *Accepted at SIGIR 2026.*


## Overview

Retrieval-Augmented Generation (RAG) grounds Large Language Models (LLMs) in
external knowledge, but most deployments assume a **centralized corpus**,
which is infeasible in privacy-aware domains where knowledge remains siloed.
This motivates **federated RAG (FedRAG)**, where a central LLM server
collaborates with distributed silos without sharing raw documents.

**FedMosaic** is, to our knowledge, the first federated RAG framework built
on parametric adapters. We follow the **parametric RAG** paradigm introduced
by [PRAG](https://github.com/oneal2000/prag), which encodes each document into
a lightweight adapter that merges with a frozen LLM at inference — so raw
text never leaves its owner. Transplanting this idea to the federated setting
introduces unique challenges (per-document adapters incur heavy storage and
communication, and naively merging many adapters causes destructive
interference), and FedMosaic adapts the parametric design to address them
with two key components:

- **Multi-Document Parametric Adapters.** To reduce storage and communication overhead, FedMosaic
shares one adapter across a cluster of semantically coherent documents. 

- **Selective adapter aggregation.** To mitigate inter-silo adapter interference during adapter averaging, FedMosaic aggregates only adapters associated with the most relevant documents and least conflicting parameters.



## Repository Structure

```
FedMosaic/
├── main.py                       # Unified entry point (prep_dataset / offline / online)
├── config.yaml                   # Hyper-parameters and paths
├── requirements.txt
├── src/
│   ├── prep_dataset.py           # Dataset download, retrieval, augmentation, silo split
│   ├── cluster.py                # Document clustering inside each silo
│   ├── train.py                  # LoRA adapter training + document-specific mask training
│   ├── offline.py                # Offline stage: clustering -> LoRA -> masks
│   ├── online.py                 # Online stage: retrieve -> rank -> masked inference
│   ├── rag.py                    # RAG-related utilities (ranking, inference)
│   ├── silo.py                   # Silo abstraction and per-silo retrieval
│   └── utils.py                  # Models, evaluation, I/O helpers
├── retriever_elasticsearch/      # BM25 retrieval over Wikipedia (Elasticsearch)
│   ├── prep_elastic.py
│   └── retriever/
└── dataset/                      # Prepared datasets (or extract from dataset.tar.gz)
```



## Installation

We recommend Python 3.10.4 with CUDA-enabled PyTorch.

```bash
git clone <this-repo-url> FedMosaic
cd FedMosaic
conda create -n fedmosaic python=3.10.4
pip install -r requirements.txt
```



## Configuration

Before running, edit [config.yaml](config.yaml) to set the following paths to
match your environment:

| Field | Meaning |
| --- | --- |
| `prep_dataset.split.split_model_path` | Local path to the `all-MiniLM-L6-v2` sentence embedding model. |
| `train.save_dir` | Directory used to save trained LoRA adapters and masks. |
| `rank.rank_model_path` | Local path to the `bge-reranker-v2-m3` reranker. |

Additional paths for base LLM checkpoints (e.g. `llama3.2-1b-instruct`,
`llama3-8b-instruct`) are read inside [src/utils.py](src/utils.py) — update
them to point at your local model files.




## Dataset Preparation

FedMosaic is evaluated on four open-domain QA benchmarks: **HotpotQA**,
**2WikiMultiHopQA**, **PopQA**, and **ComplexWebQuestions**.

You have two options.

### Option A — Use the provided archive

```bash
tar -xzvf dataset.tar.gz
```

This produces a `dataset/` directory with the pre-retrieved, augmented, and
silo-split data used in our experiments.

### Option B — Build from scratch

1. Set up BM25 retrieval over the DPR Wikipedia dump with Elasticsearch.
   Follow the instructions in
   [retriever_elasticsearch/README.md](retriever_elasticsearch/README.md)
   to download `psgs_w100.tsv`, install Elasticsearch 8.15.0, and build the
   `wiki` index.

2. Run the preparation pipeline. This downloads the raw QA datasets,
   performs BM25 retrieval, generates rewrites / pseudo-QA augmentations, and
   splits passages into non-IID silos via topic-conditioned Dirichlet sampling:

   ```bash
   python main.py \
       --mode prep_dataset \
       --dataset hotpotqa \
       --type bridge \
       --augment_model llama3.2-1b-instruct
   ```

   Replace `--dataset` with one of `hotpotqa`, `2wikimultihopqa`, `popqa`, or
   `complexwebquestions`, and `--type` with the corresponding question type
   (e.g. `bridge` / `comparison` for HotpotQA).



## Running FedMosaic

FedMosaic follows a two-stage pipeline.

### Stage 1 — Offline: cluster documents and train adapters

Within each silo, documents are clustered by semantic similarity. A shared LoRA
adapter is trained per cluster, after which document-specific binary masks are
learned on top of the frozen cluster LoRA.

```bash
python main.py \
    --mode offline \
    --dataset hotpotqa \
    --type bridge \
    --model_name llama3.2-1b-instruct \
    --augment_model llama3.2-1b-instruct
```

Outputs (cluster LoRAs and masks) are written to `train.save_dir` as
configured in [config.yaml](config.yaml).

### Stage 2 — Online: federated retrieval, ranking, and masked inference

For each question, every silo retrieves its top-`k` passages; the reranker
then selects relevance-aligned, non-conflicting documents, whose corresponding
cluster LoRAs are merged into the frozen LLM together with the document-specific
masks for final generation.

```bash
python main.py \
    --mode online \
    --dataset hotpotqa \
    --type bridge \
    --model_name llama3.2-1b-instruct \
    --augment_model llama3.2-1b-instruct \
    --k 5
```

Predictions and evaluation metrics are written to
`output/fedmosaic/<dataset>/<type>/<model>/...`.



## Supported Models and Datasets

| Flag | Values |
| --- | --- |
| `--model_name` | `llama3.2-1b-instruct`, `llama3-8b-instruct` |
| `--augment_model` | `llama3.2-1b-instruct` (default) |
| `--dataset` | `hotpotqa`, `popqa`, `2wikimultihopqa`, `complexwebquestions` |
| `--type` | dataset-specific question type (e.g. `bridge`, `comparison`, `total`) |
| `--mode` | `prep_dataset`, `offline`, `online` |
| `--k` | retrieval top-`k` per silo (online stage) |



## Acknowledgements

Our retrieval pipeline builds on [PRAG](https://github.com/oneal2000/prag) and
the [DPR](https://github.com/facebookresearch/DPR) Wikipedia dump. We thank
the authors of these projects for releasing their code and data.
