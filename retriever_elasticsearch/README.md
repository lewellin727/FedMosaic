
### Prepare client corpus for FedRAG

Reference [PRAG](https://github.com/oneal2000/prag)
#### Prepare BM25 for retrieval
1. Download the Wikipedia dump from the [DPR repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) using the following command

```bash
mkdir -p dpr
wget -O dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd dpr
gzip -d psgs_w100.tsv.gz
```

2. Use Elasticsearch to index the Wikipedia dump

```bash
cd ..
wget -O elasticsearch-8.15.0.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-8.15.0.tar.gz
rm elasticsearch-8.15.0.tar.gz 
```

Enter the `elasticsearch-8.15.0/config/elasticsearch.yml` file and replace the following:
1. Customize the index's data and logs paths. Ensure the directories exist and have write permissions.
2. Disable security verification.
```bash
# Custom Path
path.data: path-to-index-data
path.logs: path-to-index-logs
# Turn off security verification. Be careful to eliminate existing content in the yml file and do not redefine it.
xpack.security.enabled: false
xpack.security.enrollment.enabled: false
xpack.security.http.ssl.enabled: false
xpack.security.transport.ssl.enabled: false
```

Start elasticsearch and create indexes
```bash
cd elasticsearch-8.15.0
nohup bin/elasticsearch &      # run Elasticsearch in background
curl -k http://localhost:9200  # check Elasticsearch status
cd ..
python prep_elastic.py --data_path dpr/psgs_w100.tsv --index_name wiki  # build index
```

