import os
import yaml
import argparse
from retriever_elasticsearch.retriever.retriever import bm25_retrieve
from src.prep_dataset import prep_dataset
from src.offline import offline
from src.online import online

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default='llama3.2-1b-instruct', choices=['llama3.2-1b-instruct', 'llama3-8b-instruct'])
    parser.add_argument("--dataset", type=str, default='hotpotqa', choices=['hotpotqa', 'popqa', '2wikimultihopqa', 'complexwebquestions'])
    parser.add_argument("--type", type=str, default='bridge')
    parser.add_argument("--mode", type=str, default='prep_dataset')
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--augment_model", type=str, default="llama3.2-1b-instruct")
    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(curr_dir, 'config.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    if args.mode == 'prep_dataset':
        prep_dataset(curr_dir, args.dataset, args.augment_model, args.type, bm25_retrieve, config=config)
    elif args.mode == 'offline':
        offline(curr_dir, args.dataset, args.model_name, args.augment_model, args.type, config)
    elif args.mode == 'online':
        online(curr_dir, args.dataset, args.model_name, args.augment_model, args.type, config, args.k)
    else:
        raise ValueError("Invalid mode")




