import argparse
import sys
import yaml
from pathlib import Path
from vmevalkit.runner.retriever import Retriever

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', help='Train config file path')
    return parser.parse_args()

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    dataset_config_list = config_dict['datasets']
    # 创建多个retriever, 对于每个retriever 传入dataset_config
    for dataset_config in dataset_config_list:
        retriever = Retriever(
            dataset_config=dataset_config,
        )
        retriever.retrieve_tasks()



if __name__ == '__main__':
    main()
