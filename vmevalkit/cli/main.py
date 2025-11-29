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
        config = yaml.load(f)

    dataset_config = config['datasets'][0]
    retriever = Retriever(
        output_dir=dataset_config.get('output_dir', 'data/questions'),
        tasks=dataset_config.get('tasks', None),
        random_seed=dataset_config.get('random_seed', 42),
        pairs_per_domain=dataset_config.get('pairs_per_domain', 5)
    )
    retriever.download_hf_domains()
    retriever.create_regular_dataset()


if __name__ == '__main__':
    main()
