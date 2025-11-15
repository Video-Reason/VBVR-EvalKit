#!/usr/bin/env python3
"""Download VMEvalKit datasets from HuggingFace Hub."""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def download_questions_dataset(repo_id: str):
    """Download questions dataset from HuggingFace."""
    
    target_dir = Path("data/questions")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"⬇️  Downloading questions dataset from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        token=os.getenv("HF_TOKEN"),
    )
    
    print(f"✅ Downloaded to: {target_dir}")


def download_outputs_dataset(repo_id: str, experiment: str = "pilot_experiment"):
    """Download model outputs from HuggingFace."""
    
    target_dir = Path("data/outputs") / experiment
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"⬇️  Downloading outputs from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        token=os.getenv("HF_TOKEN"),
    )
    
    print(f"✅ Downloaded to: {target_dir}")


def download_evaluations_dataset(repo_id: str, experiment: str = "pilot_experiment"):
    """Download evaluations from HuggingFace."""
    
    target_dir = Path("data/evaluations") / experiment
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"⬇️  Downloading evaluations from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        token=os.getenv("HF_TOKEN"),
    )
    
    print(f"✅ Downloaded to: {target_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download VMEvalKit datasets from HuggingFace")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID")
    parser.add_argument("--type", choices=["questions", "outputs", "evaluations"], required=True)
    parser.add_argument("--experiment", default="pilot_experiment", help="Experiment name for outputs/evaluations")
    
    args = parser.parse_args()
    
    if args.type == "questions":
        download_questions_dataset(args.repo_id)
    elif args.type == "outputs":
        download_outputs_dataset(args.repo_id, args.experiment)
    elif args.type == "evaluations":
        download_evaluations_dataset(args.repo_id, args.experiment)

