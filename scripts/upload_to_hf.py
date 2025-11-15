#!/usr/bin/env python3
"""Upload VMEvalKit datasets to HuggingFace Hub."""

import os
from pathlib import Path
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def upload_questions_dataset(repo_id: str, private: bool = False):
    """Upload questions dataset to HuggingFace."""
    
    api = HfApi(token=os.getenv("HF_TOKEN"))
    
    # Create repository
    print(f"📦 Creating repository: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True
    )
    
    # Upload questions folder
    print(f"⬆️  Uploading questions dataset...")
    api.upload_folder(
        folder_path="data/questions",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload VMEval questions dataset",
        ignore_patterns=[".DS_Store", "__pycache__", "*.pyc"]
    )
    
    print(f"✅ Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")


def upload_outputs_dataset(repo_id: str, experiment: str = "pilot_experiment", private: bool = True):
    """Upload model outputs (videos) to HuggingFace."""
    
    api = HfApi(token=os.getenv("HF_TOKEN"))
    
    output_path = Path("data/outputs") / experiment
    if not output_path.exists():
        print(f"❌ Output directory not found: {output_path}")
        return
    
    # Create repository
    print(f"📦 Creating repository: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True
    )
    
    # Upload outputs folder
    print(f"⬆️  Uploading outputs from {experiment}...")
    api.upload_folder(
        folder_path=str(output_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {experiment} outputs",
        ignore_patterns=[".DS_Store", "__pycache__", "*.pyc"]
    )
    
    print(f"✅ Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")


def upload_evaluations_dataset(repo_id: str, experiment: str = "pilot_experiment", private: bool = True):
    """Upload evaluation results to HuggingFace."""
    
    api = HfApi(token=os.getenv("HF_TOKEN"))
    
    eval_path = Path("data/evaluations") / experiment
    if not eval_path.exists():
        print(f"❌ Evaluations directory not found: {eval_path}")
        return
    
    # Create repository
    print(f"📦 Creating repository: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True
    )
    
    # Upload evaluations folder
    print(f"⬆️  Uploading evaluations from {experiment}...")
    api.upload_folder(
        folder_path=str(eval_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {experiment} evaluations",
        ignore_patterns=[".DS_Store", "__pycache__", "*.pyc"]
    )
    
    print(f"✅ Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload VMEvalKit datasets to HuggingFace")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID (e.g., username/dataset-name)")
    parser.add_argument("--type", choices=["questions", "outputs", "evaluations"], required=True)
    parser.add_argument("--experiment", default="pilot_experiment", help="Experiment name for outputs/evaluations")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    
    args = parser.parse_args()
    
    if args.type == "questions":
        upload_questions_dataset(args.repo_id, args.private)
    elif args.type == "outputs":
        upload_outputs_dataset(args.repo_id, args.experiment, args.private)
    elif args.type == "evaluations":
        upload_evaluations_dataset(args.repo_id, args.experiment, args.private)

