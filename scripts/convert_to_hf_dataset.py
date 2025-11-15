#!/usr/bin/env python3
"""Convert VMEvalKit questions to HuggingFace Dataset format."""

import json
from pathlib import Path
from datasets import Dataset, Features, Value, Image, DatasetDict
from dotenv import load_dotenv

load_dotenv()


def convert_questions_to_dataset() -> Dataset:
    """Convert VMEvalKit questions structure to HuggingFace Dataset."""
    
    # Load master manifest
    manifest_path = Path("data/questions/vmeval_dataset.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print(f"📊 Loading {len(manifest['pairs'])} task pairs...")
    
    # Prepare data rows
    data_rows = []
    for pair in manifest["pairs"]:
        task_id = pair["id"]
        domain = pair["domain"]
        
        # Build paths
        base_path = Path("data/questions") / f"{domain}_task" / task_id
        first_frame_path = base_path / "first_frame.png"
        final_frame_path = base_path / "final_frame.png"
        
        # Verify files exist
        if not first_frame_path.exists():
            print(f"⚠️  Missing: {first_frame_path}")
            continue
        if not final_frame_path.exists():
            print(f"⚠️  Missing: {final_frame_path}")
            continue
        
        row = {
            "id": task_id,
            "domain": domain,
            "difficulty": pair.get("difficulty", "unknown"),
            "task_category": pair.get("task_category", "unknown"),
            "prompt": pair["prompt"],
            "first_frame": str(first_frame_path),
            "final_frame": str(final_frame_path),
            "metadata": json.dumps(pair.get("metadata", {})),
            "created_at": pair.get("created_at", ""),
        }
        data_rows.append(row)
    
    print(f"✅ Prepared {len(data_rows)} valid task pairs")
    
    # Define schema with Image types
    features = Features({
        "id": Value("string"),
        "domain": Value("string"),
        "difficulty": Value("string"),
        "task_category": Value("string"),
        "prompt": Value("string"),
        "first_frame": Image(),  # Automatically encodes image
        "final_frame": Image(),  # Automatically encodes image
        "metadata": Value("string"),  # JSON as string
        "created_at": Value("string"),
    })
    
    # Create dataset
    dataset = Dataset.from_dict(
        {key: [row[key] for row in data_rows] for key in data_rows[0].keys()},
        features=features
    )
    
    return dataset


def create_domain_splits(dataset: Dataset) -> DatasetDict:
    """Create dataset splits by cognitive domain."""
    
    splits = {
        "full": dataset,
        "chess": dataset.filter(lambda x: x["domain"] == "chess"),
        "maze": dataset.filter(lambda x: x["domain"] == "maze"),
        "raven": dataset.filter(lambda x: x["domain"] == "raven"),
        "rotation": dataset.filter(lambda x: x["domain"] == "rotation"),
        "sudoku": dataset.filter(lambda x: x["domain"] == "sudoku"),
    }
    
    return DatasetDict(splits)


def upload_to_hub(repo_id: str, private: bool = False):
    """Convert and upload dataset to HuggingFace Hub."""
    
    print("🔄 Converting VMEvalKit dataset to HuggingFace format...")
    dataset = convert_questions_to_dataset()
    
    print("📦 Creating domain splits...")
    dataset_dict = create_domain_splits(dataset)
    
    # Print statistics
    print("\n📊 Dataset Statistics:")
    for split_name, split_data in dataset_dict.items():
        print(f"  {split_name}: {len(split_data)} tasks")
    
    print(f"\n⬆️  Uploading to {repo_id}...")
    dataset_dict.push_to_hub(
        repo_id=repo_id,
        private=private,
        commit_message="Upload VMEval questions dataset"
    )
    
    print(f"✅ Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert and upload VMEvalKit dataset to HuggingFace")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    
    args = parser.parse_args()
    
    upload_to_hub(args.repo_id, args.private)

