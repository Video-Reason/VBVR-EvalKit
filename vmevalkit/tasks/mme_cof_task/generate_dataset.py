#!/usr/bin/env python3
"""
Generate MME-CoF Dataset with Solutions

Downloads the original MME-CoF dataset, generates solution images using Gemini,
and creates a complete dataset in VMEvalKit format for upload to HuggingFace.

Usage:
    python generate_dataset.py --output-dir ./data/mme_cof_generated --use-imagen
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import directly to avoid loading all task modules
import importlib.util

# Load solution_generator module directly
spec = importlib.util.spec_from_file_location(
    "solution_generator",
    Path(__file__).parent / "solution_generator.py"
)
solution_generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution_generator)
generate_solution_image = solution_generator.generate_solution_image

# Load PROMPTS module directly
spec = importlib.util.spec_from_file_location(
    "PROMPTS",
    Path(__file__).parent / "PROMPTS.py"
)
prompts_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompts_module)
get_prompt_for_category = prompts_module.get_prompt_for_category
get_category_description = prompts_module.get_category_description


def download_mme_cof():
    """Download the original MME-CoF dataset from HuggingFace."""
    from datasets import load_dataset
    
    print("📥 Downloading MME-CoF dataset from HuggingFace...")
    dataset = load_dataset("ZiyuG/MME-CoF", split="train")
    print(f"✅ Downloaded {len(dataset)} tasks")
    
    # Get label names from dataset features
    if hasattr(dataset.features['label'], 'names'):
        label_names = dataset.features['label'].names
        print(f"   Found {len(label_names)} categories: {', '.join(label_names)}")
        
        # Convert integer labels to string category names
        def map_label(example):
            example['category'] = label_names[example['label']]
            return example
        
        dataset = dataset.map(map_label)
    else:
        # Fallback: label is already a string
        dataset = dataset.map(lambda x: {'category': x['label']})
    
    return dataset


def generate_solutions_for_dataset(
    dataset, 
    output_dir: Path,
    use_imagen: bool = True,
    skip_existing: bool = True
):
    """
    Generate solution images for all tasks in the dataset.
    
    Args:
        dataset: HuggingFace dataset
        output_dir: Directory to save generated dataset
        use_imagen: Whether to use Imagen for image generation
        skip_existing: Skip tasks that already have solutions
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    stats = {
        "total": len(dataset),
        "generated": 0,
        "skipped": 0,
        "failed": 0,
        "by_category": {}
    }
    
    tasks_data = []
    
    print(f"\n🎬 Generating solution images for {len(dataset)} tasks...")
    print(f"   Output directory: {output_dir}")
    print(f"   Using Imagen: {use_imagen}")
    print()
    
    for idx, item in enumerate(tqdm(dataset, desc="Generating solutions")):
        category = item['category']  # Use mapped category string
        image = item['image']
        
        task_id = f"mme_cof_{idx:04d}"
        task_dir = output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Save first frame (ensure it's a PIL Image)
        from PIL import Image as PILImage
        if not isinstance(image, PILImage.Image):
            # Convert to PIL Image if it's not already
            if hasattr(image, '__array__'):
                import numpy as np
                image = PILImage.fromarray(np.array(image))
            else:
                print(f"      ⚠️  Cannot convert image to PIL format for {task_id}")
                continue
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        first_frame_path = task_dir / "first_frame.png"
        image.save(first_frame_path, format="PNG")
        
        # Check if solution already exists
        final_frame_path = task_dir / "final_frame.png"
        if skip_existing and final_frame_path.exists():
            print(f"   ⏭️  Skipping {task_id} (solution exists)")
            stats['skipped'] += 1
            
            # Load existing metadata
            metadata_path = task_dir / "question_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    task_data = json.load(f)
                    tasks_data.append(task_data)
            continue
        
        # Generate prompt
        prompt = get_prompt_for_category(category)
        prompt_path = task_dir / "prompt.txt"
        prompt_path.write_text(prompt)
        
        # Generate solution image
        print(f"\n🔄 [{idx+1}/{len(dataset)}] Generating solution for {task_id} ({category})...")
        
        solution_image = generate_solution_image(
            image, 
            category, 
            metadata={'id': task_id, 'category': category},
            use_imagen=use_imagen
        )
        
        if solution_image:
            # Save solution image
            solution_image.save(final_frame_path, format="PNG")
            print(f"   ✅ Solution saved: {final_frame_path}")
            stats['generated'] += 1
            
            # Update category stats
            if category not in stats['by_category']:
                stats['by_category'][category] = {'generated': 0, 'failed': 0}
            stats['by_category'][category]['generated'] += 1
        else:
            print(f"   ❌ Failed to generate solution for {task_id}")
            stats['failed'] += 1
            if category not in stats['by_category']:
                stats['by_category'][category] = {'generated': 0, 'failed': 0}
            stats['by_category'][category]['failed'] += 1
        
        # Create metadata
        task_metadata = {
            "id": task_id,
            "domain": "mme_cof",
            "category": category,
            "category_description": get_category_description(category),
            "prompt": prompt,
            "first_image_path": f"{task_id}/first_frame.png",
            "final_image_path": f"{task_id}/final_frame.png" if solution_image else None,
            "created_at": datetime.now().isoformat() + 'Z',
            "source": "ZiyuG/MME-CoF",
            "solution_generated_with": "Gemini 2.0 Flash + Imagen 3" if use_imagen else "Gemini 2.0 Flash (annotated)",
            "original_idx": idx
        }
        
        # Save metadata
        metadata_path = task_dir / "question_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(task_metadata, f, indent=2, default=str)
        
        tasks_data.append(task_metadata)
    
    # Save dataset summary
    dataset_summary = {
        "name": "mme_cof_vmeval_format",
        "description": "MME-CoF dataset converted to VMEvalKit format with LLM-generated solutions",
        "total_tasks": len(dataset),
        "generated_at": datetime.now().isoformat() + 'Z',
        "statistics": stats,
        "tasks": tasks_data
    }
    
    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(dataset_summary, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("📊 Generation Statistics:")
    print("=" * 70)
    print(f"Total tasks:      {stats['total']}")
    print(f"✅ Generated:     {stats['generated']}")
    print(f"⏭️  Skipped:       {stats['skipped']}")
    print(f"❌ Failed:        {stats['failed']}")
    print()
    print("By Category:")
    for category, cat_stats in sorted(stats['by_category'].items()):
        print(f"  {category:30} → ✅ {cat_stats['generated']:2d}  ❌ {cat_stats['failed']:2d}")
    print("=" * 70)
    print(f"\n💾 Dataset saved to: {output_dir}")
    print(f"📄 Summary saved to: {summary_path}")
    
    return dataset_summary


def create_huggingface_dataset(source_dir: Path, hf_dataset_name: str = None):
    """
    Upload the generated dataset to HuggingFace.
    
    Args:
        source_dir: Directory containing the generated dataset
        hf_dataset_name: HuggingFace dataset name (e.g., "YourOrg/MME-CoF-VMEval")
    """
    from datasets import Dataset, Features, Image as HFImage, Value
    from huggingface_hub import HfApi
    
    print("\n📤 Preparing dataset for HuggingFace upload...")
    
    # Load all tasks
    tasks = []
    source_dir = Path(source_dir)
    
    for task_dir in sorted(source_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith('.'):
            continue
        
        metadata_path = task_dir / "question_metadata.json"
        if not metadata_path.exists():
            continue
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        first_frame_path = task_dir / "first_frame.png"
        final_frame_path = task_dir / "final_frame.png"
        prompt_path = task_dir / "prompt.txt"
        
        if not first_frame_path.exists():
            continue
        
        task_data = {
            "id": metadata['id'],
            "image": str(first_frame_path),
            "solution_image": str(final_frame_path) if final_frame_path.exists() else None,
            "prompt": prompt_path.read_text() if prompt_path.exists() else metadata.get('prompt', ''),
            "category": metadata.get('category', ''),
            "category_description": metadata.get('category_description', ''),
        }
        
        tasks.append(task_data)
    
    print(f"   Found {len(tasks)} tasks")
    
    # Create HuggingFace dataset
    features = Features({
        "id": Value("string"),
        "image": HFImage(),
        "solution_image": HFImage(),
        "prompt": Value("string"),
        "category": Value("string"),
        "category_description": Value("string"),
    })
    
    hf_dataset = Dataset.from_list(tasks, features=features)
    
    print(f"   Created dataset with {len(hf_dataset)} examples")
    
    if hf_dataset_name:
        print(f"\n📤 Uploading to HuggingFace: {hf_dataset_name}")
        hf_dataset.push_to_hub(
            hf_dataset_name,
            private=False,
            commit_message="Add MME-CoF dataset in VMEvalKit format with LLM-generated solutions"
        )
        print(f"✅ Successfully uploaded to: https://huggingface.co/datasets/{hf_dataset_name}")
    else:
        print("\n💡 To upload to HuggingFace, run with --hf-dataset-name YourOrg/MME-CoF-VMEval")
    
    return hf_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Generate MME-CoF dataset with LLM-generated solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate with Imagen (full pipeline)
    python generate_dataset.py --output-dir ./data/mme_cof_generated --use-imagen
    
    # Generate with text annotations only (faster, cheaper)
    python generate_dataset.py --output-dir ./data/mme_cof_generated --no-imagen
    
    # Upload to HuggingFace after generation
    python generate_dataset.py --output-dir ./data/mme_cof_generated --use-imagen \\
        --upload --hf-dataset-name YourOrg/MME-CoF-VMEval
    
Environment Variables:
    GEMINI_API_KEY or GOOGLE_API_KEY: Required for solution generation
        """
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/mme_cof_generated",
        help="Output directory for generated dataset (default: ./data/mme_cof_generated)"
    )
    
    parser.add_argument(
        "--use-imagen",
        action="store_true",
        help="Use Imagen 3 to generate solution images (default: False, uses annotated images)"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip tasks that already have solutions (default: True)"
    )
    
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the generated dataset to HuggingFace"
    )
    
    parser.add_argument(
        "--hf-dataset-name",
        type=str,
        help="HuggingFace dataset name (e.g., 'YourOrg/MME-CoF-VMEval')"
    )
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY') and not os.getenv('GOOGLE_API_KEY'):
        print("❌ Error: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        print("   Please set your API key:")
        print("   export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)
    
    print("=" * 70)
    print("🎬 MME-CoF Dataset Generation Pipeline")
    print("=" * 70)
    
    # Step 1: Download original dataset
    dataset = download_mme_cof()
    
    # Step 2: Generate solutions
    dataset_summary = generate_solutions_for_dataset(
        dataset,
        output_dir=args.output_dir,
        use_imagen=args.use_imagen,
        skip_existing=args.skip_existing
    )
    
    # Step 3: Upload to HuggingFace (optional)
    if args.upload:
        create_huggingface_dataset(
            source_dir=args.output_dir,
            hf_dataset_name=args.hf_dataset_name
        )
    
    print("\n🎉 Dataset generation complete!")
    print(f"\n📁 Local dataset: {args.output_dir}")
    if args.upload and args.hf_dataset_name:
        print(f"🌐 HuggingFace: https://huggingface.co/datasets/{args.hf_dataset_name}")


if __name__ == "__main__":
    main()

