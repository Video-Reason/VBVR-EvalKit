#!/usr/bin/env python3
"""
Evaluation script for VBVR-Bench with flexible folder structure support.

Supports model video directories structured as:
    model_path/
    ├── In-Domain_50/
    │   ├── G-45_key_door_matching_data-generator/
    │   │   ├── 00000.mp4
    │   │   ├── 00001.mp4
    │   │   └── ...
    │   └── ...
    └── Out-of-Domain_50/
        ├── G-135_select_next_figure_.../
        │   ├── 00000.mp4
        │   └── ...
        └── ...

Also supports checkpoint subdirectories:
    model_path/
    ├── checkpoint-100/
    │   ├── In-Domain_50/
    │   └── Out-of-Domain_50/
    └── checkpoint-250/
        ├── In-Domain_50/
        └── Out-of-Domain_50/

Usage:
    # Evaluate a single model
    python run_evaluation.py --model_path /path/to/model_videos/65frames_65frames

    # Evaluate all models under a base directory
    python run_evaluation.py --models_base /path/to/model_videos

    # Evaluate specific models by name
    python run_evaluation.py --models_base /path/to/model_videos --models 65frames_65frames chronoedit

    # Evaluate with checkpoints (auto-detected)
    python run_evaluation.py --model_path /path/to/model_with_checkpoints
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np

import re
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Add VBVR-Bench to path
sys.path.insert(0, str(Path(__file__).parent))

from vbvr_bench.evaluators import (
    get_evaluator, TASK_EVALUATOR_MAP, get_task_category,
    is_out_of_domain, get_split,
)


def find_gt_info(task_name: str, video_idx: int, gt_base_path: str) -> dict:
    """Find GT video, first frame, final frame for reference.

    Expects GT directory structure:
      gt_base_path/In-Domain_50/{task_name}/{idx}/ground_truth.mp4
      gt_base_path/Out-of-Domain_50/{task_name}/{idx}/ground_truth.mp4
    """
    gt_info = {
        'gt_video_path': None,
        'gt_first_frame': None,
        'gt_final_frame': None,
        'prompt': None
    }

    # Search GT sub-folders
    for folder in ['In-Domain_50', 'Out-of-Domain_50']:
        task_path = os.path.join(gt_base_path, folder, task_name)
        if os.path.exists(task_path):
            sample_folder = os.path.join(task_path, f'{video_idx:05d}')
            if os.path.exists(sample_folder):
                gt_video = os.path.join(sample_folder, 'ground_truth.mp4')
                if os.path.exists(gt_video):
                    gt_info['gt_video_path'] = gt_video

                first_frame = os.path.join(sample_folder, 'first_frame.png')
                if os.path.exists(first_frame):
                    gt_info['gt_first_frame'] = first_frame

                final_frame = os.path.join(sample_folder, 'final_frame.png')
                if os.path.exists(final_frame):
                    gt_info['gt_final_frame'] = final_frame

                prompt_file = os.path.join(sample_folder, 'prompt.txt')
                if os.path.exists(prompt_file):
                    with open(prompt_file, 'r') as f:
                        gt_info['prompt'] = f.read().strip()

                return gt_info

    return gt_info


def evaluate_single_video(video_path: str, task_name: str, gt_info: dict = None, device: str = 'cuda'):
    """Evaluate a single video using task-specific scoring only."""
    try:
        evaluator = get_evaluator(task_name, device)

        eval_info = {
            'video_path': video_path,
            'task_name': task_name,
            'no_ssim_fallback': True,
        }

        if gt_info:
            eval_info.update(gt_info)

        result = evaluator.evaluate(eval_info, task_specific_only=True)
        return result
    except Exception as e:
        print(f"Error evaluating {video_path}: {e}")
        traceback.print_exc()
        return {
            'score': 0.0,
            'error': str(e),
            'dimensions': {}
        }


def detect_folder_structure(model_path: str) -> str:
    """
    Detect the folder structure of a model video directory.

    Returns:
        'in_out_domain': In-Domain_50 / Out-of-Domain_50 structure
        'checkpoints':   Contains checkpoint-XXX subdirectories
        'unknown':       Unknown structure
    """
    entries = os.listdir(model_path)

    if any(e.startswith('checkpoint-') for e in entries):
        return 'checkpoints'
    elif 'In-Domain_50' in entries or 'Out-of-Domain_50' in entries:
        return 'in_out_domain'
    else:
        return 'unknown'


# Folder name → logical split mapping
FOLDER_TO_SPLIT = {
    'In-Domain_50': 'In_Domain',
    'Out-of-Domain_50': 'Out_of_Domain',
}


def collect_videos(model_path: str) -> list:
    """
    Collect all video files from a model directory, supporting multiple folder structures.

    Returns list of dicts with video_path, task_name, video_file, video_idx, split, category, gt_info, folder.
    """
    structure = detect_folder_structure(model_path)

    if structure != 'in_out_domain':
        print(f"Warning: Unknown folder structure in {model_path}. "
              f"Expected In-Domain_50/ and Out-of-Domain_50/ subdirectories.")
        return []

    folders = ['In-Domain_50', 'Out-of-Domain_50']
    all_videos = []

    for folder in folders:
        folder_path = os.path.join(model_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist")
            continue

        for task_name in sorted(os.listdir(folder_path)):
            task_path = os.path.join(folder_path, task_name)
            if not os.path.isdir(task_path):
                continue

            if task_name not in TASK_EVALUATOR_MAP:
                print(f"Warning: No evaluator for task {task_name}")
                continue

            # Determine logical split from folder name
            split = FOLDER_TO_SPLIT[folder]

            category = get_task_category(task_name)

            for video_file in sorted(os.listdir(task_path)):
                if video_file.endswith('.mp4') and re.fullmatch(r'\d{5}\.mp4', video_file):

                    video_path = os.path.join(task_path, video_file)

                    try:
                        video_idx = int(video_file.replace('.mp4', ''))
                    except ValueError:
                        video_idx = 0

                    all_videos.append({
                        'video_path': video_path,
                        'task_name': task_name,
                        'video_file': video_file,
                        'video_idx': video_idx,
                        'split': split,
                        'category': category,
                        'folder': folder,
                    })

    return all_videos


def init_results(model_name: str, model_path: str) -> dict:
    """Initialize a results dictionary."""
    return {
        'model_name': model_name,
        'model_path': model_path,
        'timestamp': datetime.now().isoformat(),
        'samples': [],
        'summary': {
            'In_Domain': {'scores': [], 'by_task': {}, 'by_category': {}},
            'Out_of_Domain': {'scores': [], 'by_task': {}, 'by_category': {}},
            'overall': {'scores': [], 'by_task': {}, 'by_category': {}}
        }
    }


def aggregate_score(results: dict, sample_result: dict):
    """Add a sample result into the aggregated summary."""
    score = sample_result['score']
    split = sample_result['split']
    task_name = sample_result['task_name']
    category = sample_result['category']

    results['summary'][split]['scores'].append(score)
    results['summary']['overall']['scores'].append(score)

    for key in [split, 'overall']:
        results['summary'][key]['by_task'].setdefault(task_name, [])
        results['summary'][key]['by_task'][task_name].append(score)

        results['summary'][key]['by_category'].setdefault(category, [])
        results['summary'][key]['by_category'][category].append(score)


def finalize_summary(results: dict):
    """Compute mean scores from accumulated lists."""
    for split in ['In_Domain', 'Out_of_Domain', 'overall']:
        scores = results['summary'][split]['scores']
        if scores:
            results['summary'][split]['mean_score'] = sum(scores) / len(scores)
            results['summary'][split]['num_samples'] = len(scores)
        else:
            results['summary'][split]['mean_score'] = 0.0
            results['summary'][split]['num_samples'] = 0

        for task_name, task_scores in results['summary'][split]['by_task'].items():
            results['summary'][split]['by_task'][task_name] = (
                sum(task_scores) / len(task_scores) if isinstance(task_scores, list) else task_scores
            )

        for category, cat_scores in results['summary'][split]['by_category'].items():
            results['summary'][split]['by_category'][category] = (
                sum(cat_scores) / len(cat_scores) if isinstance(cat_scores, list) else cat_scores
            )


def print_results(results: dict):
    """Print a summary of evaluation results."""
    s = results['summary']
    print(f"  In-Domain:      {s['In_Domain']['mean_score']:.4f}  ({s['In_Domain']['num_samples']} samples)")
    print(f"  Out-of-Domain:  {s['Out_of_Domain']['mean_score']:.4f}  ({s['Out_of_Domain']['num_samples']} samples)")
    print(f"  Overall:        {s['overall']['mean_score']:.4f}  ({s['overall']['num_samples']} samples)")

    if s['overall']['by_category']:
        print(f"  By Category:")
        for cat, score in sorted(s['overall']['by_category'].items(), key=lambda x: x[1], reverse=True):
            print(f"    {cat:<16}: {score:.4f}")


def evaluate_model(model_name: str, model_path: str, gt_base_path: str,
                   output_dir: str, device: str = 'cuda') -> dict:
    """Evaluate all videos for a single model directory."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'=' * 60}")

    all_videos = collect_videos(model_path)
    if not all_videos:
        print(f"No videos found in {model_path}")
        return None

    print(f"Found {len(all_videos)} videos to evaluate")

    # Attach GT info
    for v in all_videos:
        v['gt_info'] = find_gt_info(v['task_name'], v['video_idx'], gt_base_path)

    results = init_results(model_name, model_path)

    for video_info in tqdm(all_videos, desc=f"Evaluating {model_name}"):
        result = evaluate_single_video(
            video_info['video_path'],
            video_info['task_name'],
            video_info['gt_info'],
            device,
        )

        sample_result = {
            'video_path': video_info['video_path'],
            'video_file': video_info['video_file'],
            'task_name': video_info['task_name'],
            'split': video_info['split'],
            'category': video_info['category'],
            'folder': video_info['folder'],
            'score': float(result.get('score', 0.0)),
            'dimensions': result.get('dimensions', {}),
            'error': result.get('error', None),
        }

        results['samples'].append(sample_result)
        aggregate_score(results, sample_result)

    finalize_summary(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{model_name}_vbvr_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {output_file}")
    print_results(results)
    return results


def evaluate_checkpoints(model_name: str, model_path: str, gt_base_path: str,
                         output_dir: str, device: str = 'cuda') -> dict:
    """Evaluate a model directory that contains checkpoint subdirectories."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating checkpoints for: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'=' * 60}")

    ckpt_dirs = sorted([
        d for d in os.listdir(model_path)
        if d.startswith('checkpoint-') and os.path.isdir(os.path.join(model_path, d))
    ], key=lambda x: int(x.split('-')[1]) if x.split('-')[1].isdigit() else 0)

    if not ckpt_dirs:
        print(f"No checkpoint directories found in {model_path}")
        return None

    print(f"Found {len(ckpt_dirs)} checkpoints: {ckpt_dirs}")

    all_ckpt_results = {}
    for ckpt_dir in ckpt_dirs:
        ckpt_path = os.path.join(model_path, ckpt_dir)
        ckpt_name = f"{model_name}_{ckpt_dir}"
        results = evaluate_model(ckpt_name, ckpt_path, gt_base_path, output_dir, device)
        if results:
            all_ckpt_results[ckpt_dir] = {
                'In_Domain': results['summary']['In_Domain']['mean_score'],
                'Out_of_Domain': results['summary']['Out_of_Domain']['mean_score'],
                'overall': results['summary']['overall']['mean_score'],
                'by_category': results['summary']['overall']['by_category'],
            }

    # Save checkpoint summary
    if all_ckpt_results:
        summary_file = os.path.join(output_dir, f'{model_name}_checkpoints_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(all_ckpt_results, f, indent=2, cls=NumpyEncoder)
        print(f"\nCheckpoint summary saved to {summary_file}")

    return all_ckpt_results


def main():
    parser = argparse.ArgumentParser(
        description='Run VBVR-Bench evaluation for models with flexible folder structures.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single model directory
  python run_evaluation.py --model_path /path/to/65frames_65frames

  # Evaluate all models under a base directory
  python run_evaluation.py --models_base /path/to/model_videos

  # Evaluate specific models by name
  python run_evaluation.py --models_base /path/to/model_videos --models 65frames_65frames chronoedit

  # Custom GT path and output directory
  python run_evaluation.py --model_path /path/to/model --gt_base /path/to/gt --output_dir /path/to/output
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model_path', type=str,
                       help='Path to a single model video directory to evaluate')
    group.add_argument('--models_base', type=str,
                       help='Base directory containing multiple model folders')

    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific model names to evaluate (used with --models_base)')
    parser.add_argument('--gt_base', type=str, required=True,
                        help='Base path for ground truth data (downloaded from HuggingFace)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: auto-generated)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.model_path:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'evaluation_results',
            os.path.basename(args.model_path)
        )
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'evaluation_results',
            os.path.basename(args.models_base.rstrip('/'))
        )
    os.makedirs(output_dir, exist_ok=True)

    # Collect model paths to evaluate
    model_entries = []  # list of (name, path)

    if args.model_path:
        # Single model
        name = os.path.basename(args.model_path.rstrip('/'))
        model_entries.append((name, args.model_path))
    else:
        # Multiple models under base directory
        all_dirs = sorted([
            d for d in os.listdir(args.models_base)
            if os.path.isdir(os.path.join(args.models_base, d))
        ])

        if args.models:
            selected = set(args.models)
            all_dirs = [d for d in all_dirs if d in selected]

        for d in all_dirs:
            model_entries.append((d, os.path.join(args.models_base, d)))

    if not model_entries:
        print("No model directories found to evaluate.")
        sys.exit(1)

    print(f"Models to evaluate: {[name for name, _ in model_entries]}")
    print(f"GT base: {args.gt_base}")
    print(f"Output dir: {output_dir}")

    # Evaluate each model
    all_summaries = {}

    for model_name, model_path in model_entries:
        structure = detect_folder_structure(model_path)
        print(f"\nDetected structure for {model_name}: {structure}")

        if structure == 'checkpoints':
            ckpt_results = evaluate_checkpoints(
                model_name, model_path, args.gt_base, output_dir, args.device
            )
            if ckpt_results:
                all_summaries[model_name] = {'checkpoints': ckpt_results}
        elif structure == 'in_out_domain':
            results = evaluate_model(
                model_name, model_path, args.gt_base, output_dir, args.device
            )
            if results:
                all_summaries[model_name] = {
                    'In_Domain': results['summary']['In_Domain']['mean_score'],
                    'Out_of_Domain': results['summary']['Out_of_Domain']['mean_score'],
                    'overall': results['summary']['overall']['mean_score'],
                    'by_category': results['summary']['overall']['by_category'],
                }
        else:
            print(f"Skipping {model_name}: unknown folder structure")

    # Save overall summary
    if all_summaries:
        summary_file = os.path.join(output_dir, 'all_models_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(all_summaries, f, indent=2, cls=NumpyEncoder)

        print(f"\n{'=' * 60}")
        print("All Models Summary")
        print(f"{'=' * 60}")

        for model_name, scores in all_summaries.items():
            print(f"\n{model_name}:")
            if 'checkpoints' in scores:
                for ckpt, ckpt_scores in scores['checkpoints'].items():
                    print(f"  {ckpt}:")
                    print(f"    In-Domain:     {ckpt_scores['In_Domain']:.4f}")
                    print(f"    Out-of-Domain: {ckpt_scores['Out_of_Domain']:.4f}")
                    print(f"    Overall:       {ckpt_scores['overall']:.4f}")
            else:
                print(f"  In-Domain:     {scores['In_Domain']:.4f}")
                print(f"  Out-of-Domain: {scores['Out_of_Domain']:.4f}")
                print(f"  Overall:       {scores['overall']:.4f}")
                if 'by_category' in scores and scores['by_category']:
                    print(f"  By Category:")
                    for cat, score in sorted(scores['by_category'].items(), key=lambda x: x[1], reverse=True):
                        print(f"    {cat:<16}: {score:.4f}")

        print(f"\nSummary saved to {summary_file}")


if __name__ == '__main__':
    main()
