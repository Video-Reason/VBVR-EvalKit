"""
Evaluation script for ICML video models (Open_60/Hidden_40 structure).

Usage:
    # Evaluate a single model
    python run_evaluation_video_icml.py \
        --model_path /mnt/umm/users/wangruisi/01-project/2026_ICML_VBVR/model_videos_modified/VBVR-Wan2.2 \
        --gt_base /mnt/umm/users/wangruisi/01-project/mllm/hokin_data/VBVR-Bench

    # Evaluate all models under a base directory
    python run_evaluation_video_icml.py \
        --models_base /mnt/umm/users/wangruisi/01-project/2026_ICML_VBVR/model_videos_modified \
        --gt_base /mnt/umm/users/wangruisi/01-project/mllm/hokin_data/VBVR-Bench

    # Evaluate a single model on specific tasks (use --tasks to specify one or more task names)
    python run_evaluation_video_icml.py \
        --model_path /mnt/umm/users/wangruisi/01-project/2026_ICML_VBVR/model_videos_modified/VBVR-Wan2.2 \
        --gt_base /mnt/umm/users/wangruisi/01-project/mllm/hokin_data/VBVR-Bench \
        --tasks G-3_stable_sort_data-generator G-8_track_object_movement_data-generator G-9_identify_objects_in_region_data-generator

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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

sys.path.insert(0, str(Path(__file__).parent))

from vbvr_bench.evaluators import (
    get_evaluator, TASK_EVALUATOR_MAP, get_task_category,
    is_out_of_domain, get_split,
)


def build_gt_task_map(gt_base: str):
    """Build task_name -> gt_domain mapping."""
    task_to_domain = {}
    for domain in ['In-Domain_50', 'Out-of-Domain_50']:
        domain_path = os.path.join(gt_base, domain)
        if not os.path.exists(domain_path):
            continue
        for task_name in os.listdir(domain_path):
            task_path = os.path.join(domain_path, task_name)
            if os.path.isdir(task_path):
                task_to_domain[task_name] = domain
    return task_to_domain


def find_gt_info(gt_base: str, domain: str, task_name: str, video_idx: str):
    """Find GT files for a specific video."""
    gt_dir = os.path.join(gt_base, domain, task_name, video_idx)
    if not os.path.exists(gt_dir):
        return None

    info = {'gt_path': gt_dir}

    gt_video = os.path.join(gt_dir, 'ground_truth.mp4')
    if os.path.exists(gt_video):
        info['gt_video_path'] = gt_video

    first_frame = os.path.join(gt_dir, 'first_frame.png')
    if os.path.exists(first_frame):
        info['gt_first_frame'] = first_frame

    final_frame = os.path.join(gt_dir, 'final_frame.png')
    if os.path.exists(final_frame):
        info['gt_final_frame'] = final_frame

    prompt_file = os.path.join(gt_dir, 'prompt.txt')
    if os.path.exists(prompt_file):
        with open(prompt_file) as f:
            info['prompt'] = f.read().strip()

    return info


def evaluate_model(model_path: str, gt_base: str, output_dir: str, device: str = 'cuda', tasks: list = None):
    """Evaluate a single model. If tasks is provided, only evaluate those task names."""
    model_name = os.path.basename(model_path)
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    # Build GT mapping
    task_to_domain = build_gt_task_map(gt_base)

    # Collect all videos
    videos = []
    for split in ['Open_60', 'Hidden_40']:
        split_path = os.path.join(model_path, split)
        if not os.path.exists(split_path):
            continue
        for task_name in sorted(os.listdir(split_path)):
            task_path = os.path.join(split_path, task_name)
            if not os.path.isdir(task_path):
                continue
            # Skip _discarded, _preprocessed
            if task_name.startswith('_'):
                continue

            # Filter by task names if specified
            if tasks and task_name not in tasks:
                continue

            gt_domain = task_to_domain.get(task_name)
            if gt_domain is None:
                continue

            for video_file in sorted(os.listdir(task_path)):
                if not video_file.endswith('.mp4'):
                    continue
                video_idx = video_file.replace('.mp4', '')
                video_path = os.path.join(task_path, video_file)

                gt_info = find_gt_info(gt_base, gt_domain, task_name, video_idx)
                if gt_info is None:
                    continue

                domain_label = 'In_Domain' if gt_domain == 'In-Domain_50' else 'Out_of_Domain'

                videos.append({
                    'video_path': video_path,
                    'task_name': task_name,
                    'video_idx': video_idx,
                    'split': split,
                    'domain': domain_label,
                    'gt_info': gt_info,
                })

    print(f"Found {len(videos)} videos to evaluate")

    # Results
    results = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'samples': [],
        'summary': {
            'In_Domain': {'scores': [], 'by_task': {}},
            'Out_of_Domain': {'scores': [], 'by_task': {}},
            'overall': {'scores': [], 'by_task': {}},
        },
    }

    for video_info in tqdm(videos, desc=f"[{model_name}]"):
        task_name = video_info['task_name']
        gt_info = video_info['gt_info']
        domain = video_info['domain']

        eval_info = {
            'video_path': video_info['video_path'],
            'gt_path': gt_info.get('gt_path', ''),
            'gt_video_path': gt_info.get('gt_video_path', ''),
            'gt_first_frame': gt_info.get('gt_first_frame', ''),
            'gt_final_frame': gt_info.get('gt_final_frame', ''),
            'prompt': gt_info.get('prompt', ''),
        }

        try:
            evaluator = get_evaluator(task_name, device)
            result = evaluator.evaluate(eval_info, task_specific_only=True)
        except Exception as e:
            result = {'score': 0.0, 'error': str(e), 'dimensions': {}}

        score = float(result.get('score', 0.0))

        sample_result = {
            'task_name': task_name,
            'video_idx': video_info['video_idx'],
            'split': video_info['split'],
            'domain': domain,
            'score': score,
            'dimensions': result.get('dimensions', {}),
            'error': result.get('error', None),
        }
        results['samples'].append(sample_result)

        # Aggregate
        results['summary'][domain]['scores'].append(score)
        results['summary']['overall']['scores'].append(score)

        for key in [domain, 'overall']:
            results['summary'][key]['by_task'].setdefault(task_name, [])
            results['summary'][key]['by_task'][task_name].append(score)

    # Finalize
    for split in ['In_Domain', 'Out_of_Domain', 'overall']:
        scores = results['summary'][split]['scores']
        results['summary'][split]['mean_score'] = sum(scores) / len(scores) if scores else 0.0
        results['summary'][split]['num_samples'] = len(scores)

        for task_name, task_scores in results['summary'][split]['by_task'].items():
            if isinstance(task_scores, list):
                results['summary'][split]['by_task'][task_name] = (
                    sum(task_scores) / len(task_scores)
                )

    # Save
    result_dir = os.path.join(output_dir, model_name)
    os.makedirs(result_dir, exist_ok=True)

    detail_file = os.path.join(result_dir, 'eval_results.json')
    with open(detail_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    summary = {
        'model_name': model_name,
        'In_Domain': {
            'mean_score': results['summary']['In_Domain']['mean_score'],
            'num_samples': results['summary']['In_Domain']['num_samples'],
        },
        'Out_of_Domain': {
            'mean_score': results['summary']['Out_of_Domain']['mean_score'],
            'num_samples': results['summary']['Out_of_Domain']['num_samples'],
        },
        'overall': {
            'mean_score': results['summary']['overall']['mean_score'],
            'num_samples': results['summary']['overall']['num_samples'],
        },
    }
    summary_file = os.path.join(result_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Print
    s = results['summary']
    print(f"\n  In-Domain:      {s['In_Domain']['mean_score']:.4f}  ({s['In_Domain']['num_samples']} samples)")
    print(f"  Out-of-Domain:  {s['Out_of_Domain']['mean_score']:.4f}  ({s['Out_of_Domain']['num_samples']} samples)")
    print(f"  Overall:        {s['overall']['mean_score']:.4f}  ({s['overall']['num_samples']} samples)")
    print(f"  Results: {result_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ICML video models (Open_60/Hidden_40 structure).',
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model_path', type=str,
                       help='Path to a single model directory')
    group.add_argument('--models_base', type=str,
                       help='Base directory containing multiple model folders')

    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific model names to evaluate (used with --models_base)')
    parser.add_argument('--gt_base', type=str,
                        default='/mnt/umm/users/wangruisi/01-project/mllm/hokin_data/VBVR-Bench',
                        help='Base path for ground truth data')
    parser.add_argument('--output_dir', type=str, default='./video_eval_results_icml',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='Only evaluate these task names (e.g. G-5_multi_object_placement_data-generator)')

    args = parser.parse_args()

    if args.model_path:
        evaluate_model(args.model_path, args.gt_base, args.output_dir, args.device, args.tasks)
    else:
        # Evaluate multiple models
        all_models = sorted([
            d for d in os.listdir(args.models_base)
            if os.path.isdir(os.path.join(args.models_base, d))
        ])

        if args.models:
            all_models = [m for m in all_models if m in args.models]

        print(f"Models to evaluate: {all_models}")

        for model_name in all_models:
            model_path = os.path.join(args.models_base, model_name)
            try:
                evaluate_model(model_path, args.gt_base, args.output_dir, args.device, args.tasks)
            except Exception as e:
                print(f"\n[ERROR] {model_name}: {e}")
                traceback.print_exc()


if __name__ == '__main__':
    main()
