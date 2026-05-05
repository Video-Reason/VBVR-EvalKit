#!/usr/bin/env python3
"""
VBVR-Bench Evaluation Script

Evaluates video generation models on 100 visual reasoning tasks.
Results are reported by In-Domain (50 tasks) and Out-of-Domain (50 tasks).

Usage:
    python evaluate.py --videos_path /path/to/videos --gt_path /path/to/gt

Example:
    python evaluate.py \
        --videos_path /path/to/model_outputs \
        --gt_path /path/to/VBVR-Bench \
        --output_path ./results
"""

import argparse
import os
import sys
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vbvr_bench import VBVRBench


def parse_args():
    parser = argparse.ArgumentParser(
        description='VBVR-Bench(100 Tasks)',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--videos_path',
        type=str,
        required=True,
        help='Path to the videos to evaluate.\n'
             'Structure: {videos_path}/{split}/{task_name}/{idx}.mp4'
    )
    
    parser.add_argument(
        '--gt_path',
        type=str,
        required=True,
        help='Path to ground truth data.\n'
             'Structure: {gt_path}/{split}/{task_name}/{idx}/ground_truth.mp4'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default='./evaluation_results/',
        help='Directory to save evaluation results (default: ./evaluation_results/)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Name for this evaluation run (default: model name + timestamp)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        choices=['In-Domain_50', 'Out-of-Domain_50', 'all'],
        default='all',
        help='Which split to evaluate:\n'
             '  - In-Domain_50: In-domain test set (50 tasks)\n'
             '  - Out-of-Domain_50: Out-of-domain test set (50 tasks)\n'
             '  - all: Both splits (default)'
    )
    
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=None,
        help='Specific task names to evaluate (default: all tasks)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for computation (default: cuda)'
    )
    
    parser.add_argument(
        '--save_detailed',
        action='store_true',
        default=True,
        help='Save detailed per-video results (default: True)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set evaluation name
    if args.name is None:
        model_name = os.path.basename(args.videos_path.rstrip('/'))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.name = f'{model_name}_{timestamp}'
    
    print("=" * 70)
    print("VBVR-Bench")
    print("100 Task-Specific Rule-Based Evaluators")
    print("=" * 70)
    print(f"  Videos path:     {args.videos_path}")
    print(f"  GT path:         {args.gt_path}")
    print(f"  Output path:     {args.output_path}")
    print(f"  Evaluation name: {args.name}")
    print(f"  Split:           {args.split}")
    print(f"  Device:          {args.device}")
    print("=" * 70)
    
    # Initialize benchmark
    bench = VBVRBench(
        gt_base_path=args.gt_path,
        output_path=args.output_path,
        device=args.device
    )
    
    # Determine split
    split = None if args.split == 'all' else args.split
    
    # Run evaluation
    results = bench.evaluate(
        videos_path=args.videos_path,
        name=args.name,
        task_list=args.tasks,
        split=split,
        save_detailed=args.save_detailed
    )
    
    # Print final summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {args.output_path}/{args.name}_eval_results.json")
    print()
    print("Final Scores:")
    if 'In_Domain' in results and results['In_Domain'].get('num_videos', 0) > 0:
        print(f"  In-Domain (50 tasks):      {results['In_Domain']['mean_score']:.4f}")
    if 'Out_of_Domain' in results and results['Out_of_Domain'].get('num_videos', 0) > 0:
        print(f"  Out-of-Domain (50 tasks):  {results['Out_of_Domain']['mean_score']:.4f}")
    print(f"  Overall Average:           {results['overall']['mean_score']:.4f}")
    print("=" * 70)
    
    return results


if __name__ == '__main__':
    main()
