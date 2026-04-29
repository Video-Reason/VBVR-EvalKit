#!/usr/bin/env python3
"""
Evaluation script for VBVR-Bench interleaved image generation.

Instead of evaluating videos, this evaluates predicted images against GT images
using the task_specific dimension only (for fair comparison with video models).

GT data structure (from VBVR-Bench-Data-Image-Preview):
    gt_path/
    ├── G-131_select_next_figure_.../
    │   ├── 00000/
    │   │   ├── first_frame.png
    │   │   ├── output_0.png
    │   │   └── meta.json
    │   └── ...
    └── ...

Model prediction structure:
    pred_path/
    ├── G-131_select_next_figure_.../
    │   ├── 00000/
    │   │   ├── output_0.png
    │   │   └── ...
    │   └── ...
    └── ...

Usage:
    # Evaluate a single model
    python run_evaluation_image_preview.py \
        --pred_path /path/to/model_outputs \
        --gt_path /path/to/VBVR-Bench-Image

    # Evaluate multiple models under a base directory
    python run_evaluation_image_preview.py \
        --models_base /path/to/all_model_outputs \
        --gt_path /path/to/VBVR-Bench-Image

    # Evaluate specific models
    python run_evaluation_image_preview.py \
        --models_base /path/to/all_model_outputs \
        --gt_path /path/to/VBVR-Bench-Image \
        --models model_A model_B
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
    get_split,
)


def resolve_task_name(dir_name: str) -> str:
    """Resolve directory name to TASK_EVALUATOR_MAP key.

    The evaluator registry uses names with '_data-generator' suffix,
    but the public bench data directories may not have this suffix.
    """
    if dir_name in TASK_EVALUATOR_MAP:
        return dir_name
    with_suffix = dir_name + "_data-generator"
    if with_suffix in TASK_EVALUATOR_MAP:
        return with_suffix
    return None


def collect_samples(gt_path: str, pred_path: str):
    """Collect all evaluation samples by scanning the GT directory.

    Returns list of dicts with task_name, sample_id, gt_images, pred_images,
    input_image, split, category.
    """
    samples = []

    for generator_dir in sorted(os.listdir(gt_path)):
        generator_path = os.path.join(gt_path, generator_dir)
        if not os.path.isdir(generator_path):
            continue

        task_name = resolve_task_name(generator_dir)
        if task_name is None:
            print(f"Warning: No evaluator for {generator_dir}, skipping")
            continue

        split = get_split(task_name)
        category = get_task_category(task_name)

        for sample_id in sorted(os.listdir(generator_path)):
            sample_gt_path = os.path.join(generator_path, sample_id)
            if not os.path.isdir(sample_gt_path):
                continue

            # Collect GT output images
            gt_images = sorted([
                os.path.join(sample_gt_path, f)
                for f in os.listdir(sample_gt_path)
                if f.startswith("output_") and f.endswith(".png")
            ])

            input_image = os.path.join(sample_gt_path, "first_frame.png")
            if not os.path.exists(input_image):
                input_image = None

            # Collect predicted output images
            sample_pred_path = os.path.join(pred_path, generator_dir, sample_id)
            pred_images = []
            if os.path.isdir(sample_pred_path):
                pred_images = sorted([
                    os.path.join(sample_pred_path, f)
                    for f in os.listdir(sample_pred_path)
                    if f.startswith("output_") and f.endswith(".png")
                ])

            samples.append({
                "task_name": task_name,
                "generator_dir": generator_dir,
                "sample_id": sample_id,
                "input_image": input_image,
                "gt_images": gt_images,
                "pred_images": pred_images,
                "split": split,
                "category": category,
            })

    return samples


def evaluate_model(model_name: str, pred_path: str, gt_path: str,
                   output_dir: str, device: str = "cuda"):
    """Evaluate all samples for a single model."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_name}")
    print(f"Pred: {pred_path}")
    print(f"GT:   {gt_path}")
    print(f"{'=' * 60}")

    samples = collect_samples(gt_path, pred_path)
    if not samples:
        print(f"No samples found")
        return None

    print(f"Found {len(samples)} samples to evaluate")

    results = {
        "model_name": model_name,
        "pred_path": pred_path,
        "gt_path": gt_path,
        "timestamp": datetime.now().isoformat(),
        "samples": [],
        "summary": {
            "In_Domain": {"scores": [], "by_task": {}, "by_category": {}},
            "Out_of_Domain": {"scores": [], "by_task": {}, "by_category": {}},
            "overall": {"scores": [], "by_task": {}, "by_category": {}},
        },
    }

    for sample in tqdm(samples, desc=f"Evaluating {model_name}"):
        task_name = sample["task_name"]
        split = sample["split"]
        category = sample["category"]

        if not sample["pred_images"]:
            sample_result = {
                "generator_dir": sample["generator_dir"],
                "sample_id": sample["sample_id"],
                "task_name": task_name,
                "split": split,
                "category": category,
                "score": 0.0,
                "error": "no prediction found",
                "dimensions": {},
            }
            results["samples"].append(sample_result)
            # Still count missing predictions as 0 score
            results["summary"][split]["scores"].append(0.0)
            results["summary"]["overall"]["scores"].append(0.0)
            for key in [split, "overall"]:
                results["summary"][key]["by_task"].setdefault(task_name, [])
                results["summary"][key]["by_task"][task_name].append(0.0)
                results["summary"][key]["by_category"].setdefault(category, [])
                results["summary"][key]["by_category"][category].append(0.0)
            continue

        eval_info = {
            "input_image": sample["input_image"],
            "pred_images": sample["pred_images"],
            "gt_images": sample["gt_images"],
            "task_name": task_name,
        }

        try:
            evaluator = get_evaluator(task_name, device)
            result = evaluator.evaluate_interleave(eval_info)
        except Exception as e:
            traceback.print_exc()
            result = {"score": 0.0, "error": str(e), "dimensions": {}}

        score = float(result.get("score", 0.0))

        sample_result = {
            "generator_dir": sample["generator_dir"],
            "sample_id": sample["sample_id"],
            "task_name": task_name,
            "split": split,
            "category": category,
            "score": score,
            "dimensions": result.get("dimensions", {}),
            "error": result.get("error", None),
            "details": result.get("details", {}),
        }
        results["samples"].append(sample_result)

        # Aggregate
        results["summary"][split]["scores"].append(score)
        results["summary"]["overall"]["scores"].append(score)

        for key in [split, "overall"]:
            results["summary"][key]["by_task"].setdefault(task_name, [])
            results["summary"][key]["by_task"][task_name].append(score)
            results["summary"][key]["by_category"].setdefault(category, [])
            results["summary"][key]["by_category"][category].append(score)

    # Finalize
    for split in ["In_Domain", "Out_of_Domain", "overall"]:
        scores = results["summary"][split]["scores"]
        results["summary"][split]["mean_score"] = sum(scores) / len(scores) if scores else 0.0
        results["summary"][split]["num_samples"] = len(scores)

        for task_name, task_scores in results["summary"][split]["by_task"].items():
            if isinstance(task_scores, list):
                results["summary"][split]["by_task"][task_name] = (
                    sum(task_scores) / len(task_scores)
                )

        for cat, cat_scores in results["summary"][split]["by_category"].items():
            if isinstance(cat_scores, list):
                results["summary"][split]["by_category"][cat] = (
                    sum(cat_scores) / len(cat_scores)
                )

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name}_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # Print summary
    s = results["summary"]
    print(f"\nResults saved to {output_file}")
    print(f"  In-Domain:      {s['In_Domain']['mean_score']:.4f}  ({s['In_Domain']['num_samples']} samples)")
    print(f"  Out-of-Domain:  {s['Out_of_Domain']['mean_score']:.4f}  ({s['Out_of_Domain']['num_samples']} samples)")
    print(f"  Overall:        {s['overall']['mean_score']:.4f}  ({s['overall']['num_samples']} samples)")

    if s["overall"]["by_category"]:
        print(f"  By Category:")
        for cat, score in sorted(s["overall"]["by_category"].items(), key=lambda x: x[1], reverse=True):
            print(f"    {cat:<16}: {score:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run VBVR-Bench evaluation for interleaved image generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single model
  python run_evaluation_image_preview.py --pred_path /path/to/model_outputs --gt_path /path/to/bench_data

  # Evaluate all models under a base directory
  python run_evaluation_image_preview.py --models_base /path/to/all_models --gt_path /path/to/bench_data

  # Evaluate specific models
  python run_evaluation_image_preview.py --models_base /path/to/all_models --gt_path /path/to/bench_data --models model_A model_B
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred_path", type=str,
                       help="Path to a single model's prediction directory")
    group.add_argument("--models_base", type=str,
                       help="Base directory containing multiple model folders")

    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Specific model names to evaluate (used with --models_base)")
    parser.add_argument("--gt_path", type=str, required=True,
                        help="Path to VBVR-Bench-Data-Image-Preview ground truth data")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    # Collect model paths
    model_entries = []
    if args.pred_path:
        name = os.path.basename(args.pred_path.rstrip("/"))
        model_entries.append((name, args.pred_path))
    else:
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
    print(f"GT path: {args.gt_path}")
    print(f"Output dir: {args.output_dir}")

    # Evaluate each model
    all_summaries = {}
    for model_name, pred_path in model_entries:
        results = evaluate_model(model_name, pred_path, args.gt_path,
                                 args.output_dir, args.device)
        if results:
            all_summaries[model_name] = {
                "In_Domain": results["summary"]["In_Domain"]["mean_score"],
                "Out_of_Domain": results["summary"]["Out_of_Domain"]["mean_score"],
                "overall": results["summary"]["overall"]["mean_score"],
                "by_category": results["summary"]["overall"]["by_category"],
            }

    # Save overall summary
    if all_summaries and len(all_summaries) > 1:
        summary_file = os.path.join(args.output_dir, "all_models_summary.json")
        with open(summary_file, "w") as f:
            json.dump(all_summaries, f, indent=2, cls=NumpyEncoder)

        print(f"\n{'=' * 60}")
        print("All Models Summary")
        print(f"{'=' * 60}")
        for model_name, scores in all_summaries.items():
            print(f"\n{model_name}:")
            print(f"  In-Domain:     {scores['In_Domain']:.4f}")
            print(f"  Out-of-Domain: {scores['Out_of_Domain']:.4f}")
            print(f"  Overall:       {scores['overall']:.4f}")
        print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
