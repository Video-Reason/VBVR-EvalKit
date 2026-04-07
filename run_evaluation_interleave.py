#!/usr/bin/env python3
"""
Evaluation script for VBVR-Bench interleaved image generation.

Instead of evaluating videos, this evaluates predicted images against GT images
using the task_specific dimension only (for fair comparison with video models).

Input:
    - bench_jsonl: VBVR-Interleave-Bench jsonl file (contains id, image paths, conversations)
    - pred_folder: Directory containing predicted images (gen_sample_{id}_*.png)
    - bench_root: Root directory for bench image paths

Usage:
    # ThinkMorph indomain (pred already split into subdirectories)
    python run_evaluation_interleave.py \
        --bench_jsonl /mnt/aigc/xujunxiang/vbvr_interleave_data/20260316_vbvr_all_data/interleave_dataset/VBVR-Interleave-Bench/VBVR_Bench_interleave_allframes_indomain.jsonl \
        --pred_folder /mnt/aigc/caizhongang/ThinkMorph/results/vbvr_all_data_allframes_anno/inference_results/0025000/VBVR_Bench_indomain/generated_images

    # Neo outdomain
    python run_evaluation_interleave.py \
        --bench_jsonl /mnt/aigc/xujunxiang/vbvr_interleave_data/20260316_vbvr_all_data/interleave_dataset/VBVR-Interleave-Bench/VBVR_Bench_interleave_allframes_outdomain.jsonl \
        --pred_folder /mnt/aigc/caizhongang/Neo_Unify/RUN/neo_interleave_vbvr_all_data_anno_unfreeze_llm/eval_step25000/images_step_25000 \
        --domain outdomain

"""

import os
import sys
import re
import json
import argparse
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


def build_pred_index(folder_path: str, domain: str = None):
    """
    Index predicted images by task_id.
    Supports:
      - gen_sample_{id}_*.png  (ThinkMorph)
      - *_sample_{id}_gen_*.png  (Neo)
    """
    pattern = re.compile(r'sample_(\d+)')
    pred_map = {}
    if not os.path.exists(folder_path):
        return pred_map
    for item in os.listdir(folder_path):
        if domain and ("indomain" in item or "outdomain" in item) and domain not in item:
            continue
        m = pattern.search(item)
        if not m:
            continue
        full_path = os.path.join(folder_path, item)
        task_id = m.group(1)
        pred_map.setdefault(task_id, []).append(full_path)

    for k in pred_map:
        pred_map[k].sort()
    return pred_map


def load_bench_samples(bench_jsonl: str, bench_root: str, bench_json: str):
    """
    Load benchmark samples from jsonl and build id -> task_name mapping from bench json.

    Returns:
        samples: list of dicts with id, input_image, gt_images, question, task_name, split, category
    """
    # Build id -> task_name mapping from VBVR-Bench.json
    id_to_task = {}
    with open(bench_json) as f:
        bench_data = json.load(f)
    for item in bench_data:
        task_name = item["prompt_path"].split("/")[1]
        id_to_task[item["id"]] = task_name

    samples = []
    with open(bench_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            sample_id = data["id"]
            images = data["image"]

            input_image = os.path.join(bench_root, images[0])
            gt_images = [os.path.join(bench_root, img) for img in images[1:]]

            task_name = id_to_task.get(sample_id, "unknown")
            split = get_split(task_name) if task_name != "unknown" else "unknown"
            category = get_task_category(task_name) if task_name in TASK_EVALUATOR_MAP else "unknown"

            samples.append({
                "id": sample_id,
                "input_image": input_image,
                "gt_images": gt_images,
                "task_name": task_name,
                "split": split,
                "category": category,
            })

    return samples


def evaluate_interleave(
    bench_jsonl: str,
    pred_folder: str,
    bench_root: str,
    bench_json: str,
    output_dir: str,
    name: str = None,
    domain: str = None,
    device: str = "cuda",
):
    """Run interleave evaluation."""

    if name is None:
        name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    os.makedirs(output_dir, exist_ok=True)

    # Load benchmark samples
    samples = load_bench_samples(bench_jsonl, bench_root, bench_json)
    print(f"Loaded {len(samples)} benchmark samples")

    # Build prediction index
    pred_map = build_pred_index(pred_folder, domain=domain)
    print(f"Found predictions for {len(pred_map)} sample IDs")

    # Results
    results = {
        "name": name,
        "bench_jsonl": bench_jsonl,
        "pred_folder": pred_folder,
        "timestamp": datetime.now().isoformat(),
        "samples": [],
        "summary": {
            "In_Domain": {"scores": [], "by_task": {}, "by_category": {}},
            "Out_of_Domain": {"scores": [], "by_task": {}, "by_category": {}},
            "overall": {"scores": [], "by_task": {}, "by_category": {}},
        },
    }

    for sample in tqdm(samples, desc="Evaluating"):
        sample_id = str(sample["id"])
        task_name = sample["task_name"]
        split = sample["split"]
        category = sample["category"]

        pred_images = pred_map.get(sample_id, [])

        if not pred_images:
            sample_result = {
                "id": sample["id"],
                "task_name": task_name,
                "split": split,
                "category": category,
                "score": 0.0,
                "error": "no prediction found",
                "dimensions": {},
            }
            results["samples"].append(sample_result)
            continue

        # Build eval_info for evaluator
        eval_info = {
            "input_image": sample["input_image"],
            "pred_images": pred_images,
            "gt_images": sample["gt_images"],
            "task_name": task_name,
        }

        # Get evaluator and run
        try:
            evaluator = get_evaluator(task_name, device)
            result = evaluator.evaluate_interleave(eval_info)
        except Exception as e:
            result = {"score": 0.0, "error": str(e), "dimensions": {}}

        score = float(result.get("score", 0.0))

        sample_result = {
            "id": sample["id"],
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

    # Save detailed results
    result_dir = os.path.join(output_dir, name)
    os.makedirs(result_dir, exist_ok=True)

    detail_file = os.path.join(result_dir, "eval_results.json")
    with open(detail_file, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    # Save summary separately
    s = results["summary"]
    summary = {
        "name": name,
        "timestamp": results["timestamp"],
        "In_Domain": {
            "mean_score": s["In_Domain"]["mean_score"],
            "num_samples": s["In_Domain"]["num_samples"],
            "by_task": s["In_Domain"]["by_task"],
            "by_category": s["In_Domain"]["by_category"],
        },
        "Out_of_Domain": {
            "mean_score": s["Out_of_Domain"]["mean_score"],
            "num_samples": s["Out_of_Domain"]["num_samples"],
            "by_task": s["Out_of_Domain"]["by_task"],
            "by_category": s["Out_of_Domain"]["by_category"],
        },
        "overall": {
            "mean_score": s["overall"]["mean_score"],
            "num_samples": s["overall"]["num_samples"],
            "by_task": s["overall"]["by_task"],
            "by_category": s["overall"]["by_category"],
        },
    }
    summary_file = os.path.join(result_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results: {name}")
    print(f"{'='*60}")
    print(f"  In-Domain:      {s['In_Domain']['mean_score']:.4f}  ({s['In_Domain']['num_samples']} samples)")
    print(f"  Out-of-Domain:  {s['Out_of_Domain']['mean_score']:.4f}  ({s['Out_of_Domain']['num_samples']} samples)")
    print(f"  Overall:        {s['overall']['mean_score']:.4f}  ({s['overall']['num_samples']} samples)")

    if s["overall"]["by_category"]:
        print(f"  By Category:")
        for cat, score in sorted(s["overall"]["by_category"].items(), key=lambda x: x[1], reverse=True):
            print(f"    {cat:<16}: {score:.4f}")

    print(f"\nDetailed results: {detail_file}")
    print(f"Summary: {summary_file}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run VBVR-Bench evaluation for interleaved image generation.",
    )
    parser.add_argument("--bench_jsonl", type=str, required=True,
                        help="Path to bench jsonl (e.g. VBVR_Bench_interleave_allframes_indomain.jsonl)")
    parser.add_argument("--pred_folder", type=str, required=True,
                        help="Directory containing predicted images")
    parser.add_argument("--bench_root", type=str,
                        default="/mnt/aigc/xujunxiang/vbvr_interleave_data/20260316_vbvr_all_data/interleave_dataset",
                        help="Root directory for bench image paths")
    parser.add_argument("--bench_json", type=str,
                        default="/mnt/aigc/xujunxiang/vbvr_interleave_data/20260316_vbvr_all_data/VBVR-Bench.json",
                        help="Path to VBVR-Bench.json (for id -> task mapping)")
    parser.add_argument("--output_dir", type=str, default="./interleave_eval_results",
                        help="Output directory for results")
    parser.add_argument("--name", type=str, default=None,
                        help="Name for this evaluation run")
    parser.add_argument("--domain", type=str, default=None,
                        help="Domain filter for pred files (indomain/outdomain)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    args = parser.parse_args()

    evaluate_interleave(
        bench_jsonl=args.bench_jsonl,
        pred_folder=args.pred_folder,
        bench_root=args.bench_root,
        bench_json=args.bench_json,
        output_dir=args.output_dir,
        name=args.name,
        domain=args.domain,
        device=args.device,
    )


if __name__ == "__main__":
    main()
