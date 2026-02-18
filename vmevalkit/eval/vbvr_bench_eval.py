"""VBVR-Bench rule-based evaluator for VMEvalKit.

Provides deterministic, rule-based evaluation using VBVR-Bench's 100 task-specific
evaluators. Unlike VLM-based scoring (GPT-4O, InternVL), this requires no API calls
and produces fully reproducible 0-1 continuous scores.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def _import_vbvr():
    """Lazy import of VBVR-Bench modules."""
    from vmevalkit.eval.vbvr_bench.evaluators import (
        get_evaluator, TASK_EVALUATOR_MAP, get_task_category,
        get_tasks_by_split
    )
    return get_evaluator, TASK_EVALUATOR_MAP, get_task_category, get_tasks_by_split


class VBVRBenchEvaluator:
    """Rule-based evaluator using VBVR-Bench's 100 task-specific evaluators.

    Walks VMEvalKit's inference directory structure (2-layer or 3-layer),
    maps each task to a VBVR-Bench evaluator, and produces 0-1 scores.
    """

    def __init__(
        self,
        inference_dir: str,
        eval_output_dir: str = "./evaluations/rubrics",
        gt_base_path: Optional[str] = None,
        device: str = "cuda",
        task_specific_only: bool = True,
    ):
        """
        Args:
            inference_dir: Directory containing inference outputs (VMEvalKit structure).
            eval_output_dir: Directory to save evaluation results.
            gt_base_path: Optional path to VBVR-Bench GT data (for ground_truth.mp4).
            device: 'cuda' or 'cpu'.
            task_specific_only: If True, use only the task-specific dimension score.
        """
        self.inference_dir = Path(inference_dir)
        self.eval_output_dir = Path(eval_output_dir)
        self.gt_base_path = Path(gt_base_path) if gt_base_path else None
        self.device = device
        self.task_specific_only = task_specific_only
        self.evaluator_name = "VBVRBenchEvaluator"

        self.eval_output_dir.mkdir(parents=True, exist_ok=True)

        # Import and cache VBVR-Bench internals
        (self._get_evaluator, self._TASK_MAP,
         self._get_category, self._get_by_split) = _import_vbvr()

    # ------------------------------------------------------------------
    # Task name resolution
    # ------------------------------------------------------------------

    def _resolve_task_name(self, dir_name: str) -> Optional[str]:
        """Match a directory name to a VBVR-Bench task name.

        Tries exact match first, then prefix match.
        """
        if dir_name in self._TASK_MAP:
            return dir_name
        for task_name in self._TASK_MAP:
            if task_name.startswith(dir_name) or dir_name.startswith(task_name):
                return task_name
        return None

    @staticmethod
    def _is_generator_dir(name: str) -> bool:
        """Check if directory name matches VBVR-Bench generator pattern."""
        return bool(
            re.match(r'^[A-Z]-\d+', name)
            and ("_data-generator" in name or "-data-generator" in name)
        )

    # ------------------------------------------------------------------
    # eval_info construction
    # ------------------------------------------------------------------

    def _build_eval_info(
        self,
        video_path: str,
        run_dir: Path,
        task_name: str,
    ) -> Dict[str, Any]:
        """Map VMEvalKit paths to VBVR-Bench eval_info dict."""
        question_dir = run_dir / "question"

        eval_info: Dict[str, Any] = {
            "task_name": task_name,
            "video_path": video_path,
            "gt_first_frame": str(question_dir / "first_frame.png"),
            "gt_final_frame": str(question_dir / "final_frame.png"),
            "prompt_path": str(question_dir / "prompt.txt"),
            "gt_path": str(question_dir),
        }

        # Load prompt
        prompt_path = question_dir / "prompt.txt"
        if prompt_path.exists():
            eval_info["prompt"] = prompt_path.read_text().strip()

        # GT video from separate gt_base_path if available
        if self.gt_base_path:
            # Try to find ground_truth.mp4 under gt_base_path matching the task
            for split_name in ("Open_60", "Hidden_40"):
                gt_video = self.gt_base_path / split_name / task_name
                if gt_video.exists():
                    # Find matching sample directory
                    for sample_dir in sorted(gt_video.iterdir()):
                        gt_mp4 = sample_dir / "ground_truth.mp4"
                        if gt_mp4.exists():
                            eval_info["gt_video_path"] = str(gt_mp4)
                            break
                    break

        # Also check for ground_truth.mp4 in question dir
        gt_local = question_dir / "ground_truth.mp4"
        if gt_local.exists() and "gt_video_path" not in eval_info:
            eval_info["gt_video_path"] = str(gt_local)

        return eval_info

    # ------------------------------------------------------------------
    # Single evaluation
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        model_name: str,
        task_type: str,
        task_id: str,
        video_path: str,
        run_dir: Path,
    ) -> Dict[str, Any]:
        """Evaluate a single video with the matching VBVR-Bench evaluator.

        Returns:
            dict with score (0-1), dimensions, status, evaluation_type.
        """
        # Extract VBVR task name (generator part for 3-layer, task_type for 2-layer)
        dir_name = task_type.split("/")[0] if "/" in task_type else task_type
        task_name = self._resolve_task_name(dir_name)

        if task_name is None:
            return {
                "score": 0.0,
                "status": "skipped",
                "error": f"No VBVR-Bench evaluator for: {dir_name}",
            }

        eval_info = self._build_eval_info(video_path, run_dir, task_name)
        evaluator = self._get_evaluator(task_name, self.device)

        result = evaluator.evaluate(
            eval_info, task_specific_only=self.task_specific_only
        )

        return {
            "score": result.get("score", 0.0),
            "dimensions": result.get("dimensions", {}),
            "details": result.get("details", {}),
            "status": "completed" if "error" not in result else "failed",
            "evaluation_type": "rubrics",
            "vbvr_task_name": task_name,
            **({"error": result["error"]} if "error" in result else {}),
        }

    # ------------------------------------------------------------------
    # Resume support & persistence
    # ------------------------------------------------------------------

    def _has_evaluation(self, model_name: str, task_type: str, task_id: str) -> bool:
        eval_file = (
            self.eval_output_dir / model_name / task_type / task_id
            / f"{self.evaluator_name}.json"
        )
        return eval_file.exists()

    def _save_single_result(
        self, model_name: str, task_type: str, task_id: str, eval_result: Dict[str, Any]
    ):
        task_output_dir = self.eval_output_dir / model_name / task_type / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        with open(task_output_dir / f"{self.evaluator_name}.json", "w") as f:
            json.dump(
                {
                    "metadata": {
                        "evaluator": self.evaluator_name,
                        "timestamp": datetime.now().isoformat(),
                        "model_name": model_name,
                        "task_type": task_type,
                        "task_id": task_id,
                    },
                    "result": eval_result,
                },
                f,
                indent=2,
                default=str,
            )

    # ------------------------------------------------------------------
    # Model-level evaluation (directory walking)
    # ------------------------------------------------------------------

    def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """Evaluate all tasks for one model."""
        from vmevalkit.eval.VLMasjudge.run_selector import select_latest_run

        model_dir = self.inference_dir / model_name
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")

        results: Dict[str, Any] = {"model_name": model_name, "evaluations": {}}
        total = skipped = evaluated = failed = 0

        for first_level_dir in sorted(model_dir.iterdir()):
            if not first_level_dir.is_dir():
                continue

            if self._is_generator_dir(first_level_dir.name):
                # 3-layer: model/generator/task_type/task_id
                generator_name = first_level_dir.name
                for task_type_dir in sorted(first_level_dir.iterdir()):
                    if not task_type_dir.is_dir():
                        continue
                    full_task_type = f"{generator_name}/{task_type_dir.name}"
                    results["evaluations"].setdefault(full_task_type, {})

                    for task_dir in sorted(task_type_dir.iterdir()):
                        if not task_dir.is_dir():
                            continue
                        task_id = task_dir.name
                        total += 1

                        if self._has_evaluation(model_name, full_task_type, task_id):
                            skipped += 1
                            continue

                        run_dir = select_latest_run(task_dir)
                        if not run_dir:
                            continue
                        video_files = sorted((run_dir / "video").glob("*.mp4"))
                        if not video_files:
                            continue

                        try:
                            logger.info(f"Evaluating {model_name}/{full_task_type}/{task_id}")
                            eval_result = self.evaluate_single(
                                model_name, full_task_type, task_id,
                                str(video_files[0]), run_dir,
                            )
                            results["evaluations"][full_task_type][task_id] = eval_result
                            self._save_single_result(model_name, full_task_type, task_id, eval_result)
                            evaluated += 1
                        except Exception as e:
                            logger.error(f"Error {model_name}/{full_task_type}/{task_id}: {e}")
                            failed += 1
                            results["evaluations"][full_task_type][task_id] = {
                                "status": "failed", "error": str(e),
                            }
            else:
                # 2-layer: model/task_type/task_id
                task_type = first_level_dir.name
                results["evaluations"].setdefault(task_type, {})

                for task_dir in sorted(first_level_dir.iterdir()):
                    if not task_dir.is_dir():
                        continue
                    task_id = task_dir.name
                    total += 1

                    if self._has_evaluation(model_name, task_type, task_id):
                        skipped += 1
                        continue

                    run_dir = select_latest_run(task_dir)
                    if not run_dir:
                        continue
                    video_files = sorted((run_dir / "video").glob("*.mp4"))
                    if not video_files:
                        continue

                    try:
                        logger.info(f"Evaluating {model_name}/{task_type}/{task_id}")
                        eval_result = self.evaluate_single(
                            model_name, task_type, task_id,
                            str(video_files[0]), run_dir,
                        )
                        results["evaluations"][task_type][task_id] = eval_result
                        self._save_single_result(model_name, task_type, task_id, eval_result)
                        evaluated += 1
                    except Exception as e:
                        logger.error(f"Error {model_name}/{task_type}/{task_id}: {e}")
                        failed += 1
                        results["evaluations"][task_type][task_id] = {
                            "status": "failed", "error": str(e),
                        }

        logger.info(
            f"VBVR-Bench Evaluation for {model_name}: "
            f"total={total} skipped={skipped} evaluated={evaluated} failed={failed}"
        )
        return results

    # ------------------------------------------------------------------
    # All models
    # ------------------------------------------------------------------

    def evaluate_all_models(self) -> Dict[str, Any]:
        """Evaluate all models in inference_dir."""
        for model_dir in sorted(self.inference_dir.iterdir()):
            if model_dir.is_dir():
                logger.info(f"Evaluating model: {model_dir.name}")
                self.evaluate_model(model_dir.name)

        summary = self._rebuild_summary_from_files()
        return summary

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_prefix_and_number(text: str):
        match = re.match(r'([A-Z])-(\d+)', text)
        if match:
            return (match.group(1), int(match.group(2)), text)
        return ('~', 0, text)

    def _rebuild_summary_from_files(self) -> Dict[str, Any]:
        """Rebuild summary with VBVR-specific breakdowns (category, split)."""
        from statistics import mean, median, stdev
        from collections import Counter

        all_models: Dict[str, Any] = {}

        for model_dir in self.eval_output_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            evaluations: Dict[str, Dict] = {}

            eval_files = list(model_dir.rglob(f"{self.evaluator_name}.json"))
            logger.info(f"Found {len(eval_files)} evaluation files for {model_name}")

            for eval_file in eval_files:
                try:
                    with open(eval_file, "r") as f:
                        data = json.load(f)

                    parts = eval_file.relative_to(model_dir).parts
                    if len(parts) >= 3:
                        # 3-layer: generator/task_type/task_id/file.json
                        full_task_type = f"{parts[0]}/{parts[1]}"
                        task_id = parts[2]
                    elif len(parts) >= 2:
                        # 2-layer: task_type/task_id/file.json
                        full_task_type = parts[0]
                        task_id = parts[1]
                    else:
                        continue

                    evaluations.setdefault(full_task_type, {})[task_id] = data.get("result", {})
                except Exception as e:
                    logger.warning(f"Could not load {eval_file}: {e}")

            if not evaluations:
                continue

            sorted_evaluations = dict(sorted(
                evaluations.items(),
                key=lambda x: self._extract_prefix_and_number(x[0]),
            ))

            enhanced_tasks: Dict[str, Any] = {}
            model_all_scores: list[float] = []
            category_scores: Dict[str, list[float]] = {}
            split_scores: Dict[str, list[float]] = {"In_Domain": [], "Out_of_Domain": []}

            for task_type, task_samples in sorted_evaluations.items():
                task_scores: list[float] = []
                simplified_samples: Dict[str, Any] = {}

                for sample_id, sample_data in task_samples.items():
                    score = sample_data.get("score", 0.0)
                    task_scores.append(score)
                    model_all_scores.append(score)

                    simplified_samples[sample_id] = {
                        "score": score,
                        "status": sample_data.get("status", "unknown"),
                    }

                # Task statistics
                if task_scores:
                    task_statistics = {
                        "total_samples": len(task_scores),
                        "mean_score": round(mean(task_scores), 4),
                        "median_score": round(median(task_scores), 4),
                        "std_score": round(stdev(task_scores), 4) if len(task_scores) > 1 else 0.0,
                        "min_score": round(min(task_scores), 4),
                        "max_score": round(max(task_scores), 4),
                    }
                else:
                    task_statistics = {"total_samples": 0, "status": "pending"}

                # VBVR category & split
                vbvr_task = sample_data.get("vbvr_task_name", "") if task_samples else ""
                if not vbvr_task:
                    dir_name = task_type.split("/")[0]
                    vbvr_task = self._resolve_task_name(dir_name) or ""

                category = self._get_category(vbvr_task) if vbvr_task else "Unknown"
                category_scores.setdefault(category, []).extend(task_scores)

                # Determine split
                from vmevalkit.eval.vbvr_bench import is_out_of_domain
                split = "Out_of_Domain" if (vbvr_task and is_out_of_domain(vbvr_task)) else "In_Domain"
                split_scores[split].extend(task_scores)

                enhanced_tasks[task_type] = {
                    "task_statistics": task_statistics,
                    "vbvr_task_name": vbvr_task,
                    "category": category,
                    "split": split,
                    "samples": simplified_samples,
                }

            # Model-level statistics
            model_statistics: Dict[str, Any] = {}
            if model_all_scores:
                model_statistics = {
                    "total_samples": len(model_all_scores),
                    "mean_score": round(mean(model_all_scores), 4),
                    "median_score": round(median(model_all_scores), 4),
                    "std_score": round(stdev(model_all_scores), 4) if len(model_all_scores) > 1 else 0.0,
                }

            # Category breakdown
            by_category = {}
            for cat, scores in sorted(category_scores.items()):
                by_category[cat] = {
                    "mean_score": round(mean(scores), 4),
                    "num_samples": len(scores),
                }

            # Split breakdown
            by_split = {}
            for split_name, scores in split_scores.items():
                if scores:
                    by_split[split_name] = {
                        "mean_score": round(mean(scores), 4),
                        "num_samples": len(scores),
                    }

            all_models[model_name] = {
                "model_name": model_name,
                "model_statistics": model_statistics,
                "by_category": by_category,
                "by_split": by_split,
                "tasks": enhanced_tasks,
            }

        # Global statistics
        global_all_scores: list[float] = []
        for model in all_models.values():
            for task in model["tasks"].values():
                for sample in task["samples"].values():
                    global_all_scores.append(sample["score"])

        global_statistics: Dict[str, Any] = {}
        if global_all_scores:
            global_statistics = {
                "total_models": len(all_models),
                "total_tasks": sum(len(m["tasks"]) for m in all_models.values()),
                "total_samples": len(global_all_scores),
                "mean_score": round(mean(global_all_scores), 4),
                "median_score": round(median(global_all_scores), 4),
                "std_score": round(stdev(global_all_scores), 4) if len(global_all_scores) > 1 else 0.0,
                "min_score": round(min(global_all_scores), 4),
                "max_score": round(max(global_all_scores), 4),
            }

        enhanced_summary = {
            "metadata": {
                "evaluator": self.evaluator_name,
                "timestamp": datetime.now().isoformat(),
                "score_range": "0-1",
                "evaluation_type": "rubrics",
                "total_samples": len(global_all_scores),
            },
            "global_statistics": global_statistics,
            "models": all_models,
        }

        output_path = self.eval_output_dir / f"{self.evaluator_name}_summary.json"
        with open(output_path, "w") as f:
            json.dump(enhanced_summary, f, indent=2, default=str)

        logger.info(
            f"Summary rebuilt: {len(global_all_scores)} samples from {len(all_models)} model(s), "
            f"global mean={global_statistics.get('mean_score', 'N/A')}"
        )
        return enhanced_summary
