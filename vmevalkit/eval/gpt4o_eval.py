"""GPT-4O vision model evaluator for VMEvalKit.

Provides single-frame evaluation using OpenAI's GPT-4O vision model.
Can be used standalone or wrapped by MultiFrameEvaluator for multi-frame evaluation.
"""

import json
import os
import base64
import re
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import io
import httpx
from .eval_prompt import TASK_PROMPTS
from .run_selector import select_latest_run

logger = logging.getLogger(__name__)

TASK_GUIDANCE = {
    "chess_task": "Check if the final board position matches the expected position after the correct move.",
    "maze_task": "Verify that the final frame shows a complete path from start to end that matches the expected solution.",
    "rotation_task": "Check if the final rotation angle and position match the expected result.",
    "raven_task": "Verify that the pattern completion in the final frame matches the expected pattern.",
    "sudoku_task": "Check if the numbers placed in the final frame match the expected solution.",
    "object_subtraction_task": "Verify that the specified object(s) have been correctly removed from the scene, while other objects remain unchanged and the scene remains complete.",
    "object_permanence_task": "Verify that the object(s) remain unchanged in position, color, and shape, and the occluder is moved out of the frame.",
    "light_sequence_task": "Verify that the correct lights are on and all other lights are off in the final frame.",
    "sequence_completion_task": "Verify that the sequence is correctly completed with the next element that follows the pattern. The final frame should show the complete sequence with the correct answer element."
}


class GPT4OEvaluator:
    """GPT-4O vision model evaluator for single-frame evaluation."""
    
    def __init__(self, 
                 inference_dir: str,
                 eval_output_dir: str = "./evaluations/gpt4o-eval",
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o",
                 temperature: float = 0.0):
        self.eval_output_dir = Path(eval_output_dir)
        self.inference_dir = Path(inference_dir)
        
        self.eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
        
        self.model = model
        self.temperature = temperature
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        )
    
    def _has_evaluation(self, model_name: str, task_type: str, task_id: str) -> bool:
        """Check if task has already been evaluated."""
        eval_path = self.eval_output_dir / model_name / task_type / task_id
        eval_file = eval_path / "GPT4OEvaluator.json"
        return eval_file.exists()
    
    def extract_final_frame(self, video_path: str) -> np.ndarray:
        """Extract final frame from video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError(f"Video has no frames: {video_path}")
        
        # Try last frame, then second-to-last if needed
        for offset in [1, 2]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - offset)
            ret, frame = cap.read()
            if ret:
                cap.release()
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        cap.release()
        raise ValueError(f"Cannot read final frame from video: {video_path}")
    
    def encode_image(self, image: Union[np.ndarray, str]) -> str:
        """Encode image to base64."""
        pil_image = Image.open(image) if isinstance(image, str) else Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    async def call_gpt4o(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        response = await self.client.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": 1000}
        )
        if response.status_code != 200:
            raise Exception(f"API call failed: {response.status_code} - {response.text}")
        return response.json()
    
    def create_prompt(self, task_type: str) -> str:
        """Create evaluation prompt."""
        return f"""You are evaluating video generation models.
                Compare the final frame of the generated video with the expected ground truth final frame.

                Rate solution correctness on a 1-5 scale:
                1: Completely wrong - no understanding of task
                2: Mostly incorrect - minimal progress toward solution
                3: Partially correct - about half the expected solution
                4: Mostly correct - close to expected result with minor errors
                5: Perfect - matches expected result

                {TASK_PROMPTS.get(task_type, '')}

                Respond in JSON: {{"solution_correctness_score": <1-5>, "explanation": "<brief explanation>"}}
                """
    
    async def evaluate_single_async(self, model_name: str, task_type: str, task_id: str,
                                   video_path: str) -> Dict[str, Any]:
        """Evaluate a single video."""
        final_frame_video = self.extract_final_frame(video_path)
        
        task_dir = Path(video_path).parent.parent
        first_frame_path = task_dir / "question" / "first_frame.png"
        final_frame_path = task_dir / "question" / "final_frame.png"
        prompt_path = task_dir / "question" / "prompt.txt"
        
        if not final_frame_path.exists():
            logger.warning(f"No ground truth final frame for {model_name}/{task_type}/{task_id}")
            return {"error": "No ground truth final frame available", "status": "skipped"}
        
        prompt_text = prompt_path.read_text() if prompt_path.exists() else ""
        
        messages = [
            {"role": "system", "content": self.create_prompt(task_type)},
            {"role": "user", "content": [
                {"type": "text", "text": f"Task: {task_type}\nPrompt: {prompt_text}\n\n1. Input image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.encode_image(str(first_frame_path))}"}},
                {"type": "text", "text": "\n2. Expected final frame:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.encode_image(str(final_frame_path))}"}},
                {"type": "text", "text": "\n3. Actual final frame from video:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.encode_image(final_frame_video)}"}},
                {"type": "text", "text": "\nProvide your evaluation."}
            ]}
        ]
        
        response = await self.call_gpt4o(messages)
        content = response["choices"][0]["message"]["content"]
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            eval_data = json.loads(json_match.group())
            return {
                "solution_correctness_score": eval_data.get("solution_correctness_score", 0),
                "explanation": eval_data.get("explanation", ""),
                "evaluation_type": "final_frame_comparison",
                "status": "completed"
            }
        raise ValueError("Could not parse JSON from GPT-4O response")
    
    def evaluate_single(self, model_name: str, task_type: str, task_id: str,
                       video_path: str) -> Dict[str, Any]:
        """Evaluate a single video (sync wrapper)."""
        async def _single_eval_with_cleanup():
            try:
                return await self.evaluate_single_async(model_name, task_type, task_id, video_path)
            finally:
                await self.client.aclose()
        
        return asyncio.run(_single_eval_with_cleanup())
    
    async def evaluate_model_async(self, model_name: str, close_client: bool = False) -> Dict[str, Any]:
        """Evaluate all results for a model (async version)."""
        try:
            model_dir = self.inference_dir / model_name
            if not model_dir.exists():
                raise ValueError(f"Model directory not found: {model_dir}")
            
            results = {"model_name": model_name, "evaluations": {}}
            total_tasks = 0
            skipped_tasks = 0
            evaluated_tasks = 0
            failed_tasks = 0
            
            # Support both 2-layer and 3-layer directory structures
            # 2-layer: model/task_type/task_id
            # 3-layer: model/generator/task_type/task_id
            
            for first_level_dir in model_dir.iterdir():
                if not first_level_dir.is_dir(): continue
                
                # Check if this is a generator directory (3-layer) or task_type directory (2-layer)
                # Generator directories have names like "G-1_xxx_data-generator" or "G-1_xxx-data-generator"
                # Support both underscore and hyphen before "data-generator"
                is_generator_dir = (
                    first_level_dir.name.startswith("G-") and 
                    ("_data-generator" in first_level_dir.name or 
                     "-data-generator" in first_level_dir.name)
                )
                
                if is_generator_dir:
                    # 3-layer structure: model/generator/task_type/task_id
                    generator_name = first_level_dir.name
                    logger.info(f"Processing generator: {generator_name}")
                    
                    for task_type_dir in first_level_dir.iterdir():
                        if not task_type_dir.is_dir(): continue
                        task_type = task_type_dir.name
                        
                        # Use full path as task_type to keep generator info
                        full_task_type = f"{generator_name}/{task_type}"
                        results["evaluations"][full_task_type] = {}
                        
                        for task_dir in task_type_dir.iterdir():
                            if not task_dir.is_dir(): continue
                            task_id = task_dir.name
                            total_tasks += 1
                            
                            # Check if already evaluated
                            if self._has_evaluation(model_name, full_task_type, task_id):
                                logger.debug(f"Skipping {model_name}/{full_task_type}/{task_id} - already evaluated")
                                skipped_tasks += 1
                                continue
                            
                            run_dir = select_latest_run(task_dir)
                            if not run_dir:
                                logger.warning(f"No output for {model_name}/{full_task_type}/{task_id}")
                                continue

                            video_files = sorted((run_dir / "video").glob("*.mp4"))
                            if not video_files:
                                logger.warning(f"No video in {run_dir / 'video'}")
                                continue
                            
                            try:
                                logger.info(f"Evaluating {model_name}/{full_task_type}/{task_id}")
                                eval_result = await self.evaluate_single_async(model_name, full_task_type, task_id, str(video_files[0]))
                                results["evaluations"][full_task_type][task_id] = eval_result
                                
                                # Save immediately after each evaluation
                                self._save_single_result(model_name, full_task_type, task_id, eval_result)
                                evaluated_tasks += 1
                                
                            except Exception as e:
                                logger.error(f"Error evaluating {model_name}/{full_task_type}/{task_id}: {e}")
                                failed_tasks += 1
                                results["evaluations"][full_task_type][task_id] = {
                                    "status": "failed",
                                    "error": str(e)
                                }
                else:
                    # 2-layer structure: model/task_type/task_id (backward compatibility)
                    task_type_dir = first_level_dir
                    task_type = task_type_dir.name
                    results["evaluations"][task_type] = {}
                    
                    for task_dir in task_type_dir.iterdir():
                        if not task_dir.is_dir(): continue
                        task_id = task_dir.name
                        total_tasks += 1
                        
                        # Check if already evaluated
                        if self._has_evaluation(model_name, task_type, task_id):
                            logger.debug(f"Skipping {model_name}/{task_type}/{task_id} - already evaluated")
                            skipped_tasks += 1
                            continue
                        
                        run_dir = select_latest_run(task_dir)
                        if not run_dir:
                            logger.warning(f"No output for {model_name}/{task_type}/{task_id}")
                            continue

                        video_files = sorted((run_dir / "video").glob("*.mp4"))
                        if not video_files:
                            logger.warning(f"No video in {run_dir / 'video'}")
                            continue
                        
                        try:
                            logger.info(f"Evaluating {model_name}/{task_type}/{task_id}")
                            eval_result = await self.evaluate_single_async(model_name, task_type, task_id, str(video_files[0]))
                            results["evaluations"][task_type][task_id] = eval_result
                            
                            # Save immediately after each evaluation
                            self._save_single_result(model_name, task_type, task_id, eval_result)
                            evaluated_tasks += 1
                            
                        except Exception as e:
                            logger.error(f"Error evaluating {model_name}/{task_type}/{task_id}: {e}")
                            results["evaluations"][task_type][task_id] = {"error": str(e), "status": "failed"}
                            failed_tasks += 1
            
            logger.info(f"GPT-4O Evaluation Summary for {model_name}:")
            logger.info(f"  - Total tasks: {total_tasks}")
            logger.info(f"  - Already completed (skipped): {skipped_tasks}")
            logger.info(f"  - Newly evaluated: {evaluated_tasks}")
            logger.info(f"  - Failed: {failed_tasks}")
            
            return results
        finally:
            if close_client:
                await self.client.aclose()
    
    def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """Evaluate all results for a model."""
        return asyncio.run(self.evaluate_model_async(model_name, close_client=True))
    
    async def evaluate_all_models_async(self) -> Dict[str, Any]:
        """Evaluate all models in experiment (async version)."""
        try:
            # Run evaluation for all models
            for model_dir in self.inference_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    logger.info(f"Evaluating model: {model_name}")
                    await self.evaluate_model_async(model_name)
            
            # Rebuild complete summary from all evaluation files
            logger.info("Rebuilding complete summary from all evaluation files...")
            all_results = self._rebuild_summary_from_files()
            
            return all_results
        finally:
            await self.client.aclose()
    
    def evaluate_all_models(self) -> Dict[str, Any]:
        """Evaluate all models in experiment."""
        return asyncio.run(self.evaluate_all_models_async())
    
    def _save_single_result(self, model_name: str, task_type: str, task_id: str, eval_result: Dict[str, Any]):
        """Save a single evaluation result immediately (for resume support)."""
        task_output_dir = self.eval_output_dir / model_name / task_type / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(task_output_dir / "GPT4OEvaluator.json", 'w') as f:
            json.dump({
                "metadata": {
                    "evaluator": "GPT4OEvaluator",
                    "timestamp": datetime.now().isoformat(),
                    "model_name": model_name,
                    "task_type": task_type,
                    "task_id": task_id
                },
                "result": eval_result
            }, f, indent=2)
        
        logger.debug(f"Saved evaluation for {model_name}/{task_type}/{task_id}")
    
    def _save_results(self, model_name: str, results: Dict[str, Any]):
        """Save evaluation results (legacy method - now individual saves are preferred)."""
        
        for task_type, task_results in results["evaluations"].items():
            for task_id, eval_result in task_results.items():
                # Only save if not already saved by _save_single_result
                if not self._has_evaluation(model_name, task_type, task_id):
                    self._save_single_result(model_name, task_type, task_id, eval_result)
        
        logger.info(f"Completed evaluation results for {model_name}")
    
    @staticmethod
    def _extract_prefix_and_number(text: str):
        """
        Extract prefix and number from text for sorting.
        Supports formats like G-1, O-1, K-1, etc.
        
        Args:
            text: Text to extract from (e.g., "G-1_xxx/task")
        
        Returns:
            tuple: (prefix, number, original_text) for sorting
                   e.g., "G-1_xxx" -> ('G', 1, 'G-1_xxx')
        
        Examples:
            >>> _extract_prefix_and_number("G-1_task/subtask")
            ('G', 1, 'G-1_task/subtask')
            >>> _extract_prefix_and_number("O-5_task/subtask")
            ('O', 5, 'O-5_task/subtask')
        """
        match = re.match(r'([A-Z])-(\d+)', text)
        if match:
            prefix = match.group(1)
            number = int(match.group(2))
            return (prefix, number, text)
        # If no match, put at the end (~ comes after all letters)
        return ('~', 0, text)
    
    def _rebuild_summary_from_files(self) -> Dict[str, Any]:
        """
        Rebuild complete enhanced summary from all evaluation files.
        
        This method scans all GPT4OEvaluator.json files and generates an enhanced
        summary with comprehensive statistics including:
        - Global statistics (overall performance across all models)
        - Model-level statistics (per-model performance)
        - Task-level statistics (per-task performance)
        - Complete sample data with score distributions
        
        Returns:
            dict: Enhanced summary with statistics and evaluation results
        """
        from statistics import mean, median, stdev
        from collections import Counter
        
        logger.info("Rebuilding enhanced summary from all evaluation files...")
        
        all_models = {}
        
        # Iterate through each model directory
        for model_dir in self.eval_output_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            evaluations = {}
            
            # Scan all GPT4OEvaluator.json files
            eval_files = list(model_dir.rglob("GPT4OEvaluator.json"))
            logger.info(f"Found {len(eval_files)} evaluation files for {model_name}")
            
            for eval_file in eval_files:
                try:
                    with open(eval_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract hierarchy from file path
                    parts = eval_file.relative_to(model_dir).parts
                    
                    if len(parts) >= 3:
                        generator = parts[0]
                        task_type = parts[1]
                        task_id = parts[2]
                        
                        full_task_type = f"{generator}/{task_type}"
                        
                        if full_task_type not in evaluations:
                            evaluations[full_task_type] = {}
                        
                        result = data.get('result', {})
                        evaluations[full_task_type][task_id] = result
                        
                except Exception as e:
                    logger.warning(f"Could not load {eval_file}: {e}")
            
            if evaluations:
                # Sort by prefix (G, K, O, ...) then by number
                sorted_evaluations = dict(sorted(
                    evaluations.items(),
                    key=lambda x: self._extract_prefix_and_number(x[0])
                ))
                
                # Calculate statistics for each task
                enhanced_tasks = {}
                model_all_scores = []
                
                for task_type, task_samples in sorted_evaluations.items():
                    task_scores = []
                    simplified_samples = {}
                    
                    for sample_id, sample_data in task_samples.items():
                        score = sample_data.get('solution_correctness_score', 0)
                        task_scores.append(score)
                        model_all_scores.append(score)
                        
                        # Simplified sample data for summary
                        simplified_samples[sample_id] = {
                            "score": score,
                            "status": sample_data.get('status', 'unknown'),
                            "explanation_preview": sample_data.get('explanation', '')[:100] + "..." 
                                if len(sample_data.get('explanation', '')) > 100 
                                else sample_data.get('explanation', '')
                        }
                    
                    # Calculate task statistics
                    if task_scores:
                        score_dist = Counter(task_scores)
                        task_statistics = {
                            "total_samples": len(task_scores),
                            "evaluated_samples": len(task_scores),
                            "completion_rate": 1.0,
                            "mean_score": round(mean(task_scores), 2),
                            "median_score": round(median(task_scores), 2),
                            "std_score": round(stdev(task_scores), 2) if len(task_scores) > 1 else 0.0,
                            "min_score": min(task_scores),
                            "max_score": max(task_scores),
                            "score_distribution": {str(i): score_dist.get(i, 0) for i in range(6)}
                        }
                    else:
                        task_statistics = {
                            "total_samples": 0,
                            "evaluated_samples": 0,
                            "completion_rate": 0.0,
                            "status": "pending"
                        }
                    
                    enhanced_tasks[task_type] = {
                        "task_statistics": task_statistics,
                        "samples": simplified_samples
                    }
                
                # Calculate model-level statistics
                if model_all_scores:
                    model_score_dist = Counter(model_all_scores)
                    model_statistics = {
                        "total_samples": len(model_all_scores),
                        "evaluated_samples": len(model_all_scores),
                        "mean_score": round(mean(model_all_scores), 2),
                        "median_score": round(median(model_all_scores), 2),
                        "std_score": round(stdev(model_all_scores), 2) if len(model_all_scores) > 1 else 0.0,
                        "score_distribution": {str(i): model_score_dist.get(i, 0) for i in range(6)}
                    }
                else:
                    model_statistics = {}
                
                all_models[model_name] = {
                    "model_name": model_name,
                    "model_statistics": model_statistics,
                    "tasks": enhanced_tasks
                }
        
        # Calculate global statistics
        global_all_scores = []
        for model in all_models.values():
            for task in model['tasks'].values():
                for sample in task['samples'].values():
                    global_all_scores.append(sample['score'])
        
        if global_all_scores:
            global_score_dist = Counter(global_all_scores)
            global_statistics = {
                "total_models": len(all_models),
                "total_tasks": sum(len(m['tasks']) for m in all_models.values()),
                "total_samples": len(global_all_scores),
                "evaluated_samples": len(global_all_scores),
                "mean_score": round(mean(global_all_scores), 2),
                "median_score": round(median(global_all_scores), 2),
                "std_score": round(stdev(global_all_scores), 2) if len(global_all_scores) > 1 else 0.0,
                "min_score": min(global_all_scores),
                "max_score": max(global_all_scores),
                "score_distribution": {str(i): global_score_dist.get(i, 0) for i in range(6)}
            }
        else:
            global_statistics = {}
        
        # Build enhanced summary structure
        enhanced_summary = {
            "metadata": {
                "evaluator": "GPT4OEvaluator",
                "timestamp": datetime.now().isoformat(),
                "enhanced_version": True,
                "total_samples": len(global_all_scores)
            },
            "global_statistics": global_statistics,
            "models": all_models
        }
        
        # Save enhanced summary
        output_path = self.eval_output_dir / "GPT4OEvaluator_summary.json"
        with open(output_path, 'w') as f:
            json.dump(enhanced_summary, f, indent=2)
        
        logger.info(f"Enhanced summary rebuilt successfully: {len(global_all_scores)} samples from {len(all_models)} model(s)")
        logger.info(f"Global mean score: {global_statistics.get('mean_score', 'N/A')}")
        
        return enhanced_summary
