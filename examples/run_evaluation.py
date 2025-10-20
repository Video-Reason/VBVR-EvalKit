#!/usr/bin/env python3
"""
VMEvalKit Evaluation Runner

This script provides easy access to VMEvalKit's evaluation methods:
- Human evaluation with Gradio interface
- GPT-4O automatic evaluation
- Custom evaluation examples

Usage:
    python run_evaluation.py human
    python run_evaluation.py human --no-skip-existing
    python run_evaluation.py human --check-eval-types human-eval
    python run_evaluation.py gpt4o
    python run_evaluation.py custom
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vmevalkit.eval import HumanEvaluator, GPT4OEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_human_evaluation():
    """Example of running human evaluation on entire pilot experiment."""
    print("\n=== Human Evaluation Example ===")
    print(f"Evaluating ENTIRE pilot experiment")
    print("Tasks with existing evaluations will be automatically skipped")
    
    # Create evaluator
    evaluator = HumanEvaluator(
        experiment_name="pilot_experiment"
    )
    
    # Launch interface
    print(f"\nLaunching human evaluation interface...")
    print("Enter your annotator name in the interface")
    evaluator.launch_interface(port=7860, share=True)


def example_gpt4o_evaluation():
    """Example of running GPT-4O evaluation on entire pilot experiment."""
    print("\n=== GPT-4O Evaluation Example ===")
    print("Evaluating ENTIRE pilot experiment with GPT-4O")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY environment variable")
        return
    
    # Create evaluator
    evaluator = GPT4OEvaluator(
        experiment_name="pilot_experiment",
        temperature=0.1
    )
    
    # Evaluate all models and tasks
    print("\nEvaluating ALL models and ALL tasks in pilot_experiment...")
    all_results = evaluator.evaluate_all_models()
    
    # Print basic counts for each model
    for model_name, results in all_results.items():
        if "evaluations" in results:
            total_tasks = 0
            evaluated_tasks = 0
            for task_type, tasks in results["evaluations"].items():
                for task_id, result in tasks.items():
                    total_tasks += 1
                    if "error" not in result:
                        evaluated_tasks += 1
            
            print(f"\n{model_name}:")
            print(f"  - Tasks evaluated: {evaluated_tasks}/{total_tasks}")
    
    print("\nEND-TO-END EVALUATION COMPLETE!")


def example_custom_evaluation():
    """Example of creating a custom end-to-end evaluation."""
    print("\n=== Custom Evaluation Example ===")
    print("Creating custom evaluator for ENTIRE pilot experiment")
    
    # Create a simple custom evaluator without base class
    class SimpleEvaluator:
        """A simple custom evaluator for demonstration."""
        
        def __init__(self, output_dir="data/evaluations", experiment_name="pilot_experiment"):
            self.output_dir = Path(output_dir)
            self.experiment_name = experiment_name
            self.experiment_dir = Path("data/outputs") / experiment_name
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        def evaluate_single(self, model_name, task_type, task_id, video_path):
            """Evaluate a single video."""
            import random
            
            # Random score for demo
            score = random.randint(1, 5)
            
            return {
                "solution_correctness_score": score,
                "explanation": f"Demo evaluation: solution scored {score}/5",
                "status": "completed"
            }
        
        def evaluate_all_models(self):
            """Evaluate all models in the experiment."""
            all_results = {}
            
            for model_dir in self.experiment_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                    
                model_name = model_dir.name
                print(f"Evaluating model: {model_name}")
                
                results = {"model_name": model_name, "evaluations": {}}
                
                for task_type_dir in model_dir.iterdir():
                    if not task_type_dir.is_dir():
                        continue
                    
                    task_type = task_type_dir.name
                    results["evaluations"][task_type] = {}
                    
                    for task_dir in task_type_dir.iterdir():
                        if not task_dir.is_dir():
                            continue
                        
                        task_id = task_dir.name
                        output_dirs = list(task_dir.iterdir())
                        
                        if output_dirs:
                            output_dir = output_dirs[0]
                            video_files = list((output_dir / "video").glob("*.mp4"))
                            
                            if video_files:
                                eval_result = self.evaluate_single(
                                    model_name, task_type, task_id, str(video_files[0])
                                )
                                results["evaluations"][task_type][task_id] = eval_result
                                
                                # Save individual result
                                self._save_result(model_name, task_type, task_id, eval_result)
                
                all_results[model_name] = results
            
            return all_results
        
        def _save_result(self, model_name, task_type, task_id, eval_result):
            """Save evaluation result."""
            output_path = self.output_dir / self.experiment_name / model_name / task_type / task_id
            output_path.mkdir(parents=True, exist_ok=True)
            
            with open(output_path / "SimpleEvaluator.json", 'w') as f:
                json.dump({
                    "metadata": {
                        "evaluator": "SimpleEvaluator",
                        "timestamp": datetime.now().isoformat(),
                        "model_name": model_name,
                        "task_type": task_type,
                        "task_id": task_id
                    },
                    "result": eval_result
                }, f, indent=2)
    
    # Use the custom evaluator
    evaluator = SimpleEvaluator(experiment_name="pilot_experiment")
    
    # Evaluate ALL models and tasks
    print("Running custom evaluation on entire pilot experiment...")
    all_results = evaluator.evaluate_all_models()
    
    # Count results across all models
    total_tasks_all = 0
    evaluated_tasks_all = 0
    for model_name, results in all_results.items():
        if "evaluations" in results:
            for task_type, tasks in results["evaluations"].items():
                for task_id, result in tasks.items():
                    total_tasks_all += 1
                    if "error" not in result:
                        evaluated_tasks_all += 1
    
    print(f"CUSTOM END-TO-END EVALUATION COMPLETE!")
    print(f"Total evaluated: {evaluated_tasks_all}/{total_tasks_all} tasks across all models")


def main():
    """Main function to run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VMEvalKit End-to-End Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
End-to-End Evaluation Examples:
  # Run human evaluation (automatically skips already evaluated tasks)
  python run_evaluation.py human
  
  Note: 
  - Tasks with existing evaluations are automatically skipped
  - Annotator name is entered directly in the Gradio interface
  
  # Run GPT-4O evaluation on ENTIRE pilot experiment
  python run_evaluation.py gpt4o
  
  # Demonstrate custom evaluator
  python run_evaluation.py custom

Note: All methods evaluate the complete pilot experiment (all models, all tasks).
        """
    )
    
    parser.add_argument(
        'method',
        choices=['human', 'gpt4o', 'custom'],
        help='Evaluation method to use'
    )
    
    
    args = parser.parse_args()
    
    # Check if pilot_experiment exists
    if not Path("data/outputs/pilot_experiment").exists():
        print("Error: pilot_experiment not found. Please run inference first.")
        return
    
    # Run the selected evaluation method
    if args.method == "human":
        example_human_evaluation()
    elif args.method == "gpt4o":
        example_gpt4o_evaluation()
    elif args.method == "custom":
        example_custom_evaluation()


if __name__ == "__main__":
    main()