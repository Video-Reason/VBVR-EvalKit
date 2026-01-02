import os
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from vmevalkit.eval import HumanEvaluator, GPT4OEvaluator, InternVLEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_human_scoring(inference_dir: str, eval_output_dir: str):
    print("\n=== Human Scoring Example ===")
    print(f"Evaluating inference results from: {inference_dir}")
    print("Tasks with existing scorings will be automatically skipped")
    
    scorer = HumanEvaluator(
        inference_dir=inference_dir,
        eval_output_dir=eval_output_dir
    )
    
    print(f"\nLaunching human scoring interface...")
    print("Enter your annotator name in the interface")
    scorer.launch_interface(port=7860, share=True)


def example_gpt4o_scoring(inference_dir: str, eval_output_dir: str):
    print("\n=== GPT-4O Scoring Example ===")
    print(f"🤖 Evaluating inference results from: {inference_dir}")
    print("⚠️  Note: This will make API calls to OpenAI and may take time/cost money")
    print("✅ Resume-capable: Interrupted scorings can be continued")
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ Error: Please set OPENAI_API_KEY environment variable")
    
    scorer = GPT4OEvaluator(
        inference_dir=inference_dir,
        eval_output_dir=eval_output_dir,
        temperature=0.0
    )
    
    eval_dir = Path(eval_output_dir)
    if eval_dir.exists():
        existing_files = list(eval_dir.rglob("*.json"))
        if existing_files:
            print(f"📊 Found {len(existing_files)} existing GPT-4O scorings - will resume from where left off")
    
    print(f"\n🚀 Starting GPT-4O scoring on inference results...")
    print("💡 Tip: You can interrupt (Ctrl+C) and resume later - progress is saved after each task")
    
    try:
        all_results = scorer.evaluate_all_models()
        
        print("\n📈 GPT-4O EVALUATION RESULTS:")
        total_all = 0
        completed_all = 0
        for model_name, results in all_results.items():
            if "evaluations" in results:
                total_tasks = 0
                evaluated_tasks = 0
                for task_type, tasks in results["evaluations"].items():
                    for task_id, result in tasks.items():
                        total_tasks += 1
                        if "error" not in result and result.get("status") != "failed":
                            evaluated_tasks += 1
                
                total_all += total_tasks
                completed_all += evaluated_tasks
                
                status = "✅ Complete" if evaluated_tasks == total_tasks else f"🔄 {evaluated_tasks}/{total_tasks}"
                print(f"  • {model_name}: {status}")
        
        print(f"\n🎉 GPT-4O EVALUATION COMPLETE!")
        print(f"📊 Total: {completed_all}/{total_all} tasks evaluated successfully")
        print(f"💾 Results saved to: {eval_output_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  GPT-4O scoring interrupted!")
        print(f"💾 Progress has been saved. Run the same command again to resume.")
        print(f"📁 Partial results available in: {eval_output_dir}")


def example_internvl_scoring(inference_dir: str, eval_output_dir: str):
    print("\n=== InternVL Scoring Example ===")
    print(f"🤖 Evaluating inference results from: {inference_dir}")
    print("⚠️  Note: This will make API calls to local InternVL server")
    print("✅ Resume-capable: Interrupted scorings can be continued")
    
    api_key = os.getenv("VISION_API_KEY", "YOUR_API_KEY")
    base_url = os.getenv("VISION_API_BASE", "http://0.0.0.0:23333/v1")
    
    if api_key == "YOUR_API_KEY":
        print("⚠️  Warning: Using default API key. Set VISION_API_KEY if needed.")
    
    scorer = InternVLEvaluator(
        inference_dir=inference_dir,
        eval_output_dir=eval_output_dir,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0
    )
    
    eval_dir = Path(eval_output_dir)
    if eval_dir.exists():
        existing_files = list(eval_dir.rglob("*.json"))
        if existing_files:
            print(f"📊 Found {len(existing_files)} existing InternVL scorings - will resume from where left off")
    
    print(f"\n🚀 Starting InternVL scoring on inference results...")
    print(f"🌐 Base URL: {base_url}")
    print("💡 Tip: You can interrupt (Ctrl+C) and resume later - progress is saved after each task")
    
    try:
        all_results = scorer.evaluate_all_models()
        
        print("\n📈 InternVL EVALUATION RESULTS:")
        total_all = 0
        completed_all = 0
        for model_name, results in all_results.items():
            if "evaluations" in results:
                total_tasks = 0
                evaluated_tasks = 0
                for task_type, tasks in results["evaluations"].items():
                    for task_id, result in tasks.items():
                        total_tasks += 1
                        if "error" not in result and result.get("status") != "failed":
                            evaluated_tasks += 1
                
                total_all += total_tasks
                completed_all += evaluated_tasks
                
                status = "✅ Complete" if evaluated_tasks == total_tasks else f"🔄 {evaluated_tasks}/{total_tasks}"
                print(f"  • {model_name}: {status}")
        
        print(f"\n🎉 InternVL EVALUATION COMPLETE!")
        print(f"📊 Total: {completed_all}/{total_all} tasks evaluated successfully")
        print(f"💾 Results saved to: {eval_output_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  InternVL scoring interrupted!")
        print(f"💾 Progress has been saved. Run the same command again to resume.")
        print(f"📁 Partial results available in: {eval_output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VMEvalKit End-to-End Scoring Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        End-to-End Scoring Examples:
        # Run human scoring with default paths
        python score_videos.py human --inference-dir ./outputs --eval-output-dir ./evaluations
        
        # Run GPT-4O scoring on inference results
        python score_videos.py gpt4o --inference-dir ~/experiments/run1 --eval-output-dir ~/experiments/run1_scores
        
        # Run InternVL scoring on inference results  
        python score_videos.py internvl --inference-dir ./outputs --eval-output-dir ./evaluations

        Note: 
        - Tasks with existing scorings are automatically skipped
        - Annotator name is entered directly in the Gradio interface (for human scoring)
        """
    )
    
    parser.add_argument(
        'method',
        choices=['human', 'gpt4o', 'internvl'],
        help='Scoring method to use'
    )
    
    parser.add_argument(
        "--inference-dir",
        type=str,
        default="./outputs",
        help="Path to inference outputs to evaluate (default: ./outputs)"
    )
    
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default="./evaluations",
        help="Path for evaluation results (default: ./evaluations)"
    )
    
    args = parser.parse_args()
    
    inference_dir = Path(args.inference_dir)
    if not inference_dir.exists():
        print(f"Error: Inference directory not found at {inference_dir}. Please run inference first.")
        return
    
    if args.method == "human":
        example_human_scoring(args.inference_dir, args.eval_output_dir)
    elif args.method == "gpt4o":
        example_gpt4o_scoring(args.inference_dir, args.eval_output_dir)
    elif args.method == "internvl":
        example_internvl_scoring(args.inference_dir, args.eval_output_dir)


if __name__ == "__main__":
    main()