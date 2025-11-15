import os
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from vmevalkit.eval import HumanEvaluator, GPT4OEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_human_scoring():
    print("\n=== Human Scoring Example ===")
    print(f"Evaluating ENTIRE pilot experiment")
    print("Tasks with existing scorings will be automatically skipped")
    
    scorer = HumanEvaluator(
        experiment_name="pilot_experiment"
    )
    
    print(f"\nLaunching human scoring interface...")
    print("Enter your annotator name in the interface")
    scorer.launch_interface(port=7860, share=True)


def example_gpt4o_scoring():
    print("\n=== GPT-4O Scoring Example ===")
    print("🤖 Evaluating ENTIRE pilot experiment with GPT-4O")
    print("⚠️  Note: This will make API calls to OpenAI and may take time/cost money")
    print("✅ Resume-capable: Interrupted scorings can be continued")
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("❌ Error: Please set OPENAI_API_KEY environment variable")
    
    scorer = GPT4OEvaluator(
        experiment_name="pilot_experiment",
        temperature=0.0
    )
    
    eval_dir = Path("data/scorings/gpt4o-score/pilot_experiment")
    if eval_dir.exists():
        existing_files = list(eval_dir.rglob("*.json"))
        if existing_files:
            print(f"📊 Found {len(existing_files)} existing GPT-4O scorings - will resume from where left off")
    
    print(f"\n🚀 Starting GPT-4O scoring on pilot_experiment...")
    print("💡 Tip: You can interrupt (Ctrl+C) and resume later - progress is saved after each task")
    
    try:
        all_results = scorer.evaluate_all_models()
        
        print("\n📈 GPT-4O EVALUATION RESULTS:")
        total_all = 0
        completed_all = 0
        for model_name, results in all_results.items():
            if "scorings" in results:
                total_tasks = 0
                evaluated_tasks = 0
                for task_type, tasks in results["scorings"].items():
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
        print(f"💾 Results saved to: data/scorings/gpt4o-score/pilot_experiment/")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  GPT-4O scoring interrupted!")
        print(f"💾 Progress has been saved. Run the same command again to resume.")
        print(f"📁 Partial results available in: data/scorings/gpt4o-score/pilot_experiment/")



def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VMEvalKit End-to-End Scoring Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        End-to-End Scoring Examples:
        # Run human scoring (automatically skips already evaluated tasks)
        python score_videos.py human
        
        Note: 
        - Tasks with existing scorings are automatically skipped
        - Annotator name is entered directly in the Gradio interface
        
        # Run GPT-4O scoring on ENTIRE pilot experiment
        python score_videos.py gpt4o

        Note: All methods evaluate the complete pilot experiment (all models, all tasks).
        """
    )
    
    parser.add_argument(
        'method',
        choices=['human', 'gpt4o', 'r-4b'],
        help='Scoring method to use'
    )
    
    
    args = parser.parse_args()
    

    if not Path("data/outputs/pilot_experiment").exists():
        print("Error: pilot_experiment not found. Please run inference first.")
        return
    
    if args.method == "human":
        example_human_scoring()
    elif args.method == "gpt4o":
        example_gpt4o_scoring()


if __name__ == "__main__":
    main()