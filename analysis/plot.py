#!/usr/bin/env python3
"""
VMEvalKit Analysis Tool

Analyzes evaluation results to show model performance by domain and overall rankings.
Only scores 4 and 5 are considered "correct" (successful).

Usage:
    python analysis/plot.py --eval-folder data/evaluations/human-eval/
    python analysis/plot.py --eval-folder data/evaluations/gpt4o-eval/
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import numpy as np
from datetime import datetime

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_evaluation_data(eval_folder: Path) -> list:
    """Load all evaluation JSON files from the specified folder."""
    evaluations = []
    
    for json_file in eval_folder.rglob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract relevant information
            if "metadata" in data and "result" in data:
                eval_data = {
                    "model_name": data["metadata"].get("model_name", "unknown"),
                    "task_type": data["metadata"].get("task_type", "unknown"),
                    "task_id": data["metadata"].get("task_id", "unknown"),
                    "score": data["result"].get("solution_correctness_score", 0),
                    "evaluator": data["metadata"].get("evaluator", "unknown"),
                    "annotator": data["metadata"].get("annotator", "unknown")
                }
                evaluations.append(eval_data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {json_file}: {e}")
    
    return evaluations

def calculate_domain_performance(evaluations: list) -> pd.DataFrame:
    """Calculate performance by model and domain (task_type)."""
    results = []
    
    # Group by model and domain
    grouped = defaultdict(lambda: defaultdict(list))
    for eval_data in evaluations:
        model = eval_data["model_name"]
        domain = eval_data["task_type"].replace("_task", "")  # Remove "_task" suffix
        score = eval_data["score"]
        grouped[model][domain].append(score)
    
    # Calculate performance metrics
    for model, domains in grouped.items():
        for domain, scores in domains.items():
            total_tasks = len(scores)
            if total_tasks > 0:
                # Count scores 4 and 5 as correct
                correct_tasks = sum(1 for score in scores if score >= 4)
                success_rate = (correct_tasks / total_tasks) * 100
                avg_score = np.mean(scores)
                
                results.append({
                    "model": model,
                    "domain": domain,
                    "total_tasks": total_tasks,
                    "correct_tasks": correct_tasks,
                    "success_rate": success_rate,
                    "average_score": avg_score,
                    "scores": scores
                })
    
    return pd.DataFrame(results)

def calculate_overall_performance(evaluations: list) -> pd.DataFrame:
    """Calculate overall performance ranking for all models."""
    results = []
    
    # Group by model
    grouped = defaultdict(list)
    for eval_data in evaluations:
        model = eval_data["model_name"]
        score = eval_data["score"]
        grouped[model].append(score)
    
    # Calculate overall metrics
    for model, scores in grouped.items():
        total_tasks = len(scores)
        if total_tasks > 0:
            correct_tasks = sum(1 for score in scores if score >= 4)
            success_rate = (correct_tasks / total_tasks) * 100
            avg_score = np.mean(scores)
            
            results.append({
                "model": model,
                "total_tasks": total_tasks,
                "correct_tasks": correct_tasks,
                "success_rate": success_rate,
                "average_score": avg_score
            })
    
    # Sort by success rate
    df = pd.DataFrame(results)
    df = df.sort_values("success_rate", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    
    return df

def create_visualizations(domain_df: pd.DataFrame, overall_df: pd.DataFrame, output_path: str):
    """Create comprehensive visualizations and save to specified path."""
    
    # Create figures directory if it doesn't exist
    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # Create figure with 3 essential plots
    fig = plt.figure(figsize=(18, 6))
    
    # 1. Success Rate Heatmap
    plt.subplot(1, 3, 1)
    pivot_success = domain_df.pivot(index="model", columns="domain", values="success_rate")
    sns.heatmap(pivot_success, annot=True, fmt='.1f', cmap='RdYlGn', 
                vmin=0, vmax=100, cbar_kws={'label': 'Success Rate (%)'})
    plt.title("Success Rate by Model and Domain\n(Scores 4-5 = Correct)", fontsize=14, fontweight='bold')
    plt.xlabel("Domain", fontweight='bold')
    plt.ylabel("Model", fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 2. Overall Model Ranking
    plt.subplot(1, 3, 2)
    colors = plt.cm.RdYlGn(overall_df["success_rate"] / 100)
    bars = plt.barh(range(len(overall_df)), overall_df["success_rate"], color=colors)
    plt.yticks(range(len(overall_df)), overall_df["model"])
    plt.xlabel("Success Rate (%)", fontweight='bold')
    plt.title("Overall Model Ranking\n(All Domains Combined)", fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, overall_df["success_rate"])):
        plt.text(rate + 1, i, f'{rate:.1f}%', va='center', ha='left', fontweight='bold')
    
    plt.xlim(0, 100)
    plt.grid(axis='x', alpha=0.3)
    
    # 3. Domain Difficulty Ranking
    plt.subplot(1, 3, 3)
    domain_means = domain_df.groupby("domain")["success_rate"].mean().sort_values(ascending=True)
    colors_domain = plt.cm.viridis(np.linspace(0, 1, len(domain_means)))
    bars = plt.barh(range(len(domain_means)), domain_means.values, color=colors_domain)
    plt.yticks(range(len(domain_means)), domain_means.index)
    plt.xlabel("Average Success Rate (%)", fontweight='bold')
    plt.title("Domain Difficulty Ranking\n(Hardest to Easiest)", fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    for i, rate in enumerate(domain_means.values):
        plt.text(rate + 1, i, f'{rate:.1f}%', va='center', ha='left', fontweight='bold')
    
    plt.xlim(0, max(domain_means) * 1.2)
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_path}")
    plt.close()  # Close the figure to free memory

def print_detailed_results(domain_df: pd.DataFrame, overall_df: pd.DataFrame):
    """Print detailed text results."""
    
    print("\n" + "="*80)
    print("🎯 VMEVALKIT EVALUATION ANALYSIS RESULTS")
    print("="*80)
    
    # Overall ranking
    print("\n📈 OVERALL MODEL RANKING (All Domains Combined)")
    print("-" * 50)
    for _, row in overall_df.iterrows():
        print(f"{row['rank']:2d}. {row['model']:<25} | "
              f"Success: {row['success_rate']:5.1f}% ({row['correct_tasks']:3d}/{row['total_tasks']:3d}) | "
              f"Avg Score: {row['average_score']:.2f}")
    
    # Domain performance
    print(f"\n🎲 PERFORMANCE BY DOMAIN (Scores 4-5 = Correct)")
    print("-" * 50)
    
    for domain in sorted(domain_df["domain"].unique()):
        print(f"\n📊 {domain.upper()} TASKS:")
        domain_data = domain_df[domain_df["domain"] == domain].sort_values("success_rate", ascending=False)
        
        for _, row in domain_data.iterrows():
            print(f"  • {row['model']:<25} | "
                  f"{row['success_rate']:5.1f}% ({row['correct_tasks']:2d}/{row['total_tasks']:2d}) | "
                  f"Avg: {row['average_score']:.2f}")
    
    # Domain difficulty ranking
    print(f"\n🏆 DOMAIN DIFFICULTY RANKING (Easiest to Hardest)")
    print("-" * 50)
    domain_difficulty = domain_df.groupby("domain")["success_rate"].mean().sort_values(ascending=False)
    
    for rank, (domain, avg_rate) in enumerate(domain_difficulty.items(), 1):
        difficulty = "🟢 Easy" if avg_rate > 70 else "🟡 Medium" if avg_rate > 40 else "🔴 Hard"
        print(f"{rank}. {domain.upper():<10} | {avg_rate:5.1f}% average success | {difficulty}")
    
    # Summary statistics
    print(f"\n📊 SUMMARY STATISTICS")
    print("-" * 50)
    total_evaluations = sum(overall_df["total_tasks"])
    total_correct = sum(overall_df["correct_tasks"])
    overall_success = (total_correct / total_evaluations) * 100 if total_evaluations > 0 else 0
    
    print(f"Total Evaluations: {total_evaluations:,}")
    print(f"Total Correct (4-5): {total_correct:,}")
    print(f"Overall Success Rate: {overall_success:.1f}%")
    print(f"Models Evaluated: {len(overall_df)}")
    print(f"Domains Covered: {len(domain_df['domain'].unique())}")
    
    best_model = overall_df.iloc[0]["model"]
    best_rate = overall_df.iloc[0]["success_rate"]
    print(f"🏆 Best Performing Model: {best_model} ({best_rate:.1f}%)")
    
    hardest_domain = domain_difficulty.index[-1]
    hardest_rate = domain_difficulty.iloc[-1]
    print(f"🔴 Most Challenging Domain: {hardest_domain} ({hardest_rate:.1f}% avg)")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze VMEvalKit evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analysis/plot.py --eval-folder data/evaluations/human-eval/
  python analysis/plot.py --eval-folder data/evaluations/gpt4o-eval/
        """
    )
    
    parser.add_argument("--eval-folder", required=True, type=str,
                      help="Path to evaluation folder (e.g., data/evaluations/human-eval/)")
    
    args = parser.parse_args()
    
    eval_folder = Path(args.eval_folder)
    if not eval_folder.exists():
        print(f"❌ Error: Evaluation folder not found: {eval_folder}")
        return
    
    # Load and analyze data
    print(f"📂 Loading evaluations from: {eval_folder}")
    evaluations = load_evaluation_data(eval_folder)
    
    if not evaluations:
        print(f"❌ No evaluation files found in {eval_folder}")
        return
    
    print(f"✅ Loaded {len(evaluations)} evaluations")
    
    # Calculate performance metrics
    domain_df = calculate_domain_performance(evaluations)
    overall_df = calculate_overall_performance(evaluations)
    
    # Print detailed results
    print_detailed_results(domain_df, overall_df)
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    
    # Extract evaluation type from folder path
    eval_type = "unknown"
    if "human-eval" in str(eval_folder):
        eval_type = "human-eval"
    elif "gpt4o-eval" in str(eval_folder):
        eval_type = "gpt4o-eval"
    elif "custom-eval" in str(eval_folder):
        eval_type = "custom-eval"
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"analysis/figures/vmevalkit_{eval_type}_{timestamp}.png"
    
    create_visualizations(domain_df, overall_df, output_path)

if __name__ == "__main__":
    main()