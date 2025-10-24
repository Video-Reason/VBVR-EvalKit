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
import matplotlib.colors as mcolors

# Set up plotting style with sophisticated color scheme
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Define color schemes
MAIN_PALETTE = sns.color_palette("deep", 10)  # For distinct models
SUCCESS_CMAP = plt.cm.RdYlGn  # Red-Yellow-Green for success rates
DOMAIN_COLORS = {
    'chess': '#2E86AB',     # Deep blue
    'maze': '#A23B72',      # Purple
    'raven': '#F18F01',     # Orange
    'rotation': '#C73E1D',  # Red
    'sudoku': '#6A994E'     # Green
}

# Model signature colors (will be assigned dynamically)
MODEL_COLORS = {}

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

def create_overall_model_figure(overall_df: pd.DataFrame):
    """Create a clean bar chart showing overall model performance."""
    
    # Professional typography settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 11
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # Data is already sorted by success rate
    models = overall_df["model"].values
    success_rates = overall_df["success_rate"].values
    
    # Create gradient colors based on success rate
    colors = [SUCCESS_CMAP(rate/100) for rate in success_rates]
    
    # Create VERTICAL bar chart
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, success_rates, color=colors,
                  edgecolor='#333333', linewidth=1.2, width=0.7,
                  alpha=0.85)
    
    # Set labels and title
    ax.set_xlabel('Model', fontsize=13, color='#333333', fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=13, color='#333333', fontweight='bold')
    ax.set_title('Overall Model Performance Ranking', 
                fontsize=16, fontweight='bold', pad=20, color='#333333')
    
    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right')
    
    # Set y-axis (0-100%)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f'{int(y)}%' for y in np.arange(0, 101, 20)])
    
    # Add subtle grid
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Add value labels
    for i, (bar, row) in enumerate(zip(bars, overall_df.itertuples())):
        rate = row.success_rate
        # Success rate on top of bar
        ax.text(i, rate + 1, f'{rate:.1f}%', 
               ha='center', va='bottom', fontsize=10, 
               fontweight='bold', color='#333333')
        
        # Removed rank labels for cleaner look
    
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    output_path = figures_dir / "overall_model_ranking.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"📊 Overall model ranking figure saved to: {output_path}")

def create_overall_domain_figure(domain_df: pd.DataFrame):
    """Create a clean bar chart showing overall domain performance."""
    
    # Professional typography settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('white')
    
    # Calculate domain statistics
    domain_stats = domain_df.groupby("domain").agg({
        "success_rate": "mean",
        "total_tasks": "sum",
        "correct_tasks": "sum"
    }).sort_values("success_rate", ascending=False)
    
    domains = domain_stats.index
    success_rates = domain_stats["success_rate"].values
    
    # Use domain-specific colors
    colors = [DOMAIN_COLORS.get(d, MAIN_PALETTE[i % len(MAIN_PALETTE)]) 
             for i, d in enumerate(domains)]
    
    # Create vertical bar chart
    x_pos = np.arange(len(domains))
    bars = ax.bar(x_pos, success_rates, color=colors,
                 edgecolor='#333333', linewidth=1.2, width=0.7,
                 alpha=0.85)
    
    # Set labels and title
    ax.set_xlabel('Domain', fontsize=13, color='#333333', fontweight='bold')
    ax.set_ylabel('Average Success Rate', fontsize=13, color='#333333', fontweight='bold')
    ax.set_title('Domain Difficulty Analysis', 
                fontsize=16, fontweight='bold', pad=20, color='#333333')
    
    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d.capitalize() for d in domains], rotation=0, ha='center')
    
    # Set y-axis (0-100%)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f'{int(y)}%' for y in np.arange(0, 101, 20)])
    
    # Add subtle grid
    ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Add value labels and difficulty indicators
    for i, (bar, rate, domain) in enumerate(zip(bars, success_rates, domains)):
        # Success rate on top
        ax.text(i, rate + 1, f'{rate:.1f}%', 
               ha='center', va='bottom', fontsize=11,
               fontweight='bold', color='#333333')
        
        # Removed difficulty labels below x-axis for cleaner look
    
    # Removed reference lines and legend for cleaner look
    # Uncomment below if you want reference lines back:
    # ax.axhline(y=70, color='#2E7D32', linestyle=':', alpha=0.3, linewidth=1)
    # ax.axhline(y=40, color='#FFA726', linestyle=':', alpha=0.3, linewidth=1)
    
    # Removed summary statistics for cleaner look
    
    plt.tight_layout()
    
    # Save figure
    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    output_path = figures_dir / "overall_domain_difficulty.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"📊 Overall domain difficulty figure saved to: {output_path}")

def create_individual_model_plots(domain_df: pd.DataFrame, overall_df: pd.DataFrame):
    """Create clean individual bar plots for each model's performance across domains."""
    
    # Create figures/models directory
    models_dir = Path(__file__).parent / "figures" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Professional typography settings
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.titlesize'] = 15
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    
    # Assign colors to models if not already assigned
    unique_models = overall_df["model"].unique()
    for i, model in enumerate(unique_models):
        if model not in MODEL_COLORS:
            MODEL_COLORS[model] = MAIN_PALETTE[i % len(MAIN_PALETTE)]
    
    # Create individual plot for each model - SIMPLE BAR CHART
    for model in unique_models:
        model_data = domain_df[domain_df["model"] == model]
        
        if model_data.empty:
            continue
        
        # Create figure with white background - SINGLE PLOT ONLY
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        ax.set_facecolor('white')
        
        # Sort domains by success rate for better visual
        model_data_sorted = model_data.sort_values("success_rate", ascending=False)
        
        # Prepare data
        domains = model_data_sorted["domain"].values
        success_rates = model_data_sorted["success_rate"].values
        
        # Create gradient colors based on success rate
        colors = [SUCCESS_CMAP(rate/100) for rate in success_rates]
        
        # Create bar chart (vertical bars for domains)
        x_pos = np.arange(len(domains))
        bars = ax.bar(x_pos, success_rates, color=colors, 
                      edgecolor='#333333', linewidth=1.2, width=0.7,
                      alpha=0.85)
        
        # Set labels and title
        ax.set_xlabel('Domain', fontsize=13, color='#333333')
        ax.set_ylabel('Success Rate', fontsize=13, color='#333333')
        ax.set_title(f'{model} Performance by Domain', 
                    fontsize=15, fontweight='bold', pad=20, color='#333333')
        
        # Set x-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels([d.capitalize() for d in domains], rotation=0, ha='center')
        
        # Set y-axis (0-100%)
        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_yticklabels([f'{int(y)}%' for y in np.arange(0, 101, 20)])
        
        # Add subtle grid
        ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Add value labels on top of bars
        for i, (bar, rate) in enumerate(zip(bars, success_rates)):
            # Display success rate percentage
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{rate:.1f}%', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', color='#333333')
        
        # Add overall stats as subtitle (without rank)
        overall_stats = overall_df[overall_df["model"] == model].iloc[0]
        subtitle_text = (f"Overall Success: {overall_stats['success_rate']:.1f}% | "
                        f"Average Score: {overall_stats['average_score']:.2f}")
        ax.text(0.5, -0.12, subtitle_text, transform=ax.transAxes,
               ha='center', va='top', fontsize=10, color='#666666')
        
        plt.tight_layout()
        
        # Save with sanitized filename
        safe_model_name = model.replace("/", "_").replace(" ", "_")
        output_path = models_dir / f"{safe_model_name}_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"  📊 Created plot for {model}: {output_path}")

def print_detailed_results(domain_df: pd.DataFrame, overall_df: pd.DataFrame):
    """Print comprehensive statistics in table-friendly format for paper."""
    
    print("\n" + "="*100)
    print("🎯 VMEVALKIT COMPREHENSIVE EVALUATION STATISTICS")
    print("="*100)
    
    # Table 1: Overall Model Performance Ranking
    print("\n📊 TABLE 1: OVERALL MODEL PERFORMANCE RANKING")
    print("-" * 100)
    print(f"{'Rank':<6} {'Model':<30} {'Success Rate':<15} {'Correct/Total':<15} {'Avg Score':<12} {'Std Dev':<10}")
    print("-" * 100)
    
    for _, row in overall_df.iterrows():
        # Calculate std dev for this model
        model_scores = []
        for _, domain_row in domain_df[domain_df["model"] == row["model"]].iterrows():
            model_scores.extend(domain_row["scores"])
        std_dev = np.std(model_scores) if model_scores else 0
        
        print(f"{row['rank']:<6} {row['model']:<30} {row['success_rate']:>6.2f}%{'':<8} "
              f"{row['correct_tasks']:>3d}/{row['total_tasks']:<3d}{'':<8} "
              f"{row['average_score']:>6.3f}{'':<6} {std_dev:>6.3f}")
    
    # Table 2: Domain-wise Performance Matrix
    print("\n\n📊 TABLE 2: MODEL PERFORMANCE BY DOMAIN (Success Rate %)")
    print("-" * 100)
    
    pivot_table = domain_df.pivot(index="model", columns="domain", values="success_rate")
    
    # Print header
    header = f"{'Model':<30}"
    for domain in sorted(pivot_table.columns):
        header += f" {domain.capitalize():<12}"
    header += f" {'Average':<12}"
    print(header)
    print("-" * 100)
    
    # Print data
    for model in overall_df["model"]:
        row_str = f"{model:<30}"
        domain_scores = []
        for domain in sorted(pivot_table.columns):
            if model in pivot_table.index and domain in pivot_table.columns:
                value = pivot_table.loc[model, domain]
                if pd.notna(value):
                    row_str += f" {value:>6.1f}%{'':<5}"
                    domain_scores.append(value)
                else:
                    row_str += f" {'N/A':<12}"
            else:
                row_str += f" {'N/A':<12}"
        
        # Add average
        if domain_scores:
            avg_score = np.mean(domain_scores)
            row_str += f" {avg_score:>6.1f}%"
        else:
            row_str += f" {'N/A':<12}"
        print(row_str)
    
    # Table 3: Domain Statistics
    print("\n\n📊 TABLE 3: DOMAIN-LEVEL STATISTICS")
    print("-" * 100)
    print(f"{'Domain':<15} {'Avg Success':<15} {'Std Dev':<12} {'Min Score':<12} {'Max Score':<12} {'Total Tasks':<12} {'Difficulty':<12}")
    print("-" * 100)
    
    domain_stats = domain_df.groupby("domain").agg({
        "success_rate": ["mean", "std", "min", "max"],
        "total_tasks": "sum"
    }).round(2)
    
    for domain in domain_stats.index:
        avg_rate = domain_stats.loc[domain, ("success_rate", "mean")]
        std_rate = domain_stats.loc[domain, ("success_rate", "std")]
        min_rate = domain_stats.loc[domain, ("success_rate", "min")]
        max_rate = domain_stats.loc[domain, ("success_rate", "max")]
        total_tasks = int(domain_stats.loc[domain, ("total_tasks", "sum")])
        
        difficulty = "Easy" if avg_rate > 70 else "Medium" if avg_rate > 40 else "Hard"
        
        print(f"{domain.capitalize():<15} {avg_rate:>6.2f}%{'':<8} {std_rate:>8.2f}{'':<4} "
              f"{min_rate:>6.2f}%{'':<5} {max_rate:>6.2f}%{'':<5} {total_tasks:>8}{'':<4} {difficulty:<12}")
    
    # Table 4: Score Distribution Analysis
    print("\n\n📊 TABLE 4: SCORE DISTRIBUTION ANALYSIS")
    print("-" * 100)
    
    all_scores = []
    score_by_model = defaultdict(list)
    for _, row in domain_df.iterrows():
        all_scores.extend(row["scores"])
        score_by_model[row["model"]].extend(row["scores"])
    
    score_counts = np.bincount(all_scores, minlength=6)[1:]  # Scores 1-5
    total_scores = len(all_scores)
    
    print(f"{'Score':<10} {'Count':<10} {'Percentage':<15} {'Cumulative %':<15} {'Classification':<20}")
    print("-" * 100)
    
    cumulative = 0
    for i, count in enumerate(score_counts, 1):
        percentage = (count / total_scores) * 100 if total_scores > 0 else 0
        cumulative += percentage
        classification = "Success" if i >= 4 else "Failure"
        print(f"Score {i:<4} {count:<10} {percentage:>6.2f}%{'':<8} {cumulative:>6.2f}%{'':<8} {classification:<20}")
    
    # Table 5: Model-Domain Detailed Breakdown
    print("\n\n📊 TABLE 5: DETAILED MODEL-DOMAIN BREAKDOWN")
    print("-" * 100)
    print(f"{'Model':<30} {'Domain':<15} {'Tasks':<8} {'Correct':<10} {'Success%':<12} {'Avg Score':<12}")
    print("-" * 100)
    
    for model in sorted(domain_df["model"].unique()):
        model_data = domain_df[domain_df["model"] == model].sort_values("domain")
        first_row = True
        for _, row in model_data.iterrows():
            model_name = model if first_row else ""
            first_row = False
            print(f"{model_name:<30} {row['domain'].capitalize():<15} {row['total_tasks']:<8} "
                  f"{row['correct_tasks']:<10} {row['success_rate']:>6.2f}%{'':<5} {row['average_score']:>6.3f}")
        
        # Add model summary
        model_overall = overall_df[overall_df["model"] == model].iloc[0]
        print(f"{'  TOTAL':<30} {'All Domains':<15} {model_overall['total_tasks']:<8} "
              f"{model_overall['correct_tasks']:<10} {model_overall['success_rate']:>6.2f}%{'':<5} "
              f"{model_overall['average_score']:>6.3f}")
        print("-" * 100)
    
    # Table 6: Statistical Summary
    print("\n\n📊 TABLE 6: STATISTICAL SUMMARY")
    print("-" * 100)
    
    print(f"Total Evaluations: {total_scores:,}")
    print(f"Total Correct (Scores 4-5): {sum(score_counts[3:]):,}")
    print(f"Total Failed (Scores 1-3): {sum(score_counts[:3]):,}")
    print(f"Overall Success Rate: {(sum(score_counts[3:]) / total_scores * 100):.2f}%")
    print(f"Number of Models Evaluated: {len(overall_df)}")
    print(f"Number of Domains: {len(domain_df['domain'].unique())}")
    print(f"Average Tasks per Model: {total_scores / len(overall_df):.1f}")
    print(f"Average Tasks per Domain: {total_scores / len(domain_df['domain'].unique()):.1f}")
    
    # Best performers
    print("\n🏆 TOP PERFORMERS:")
    print("-" * 50)
    
    # Best overall model
    best_model = overall_df.iloc[0]
    print(f"Best Overall Model: {best_model['model']} ({best_model['success_rate']:.2f}%)")
    
    # Best model per domain
    for domain in sorted(domain_df["domain"].unique()):
        domain_best = domain_df[domain_df["domain"] == domain].nlargest(1, "success_rate").iloc[0]
        print(f"Best in {domain.capitalize()}: {domain_best['model']} ({domain_best['success_rate']:.2f}%)")
    
    # Difficulty analysis
    print("\n📊 DOMAIN DIFFICULTY RANKING:")
    print("-" * 50)
    
    domain_difficulty = domain_df.groupby("domain")["success_rate"].mean().sort_values(ascending=False)
    
    for rank, (domain, avg_rate) in enumerate(domain_difficulty.items(), 1):
        difficulty_level = "Easy" if avg_rate > 70 else "Medium" if avg_rate > 40 else "Hard"
        print(f"{rank}. {domain.capitalize():<15} - Average Success: {avg_rate:>6.2f}% ({difficulty_level})")
    
    # Table 7: Performance Variance Analysis
    print("\n\n📊 TABLE 7: PERFORMANCE VARIANCE ANALYSIS")
    print("-" * 100)
    print(f"{'Model':<30} {'Min Domain %':<15} {'Max Domain %':<15} {'Variance':<12} {'Consistency':<15}")
    print("-" * 100)
    
    for model in overall_df["model"]:
        model_domain_scores = domain_df[domain_df["model"] == model]["success_rate"].values
        if len(model_domain_scores) > 0:
            min_score = model_domain_scores.min()
            max_score = model_domain_scores.max()
            variance = max_score - min_score
            consistency = "High" if variance < 20 else "Medium" if variance < 40 else "Low"
            
            print(f"{model:<30} {min_score:>6.2f}%{'':<8} {max_score:>6.2f}%{'':<8} "
                  f"{variance:>6.2f}{'':<6} {consistency:<15}")
    
    print("\n" + "="*100)
    print("END OF COMPREHENSIVE STATISTICS")
    print("="*100)

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
    print(f"\n📊 Creating clean bar chart visualizations...")
    
    # 1. Overall model ranking figure
    create_overall_model_figure(overall_df)
    
    # 2. Overall domain difficulty figure
    create_overall_domain_figure(domain_df)
    
    # 3. Individual model performance plots
    print(f"\n📊 Creating individual model performance plots...")
    create_individual_model_plots(domain_df, overall_df)

if __name__ == "__main__":
    main()