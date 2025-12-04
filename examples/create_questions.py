#!/usr/bin/env python3
"""
VMEvalKit Question Creation - Flexible Task Generation

This script provides flexible question/task generation with customizable domain
selection and quantities. Generate questions for specific reasoning domains
or create comprehensive datasets across all domains.

Key Features:
- Generate questions for specific domains (chess, maze, raven, rotation, sudoku)

Output Structure:
- Each question gets its own folder: data/questions/{domain}_task/{question_id}/
- Contains: first_frame.png, final_frame.png, prompt.txt, question_metadata.json
- Master dataset JSON with complete metadata and organization

Author: VMEvalKit Team
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vmevalkit.runner.dataset import (
    create_vmeval_dataset_direct, 
    read_dataset_from_folders, 
    print_dataset_summary,
    download_hf_domain_to_folders
)
from vmevalkit.runner.TASK_CATALOG import TASK_REGISTRY

def main():
    """Flexible VMEvalKit question creation with customizable options."""

    parser = argparse.ArgumentParser(
        description="VMEvalKit Question Creation - Flexible task generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Download entire VideoThinkBench dataset (all 4 subsets, ~4k tasks)
            python create_questions.py --task videothinkbench
            
            # Generate 50 questions per domain (default) for original tasks
            python create_questions.py --task chess maze raven --pairs-per-domain 50
            
            # Mix VideoThinkBench and generated tasks
            python create_questions.py --task videothinkbench chess sudoku --pairs-per-domain 25
            
            # Download specific VideoThinkBench subsets
            python create_questions.py --task arc_agi_2 visual_puzzles text_centric_tasks
            
            # Generate for all original domains
            python create_questions.py --task chess maze raven rotation sudoku object_subtraction --pairs-per-domain 50
            
            # Generate with non-deterministic random (no fixed seed)
            python create_questions.py --task chess maze --pairs-per-domain 50 --no-seed
            
            # Generate with custom random seed
            python create_questions.py --task chess maze --pairs-per-domain 50 --random-seed 123
            
            # List all available domains
            python create_questions.py --list-domains
            
            # Just read and analyze existing questions
            python create_questions.py --read-only
        """
    )
    
    parser.add_argument(
        "--pairs-per-domain", 
        type=int, 
        default=50, 
        help="Number of task pairs to generate per domain (default: 50)"
    )
    
    parser.add_argument(
        "--random-seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducible generation (default: 42)"
    )
    
    parser.add_argument(
        "--no-seed", 
        action="store_true", 
        help="Use non-deterministic random generation (unset seed, each run produces different results)"
    )
    
    parser.add_argument(
        "--read-only", 
        action="store_true", 
        help="Only read and analyze existing dataset from folders, don't generate new questions"
    )
    
    parser.add_argument(
        "--task", 
        nargs='+', 
        choices=list(TASK_REGISTRY.keys()), 
        help=f"Specific task domain(s) to generate. Available: {', '.join(TASK_REGISTRY.keys())}. If not specified, generates all domains."
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/questions",
        help="Output directory for questions (default: data/questions)"
    )
    
    parser.add_argument(
        "--list-domains",
        action="store_true", 
        help="List all available task domains and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_domains:
        print("🧠 Available Task Domains:")
        print("=" * 60)
        for domain_key, domain_info in TASK_REGISTRY.items():
            hf_info = " (HuggingFace)" if domain_info.get('hf', False) else ""
            print(f"{domain_info.get('emoji', '🔹')} {domain_key:15} - {domain_info['description']}{hf_info}")
        print(f"\nTotal: {len(TASK_REGISTRY)} reasoning domains available")
        print("\nUse --task to select specific domains, or run without --task for all domains.")
        return

    if args.read_only:
        print("=" * 70)
        print("📂 Reading existing questions from folder structure...")
        print(f"   Reading from: {args.output_dir}")
        dataset = read_dataset_from_folders(Path(args.output_dir))
        print_dataset_summary(dataset)
        print("=" * 70)
        return

    output_path = Path(args.output_dir)
    task = args.task
    
    # Determine random seed: None if --no-seed is set, otherwise use --random-seed value
    random_seed = None if args.no_seed else args.random_seed
    
    # Expand meta-tasks (like 'videothinkbench') into their constituent subsets
    expanded_domains = []
    for domain in task:
        domain_config = TASK_REGISTRY.get(domain, {})
        if domain_config.get('hf_meta', False):
            # This is a meta-task that contains multiple subsets
            subsets = domain_config.get('hf_subsets', [])
            expanded_domains.extend(subsets)
            print(f"📦 Expanding '{domain}' meta-task into subsets: {', '.join(subsets)}")
        else:
            expanded_domains.append(domain)
    
    # Remove duplicates while preserving order
    seen = set()
    task = [d for d in expanded_domains if not (d in seen or seen.add(d))]
    
    hf_domains = [d for d in task if TASK_REGISTRY.get(d, {}).get('hf', False) is True and not TASK_REGISTRY.get(d, {}).get('hf_meta', False)]
    regular_domains = [d for d in task if TASK_REGISTRY.get(d, {}).get('hf', False) is not True]
    
    if hf_domains:
        for domain in hf_domains:
            download_hf_domain_to_folders(domain, output_path)
    
    if regular_domains:
        total_questions = len(regular_domains) * args.pairs_per_domain
        
        print("=" * 70)
        print("🚀 VMEvalKit Question Generation Plan")
        print("=" * 70)
        print(f"📁 Output directory: {args.output_dir}")
        print(f"🎯 Selected domains: {', '.join(regular_domains)} ({len(regular_domains)} domains)")
        print(f"📊 Questions per domain: {args.pairs_per_domain}")
        print(f"🔢 Total questions to generate: {total_questions}")
        if random_seed is not None:
            print(f"🎲 Random seed: {random_seed}")
        else:
            print(f"🎲 Random seed: None (non-deterministic)")
        print()

        dataset, questions_dir = create_vmeval_dataset_direct(
            pairs_per_domain=args.pairs_per_domain, 
            random_seed=random_seed,
            selected_tasks=regular_domains
        )
        
        print_dataset_summary(dataset)
    
    print("=" * 70)
    print("📂 Reading all questions from folder structure...")
    print("=" * 70)
    final_dataset = read_dataset_from_folders(output_path)
    print_dataset_summary(final_dataset)
    
    print(f"📁 Questions saved in: {output_path}")
    print(f"🔗 Per-question folders: {output_path}/<domain>_task/<question_id>/")
    print()
    print("🎉 VMEvalKit Questions ready for video generation!")
    print("🚀 Next steps:")
    print(f"   • Generate videos: python examples/generate_videos.py")
    print(f"   • Score videos: python examples/score_videos.py human")
    print("=" * 70)

if __name__ == "__main__":
    main()
