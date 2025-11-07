#!/usr/bin/env python3
"""
VMEvalKit Question Creation - Flexible Task Generation

This script provides flexible question/task generation with customizable domain
selection and quantities. Generate questions for specific reasoning domains
or create comprehensive datasets across all domains.

Key Features:
- Generate questions for specific domains (chess, maze, raven, rotation, sudoku)
- Control number of questions per domain with flexible quantities
- Read and analyze existing question datasets
- Support for different random seeds for reproducible generation
- Automatic organization in per-question folder structure

Output Structure:
- Each question gets its own folder: data/questions/{domain}_task/{question_id}/
- Contains: first_frame.png, final_frame.png, prompt.txt, question_metadata.json
- Master dataset JSON with complete metadata and organization

Author: VMEvalKit Team
"""

import sys
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vmevalkit.runner.dataset import (
    create_vmeval_dataset_direct, 
    read_dataset_from_folders, 
    print_dataset_summary,
    DOMAIN_REGISTRY
)

def main():
    """Flexible VMEvalKit question creation with customizable options."""

    parser = argparse.ArgumentParser(
        description="VMEvalKit Question Creation - Flexible task generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 50 questions per domain (default)
  python create_questions.py
  
  # Generate specific number per domain
  python create_questions.py --pairs-per-domain 100
  
  # Generate for specific domains only
  python create_questions.py --task chess maze --pairs-per-domain 25
  
  # Generate small test set
  python create_questions.py --task chess --pairs-per-domain 5
  
  # Just read and analyze existing questions
  python create_questions.py --read-only
  
  # Use different random seed
  python create_questions.py --random-seed 123 --pairs-per-domain 10
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
        "--read-only", 
        action="store_true", 
        help="Only read and analyze existing dataset from folders, don't generate new questions"
    )
    
    parser.add_argument(
        "--task", 
        nargs='+', 
        choices=list(DOMAIN_REGISTRY.keys()), 
        help=f"Specific task domain(s) to generate. Available: {', '.join(DOMAIN_REGISTRY.keys())}. If not specified, generates all domains."
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
    
    # Handle --list-domains
    if args.list_domains:
        print("🧠 Available Task Domains:")
        print("=" * 60)
        for domain_key, domain_info in DOMAIN_REGISTRY.items():
            print(f"{domain_info.get('emoji', '🔹')} {domain_key:10} - {domain_info['description']}")
        print(f"\nTotal: {len(DOMAIN_REGISTRY)} reasoning domains available")
        print("\nUse --task to select specific domains, or run without --task for all domains.")
        sys.exit(0)

    if args.read_only:
        # Just read existing questions from folders
        print("=" * 70)
        print("📂 Reading existing questions from folder structure...")
        print(f"   Reading from: {args.output_dir}")
        dataset = read_dataset_from_folders(Path(args.output_dir))
        print_dataset_summary(dataset)
        print("=" * 70)
        return

    # Show generation plan
    selected_domains = args.task if args.task else list(DOMAIN_REGISTRY.keys())
    total_questions = len(selected_domains) * args.pairs_per_domain
    
    print("=" * 70)
    print("🚀 VMEvalKit Question Generation Plan")
    print("=" * 70)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Selected domains: {', '.join(selected_domains)} ({len(selected_domains)} domains)")
    print(f"📊 Questions per domain: {args.pairs_per_domain}")
    print(f"🔢 Total questions to generate: {total_questions}")
    print(f"🎲 Random seed: {args.random_seed}")
    print()

    # Generate questions directly to folders
    dataset, questions_dir = create_vmeval_dataset_direct(
        pairs_per_domain=args.pairs_per_domain, 
        random_seed=args.random_seed,
        selected_tasks=args.task
    )
    
    # Print comprehensive summary
    print_dataset_summary(dataset)
    
    print(f"💾 Master dataset JSON saved: {questions_dir}/vmeval_dataset.json")
    print(f"📁 Questions generated in: {questions_dir}")
    print(f"🔗 Per-question folders: {questions_dir}/<domain>_task/<question_id>/")
    print()
    print("🎉 VMEvalKit Questions ready for video generation!")
    print("🚀 Next steps:")
    print(f"   • Generate videos: python examples/generate_videos.py")
    print(f"   • Score videos: python examples/score_videos.py human")
    print(f"   • Run inference: python -m vmevalkit.runner.inference")
    print("=" * 70)

if __name__ == "__main__":
    main()
