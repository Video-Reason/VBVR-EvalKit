"""
VMEvalKit Data Folder Verification Utility

Verifies that the data folder follows the VMEvalKit data format structure.

Example CLI usage:
    python -m vmevalkit.utils.verify
    python -m vmevalkit.utils.verify --data-dir ./data
    python -m vmevalkit.utils.verify --data-dir ./data --json

Example Python usage:
    from pathlib import Path
    from vmevalkit.utils.verify import verify_data_folder, verify_and_report

    # Basic boolean check
    is_valid, errors, warnings = verify_data_folder(Path("data"))
    if not is_valid:
        print("Data folder invalid:", errors)

    # Print human-readable report
    verify_and_report(Path("data"))
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def verify_data_folder(data_dir: Path = None) -> Tuple[bool, List[str], List[str]]:
    """
    Verify if the data folder follows VMEvalKit data format.
    
    Args:
        data_dir: Path to data directory (default: ./data)
        
    Returns:
        Tuple of (is_valid, errors, warnings)
        - is_valid: True if structure is valid
        - errors: List of critical issues
        - warnings: List of non-critical issues
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data"
    
    data_dir = Path(data_dir)
    errors = []
    warnings = []
    
    # Check if data directory exists
    if not data_dir.exists():
        errors.append(f"Data directory does not exist: {data_dir}")
        return False, errors, warnings
    
    # 1. Verify questions/ directory structure
    questions_dir = data_dir / "questions"
    if not questions_dir.exists():
        errors.append("Missing questions/ directory")
    else:
        # Check for master dataset manifest
        manifest = questions_dir / "vmeval_dataset.json"
        if not manifest.exists():
            errors.append("Missing vmeval_dataset.json in questions/")
        else:
            # Validate manifest structure
            with open(manifest) as f:
                dataset = json.load(f)
                
            if "pairs" not in dataset:
                errors.append("vmeval_dataset.json missing 'pairs' field")
            else:
                # Verify each task pair
                for pair in dataset.get("pairs", []):
                    pair_id = pair.get("id", "unknown")
                    
                    # Check required fields
                    required_fields = ["id", "domain", "first_image_path", "final_image_path", "prompt"]
                    for field in required_fields:
                        if field not in pair:
                            errors.append(f"Task {pair_id}: missing required field '{field}'")
                    
                    # Check if image files exist
                    if "first_image_path" in pair:
                        first_img = questions_dir / pair["first_image_path"]
                        if not first_img.exists():
                            errors.append(f"Task {pair_id}: missing first image: {pair['first_image_path']}")
                    
                    if "final_image_path" in pair:
                        final_img = questions_dir / pair["final_image_path"]
                        if not final_img.exists():
                            errors.append(f"Task {pair_id}: missing final image: {pair['final_image_path']}")
        
        # Check for task directories
        expected_tasks = ["chess_task", "maze_task", "raven_task", "rotation_task", "sudoku_task"]
        for task in expected_tasks:
            task_dir = questions_dir / task
            if not task_dir.exists():
                warnings.append(f"Task directory not found: {task}")
    
    # 2. Verify outputs/ directory structure
    outputs_dir = data_dir / "outputs"
    if not outputs_dir.exists():
        warnings.append("Missing outputs/ directory (will be created during inference)")
    else:
        # Check for experiments
        experiments = [d for d in outputs_dir.iterdir() if d.is_dir()]
        if not experiments:
            warnings.append("No experiments found in outputs/")
        else:
            for exp_dir in experiments:
                # Check experiment structure: outputs/{experiment}/{model}/{domain_task}/{task_id}/{run_id}/
                models = [d for d in exp_dir.iterdir() if d.is_dir()]
                for model_dir in models:
                    domains = [d for d in model_dir.iterdir() if d.is_dir()]
                    for domain_dir in domains:
                        tasks = [d for d in domain_dir.iterdir() if d.is_dir()]
                        for task_dir in tasks:
                            runs = [d for d in task_dir.iterdir() if d.is_dir()]
                            for run_dir in runs:
                                # Check for expected files/folders
                                video_dir = run_dir / "video"
                                question_dir = run_dir / "question"
                                metadata_file = run_dir / "metadata.json"
                                
                                if not video_dir.exists():
                                    warnings.append(f"Missing video/ in {run_dir.relative_to(data_dir)}")
                                if not question_dir.exists():
                                    warnings.append(f"Missing question/ in {run_dir.relative_to(data_dir)}")
                                if not metadata_file.exists():
                                    warnings.append(f"Missing metadata.json in {run_dir.relative_to(data_dir)}")
    
    # 3. Verify evaluations/ directory structure (optional)
    evaluations_dir = data_dir / "evaluations"
    if not evaluations_dir.exists():
        warnings.append("Missing evaluations/ directory (created during evaluation)")
    
    # 4. Verify data_logging/ directory (optional)
    logging_dir = data_dir / "data_logging"
    if not logging_dir.exists():
        warnings.append("Missing data_logging/ directory (created during version tracking)")
    else:
        version_log = logging_dir / "version_log.json"
        if not version_log.exists():
            warnings.append("Missing version_log.json in data_logging/")
    
    # Determine if valid (no critical errors)
    is_valid = len(errors) == 0
    
    return is_valid, errors, warnings


def print_verification_report(is_valid: bool, errors: List[str], warnings: List[str]) -> None:
    """Print a formatted verification report."""
    print("=" * 70)
    print("VMEvalKit Data Folder Verification Report")
    print("=" * 70)
    print()
    
    if is_valid:
        print("✅ Data folder structure is VALID")
    else:
        print("❌ Data folder structure has ERRORS")
    
    print()
    
    if errors:
        print(f"🔴 Errors ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
        print()
    
    if warnings:
        print(f"⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    
    if is_valid and not warnings:
        print("🎉 All checks passed! Data folder is properly structured.")
    
    print("=" * 70)


def verify_and_report(data_dir: Path = None) -> bool:
    """
    Verify data folder and print report.
    
    Args:
        data_dir: Path to data directory (default: ./data)
        
    Returns:
        True if valid, False otherwise
    """
    is_valid, errors, warnings = verify_data_folder(data_dir)
    print_verification_report(is_valid, errors, warnings)
    return is_valid


def main():
    """CLI entry point for verification."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify VMEvalKit data folder structure")
    parser.add_argument("--data-dir", type=Path, help="Path to data directory (default: ./data)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    is_valid, errors, warnings = verify_data_folder(args.data_dir)
    
    if args.json:
        import json
        result = {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings
        }
        print(json.dumps(result, indent=2))
    else:
        print_verification_report(is_valid, errors, warnings)
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
