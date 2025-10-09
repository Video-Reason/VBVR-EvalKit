#!/usr/bin/env python3
"""
Test Luma API with maze reasoning tasks.

This script demonstrates how to use the Luma Dream Machine API with VMEvalKit
to evaluate video reasoning capabilities on maze-solving tasks.

Note: Requires proper S3 configuration for image hosting. The default bucket
'vmevalkit' is in us-east-2. Ensure AWS_DEFAULT_REGION matches or update
the S3_BUCKET in your .env file.
"""

import os
import sys
from pathlib import Path
import time
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from vmevalkit.api_clients.luma_client import LumaDreamMachine
from vmevalkit.utils.s3_uploader import S3ImageUploader

def test_maze_reasoning():
    """Test Luma's ability to generate maze-solving videos."""
    
    # Initialize Luma client
    client = LumaDreamMachine(
        enhance_prompt=False,  # Don't modify our reasoning prompts
        model="ray-2",
        verbose=True
    )
    
    # Initialize S3 uploader
    uploader = S3ImageUploader()
    
    # Test cases: maze type, index, and prompts to try
    test_cases = [
        {
            "maze_type": "irregular",
            "index": "0000",
            "prompts": [
                "Show the solution path through this maze from start to finish.",
                "Animate solving this maze step by step, showing the correct path.",
                "Trace the path from the green start to the red exit in this maze."
            ]
        },
        {
            "maze_type": "knowwhat", 
            "index": "0001",
            "prompts": [
                "Show how to solve this maze by finding the path from start to goal.",
                "Draw the solution path through this maze.",
                "Navigate through the maze from the entrance to the exit."
            ]
        }
    ]
    
    results = []
    
    try:
        for test in test_cases:
            maze_type = test["maze_type"]
            index = test["index"]
            
            # Use the 'first' image (start position only)
            image_filename = f"{maze_type}_{index}_first.png"
            image_path = f"data/generated_mazes/{image_filename}"
            
            # Upload image to S3
            print(f"\nUploading {image_filename} to S3...")
            image_url = uploader.upload(image_path)
            
            if not image_url:
                print(f"Failed to upload {image_filename}, skipping...")
                continue
            
            print(f"\n{'='*60}")
            print(f"Testing {maze_type} maze {index}")
            print(f"Image: {image_filename}")
            print(f"URL: {image_url}")
            print(f"{'='*60}")
            
            # Try different prompts
            for i, prompt in enumerate(test["prompts"][:1]):  # Just test first prompt for now
                print(f"\nPrompt {i+1}: {prompt}")
                
                try:
                    start_time = time.time()
                    
                    # Generate video
                    output_path = client.generate(
                        image=image_url,
                        text_prompt=prompt,
                        duration=5.0,
                        resolution=(512, 512)  # Use square for maze
                    )
                    
                    generation_time = time.time() - start_time
                    
                    # Record result
                    result = {
                        "maze_type": maze_type,
                        "maze_index": index,
                        "prompt": prompt,
                        "output_path": str(output_path),
                        "generation_time": generation_time,
                        "success": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(result)
                    
                    print(f"✓ Success! Video saved to: {output_path}")
                    print(f"  Generation time: {generation_time:.1f}s")
                    
                except Exception as e:
                    result = {
                        "maze_type": maze_type,
                        "maze_index": index,
                        "prompt": prompt,
                        "error": str(e),
                        "success": False,
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(result)
                    print(f"✗ Error: {e}")
                
                # Wait a bit between requests
                if i < len(test["prompts"]) - 1:
                    print("\nWaiting 10s before next prompt...")
                    time.sleep(10)
            
            # Wait between different mazes
            print("\nWaiting 15s before next maze...")
            time.sleep(15)
    
    finally:
        # Clean up S3 files
        print("\nCleaning up S3 files...")
        uploader.cleanup()
        
        # Save results
        results_file = "outputs/luma_maze_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {results_file}")
        print(f"Total tests: {len(results)}")
        print(f"Successful: {sum(1 for r in results if r['success'])}")
        print(f"Failed: {sum(1 for r in results if not r['success'])}")
        print(f"{'='*60}")

if __name__ == "__main__":
    test_maze_reasoning()
