#!/usr/bin/env python3
"""
Test different Luma model endpoints: ray-flash-2, ray-2, ray-1-6
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from vmevalkit.api_clients.luma_client import LumaDreamMachine
from vmevalkit.utils.s3_uploader import S3ImageUploader

def test_luma_model(model_name: str):
    """Test a specific Luma model endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize client with specific model
        client = LumaDreamMachine(
            model=model_name,
            enhance_prompt=False,
            verbose=True
        )
        
        # Use a simple test image - pick the first available maze
        test_image = "data/generated_mazes/irregular_0000_first.png"
        if not Path(test_image).exists():
            print(f"❌ Test image not found: {test_image}")
            return False
            
        # Simple test prompt
        test_prompt = "Show a path being drawn through this maze from start to finish"
        
        # Upload image to S3
        uploader = S3ImageUploader()
        print(f"Uploading test image to S3...")
        image_url = uploader.upload(test_image)
        
        if not image_url:
            print(f"❌ Failed to upload image to S3")
            return False
            
        print(f"Image URL: {image_url}")
        
        # Try to generate video
        print(f"\nGenerating video with {model_name}...")
        output_path = client.generate(
            image=image_url,
            text_prompt=test_prompt,
            duration=3.0,  # Short duration for testing
            resolution=(512, 512)
        )
        
        # Clean up S3
        uploader.cleanup()
        
        print(f"✅ SUCCESS! Model {model_name} works!")
        print(f"   Generated video: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED! Model {model_name} error: {e}")
        # Clean up S3 on error
        try:
            uploader.cleanup()
        except:
            pass
        return False


def main():
    """Test all three model endpoints."""
    models_to_test = ["ray-flash-2", "ray-2", "ray-1-6"]
    results = {}
    
    print("Testing Luma Model Endpoints")
    print("============================")
    
    for model in models_to_test:
        results[model] = test_luma_model(model)
        
        # Wait between tests to avoid rate limits
        if model != models_to_test[-1]:
            import time
            print("\nWaiting 15 seconds before next test...")
            time.sleep(15)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    working_models = [m for m, success in results.items() if success]
    failed_models = [m for m, success in results.items() if not success]
    
    print(f"\n✅ Working models ({len(working_models)}):")
    for model in working_models:
        print(f"   - {model}")
        
    if failed_models:
        print(f"\n❌ Failed models ({len(failed_models)}):")
        for model in failed_models:
            print(f"   - {model}")
    
    print(f"\nTotal: {len(working_models)}/{len(models_to_test)} models working")
    
    return working_models


if __name__ == "__main__":
    working_models = main()
    
    # Save results for updating model registry
    import json
    with open("luma_models_test_results.json", "w") as f:
        json.dump({
            "working_models": working_models,
            "tested_models": ["ray-flash-2", "ray-2", "ray-1-6"]
        }, f, indent=2)
