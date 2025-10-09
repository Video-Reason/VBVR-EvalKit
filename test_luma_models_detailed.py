#!/usr/bin/env python3
"""
Test different Luma model endpoints with detailed error reporting
"""

import os
import sys
from pathlib import Path
import requests

sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

def test_model_direct(model_name: str):
    """Test model with direct API call to see exact error."""
    api_key = os.getenv("LUMA_API_KEY")
    if not api_key:
        print("Error: LUMA_API_KEY not set")
        return
    
    print(f"\nTesting {model_name} with direct API call...")
    
    # Simple payload without image to test model validity
    payload = {
        "prompt": "A serene lake at sunset",
        "model": model_name,
        "enhance_prompt": False,
        "loop": False,
        "aspect_ratio": "16:9",
        "duration": "5s",
        "resolution": "720p"
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            "https://api.lumalabs.ai/dream-machine/v1/generations",
            headers=headers,
            json=payload
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        
        if response.status_code == 200 or response.status_code == 201:
            print(f"✅ Model {model_name} appears to be valid!")
            return True
        else:
            print(f"❌ Model {model_name} failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing {model_name}: {e}")
        return False


def main():
    """Test all models."""
    models = ["ray-flash-2", "ray-2", "ray-1-6"]
    
    print("Testing Luma Model Endpoints - Direct API")
    print("=" * 60)
    
    results = {}
    for model in models:
        results[model] = test_model_direct(model)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    working = [m for m, ok in results.items() if ok]
    failed = [m for m, ok in results.items() if not ok]
    
    if working:
        print(f"\n✅ Working models: {', '.join(working)}")
    if failed:
        print(f"\n❌ Failed models: {', '.join(failed)}")


if __name__ == "__main__":
    main()
