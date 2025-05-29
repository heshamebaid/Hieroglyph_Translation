import requests
import json
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
IMAGE_PATH = Path(r"D:\Hieroglyph_Translation\resources\test_image.jpg")

def test_api():
    """Test single image translation endpoint"""
    try:
        # First check if API is healthy
        health_response = requests.get(f"{BASE_URL}/health")
        if health_response.status_code != 200:
            print(f"❌ API not healthy: {health_response.text}")
            return False

        # Ensure test image exists
        if not IMAGE_PATH.exists():
            print(f"❌ Test image not found at {IMAGE_PATH}")
            return False

        # Process single image
        with open(IMAGE_PATH, "rb") as image_file:
            files = {
                "file": ("test_image.jpg", image_file, "image/jpeg")
            }
            
            print(f"📤 Sending image: {IMAGE_PATH}")
            response = requests.post(
                f"{BASE_URL}/translate",
                files=files
            )

        if response.status_code == 200:
            results = response.json()
            print("\n📊 Processing Results:")
            print(f"📁 Session directory: {results.get('session_dir', 'N/A')}")
            
            # Print symbol classifications
            classifications = results.get('classifications', [])
            if classifications:
                print("\n🔍 Classifications:")
                for idx, symbol in enumerate(classifications[:5], 1):
                    print(f"  Symbol {idx}:")
                    print(f"    Code: {symbol.get('Gardiner Code', 'Unknown')}")
                    print(f"    Confidence: {symbol.get('confidence', 'N/A'):.2%}")
                if len(classifications) > 5:
                    print(f"    ... and {len(classifications) - 5} more symbols")
            
            # Print generated story
            if results.get('story'):
                print("\n📜 Generated Story:")
                print("-" * 50)
                print(results['story'])
                print("-" * 50)
            
            # Print any errors
            if results.get('error'):
                print(f"\n⚠️ Processing warning: {results['error']}")
            
            print("\n✅ Test completed successfully!")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"🔥 Critical error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting tests...")
    
    print("\n1️⃣ Testing hieroglyph translation...")
    if test_api():
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Test failed")