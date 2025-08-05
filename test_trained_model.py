#!/usr/bin/env python3
"""
Comprehensive script to test your trained model
Run this after training is complete
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and show results"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="NASimEmu-agents")
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def find_trained_model():
    """Find the path to your trained model"""
    possible_paths = [
        "wandb/latest-run/files/model.pt",
        "wandb/offline-run-*/files/model.pt"
    ]
    
    for path in possible_paths:
        full_path = os.path.join("NASimEmu-agents", path)
        if "*" in path:
            # Use glob to find pattern matches
            import glob
            matches = glob.glob(full_path)
            if matches:
                # Get the most recent one
                matches.sort(key=os.path.getmtime, reverse=True)
                return matches[0].replace("NASimEmu-agents/", "")
        else:
            if os.path.exists(full_path):
                return path
    
    return None

def main():
    print("🚀 Testing Your Trained NASimEmu Model")
    print("=" * 60)
    
    # Find your trained model
    model_path = find_trained_model()
    if not model_path:
        print("❌ No trained model found!")
        print("Expected locations:")
        print("  - NASimEmu-agents/wandb/latest-run/files/model.pt")
        print("  - NASimEmu-agents/wandb/offline-run-*/files/model.pt")
        return
    
    print(f"✅ Found trained model: {model_path}")
    
    # Test commands
    tests = [
        {
            "cmd": f"python main.py ../scenarios/uni.v2.yaml --eval -load_model {model_path}",
            "desc": "Evaluate Model Performance"
        },
        {
            "cmd": f"python main.py ../scenarios/uni.v2.yaml --debug -load_model {model_path}",
            "desc": "Enhanced Debug Visualization (opens in browser)"
        },
        {
            "cmd": f"python main.py ../scenarios/uni.v2.yaml --calc_baseline",
            "desc": "Calculate Baseline Performance (for comparison)"
        }
    ]
    
    # Run tests
    results = []
    for test in tests:
        success = run_command(test["cmd"], test["desc"])
        results.append((test["desc"], success))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    for desc, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {desc}")
    
    print(f"\n🎯 Your trained model: {model_path}")
    print("\n📋 Additional Commands You Can Try:")
    print(f"  # Watch step-by-step execution:")
    print(f"  cd NASimEmu-agents && python main.py ../scenarios/uni.v2.yaml --trace -load_model {model_path}")
    print(f"  ")
    print(f"  # Compare with pre-trained models:")
    print(f"  cd NASimEmu-agents && python main.py ../scenarios/uni.v2.yaml --eval -load_model trained_models/gnn-mact.pt")
    print(f"  cd NASimEmu-agents && python main.py ../scenarios/uni.v2.yaml --eval -load_model trained_models/att-mact.pt")

if __name__ == "__main__":
    main()