#!/usr/bin/env python3
"""
Test your partially trained model and compare with pre-trained models
"""

import os
import subprocess
import glob

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
            print("Warnings:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def find_latest_model():
    """Find your latest trained model"""
    patterns = [
        "NASimEmu-agents/wandb/latest-run/files/model.pt",
        "NASimEmu-agents/wandb/offline-run-*/files/model.pt"
    ]
    
    for pattern in patterns:
        if "*" in pattern:
            matches = glob.glob(pattern)
            if matches:
                # Get most recent
                matches.sort(key=os.path.getmtime, reverse=True)
                return matches[0].replace("NASimEmu-agents/", "")
        else:
            if os.path.exists(pattern):
                return pattern.replace("NASimEmu-agents/", "")
    
    return None

def main():
    print("🚀 Testing Your Partially Trained Model")
    print("=" * 60)
    
    # Find your model
    model_path = find_latest_model()
    if not model_path:
        print("❌ No trained model found!")
        return
    
    print(f"✅ Found your model: {model_path}")
    
    # Base parameters for your model
    base_params = "-net_class NASimNetMLP -use_a_t -episode_step_limit 100 -augment_with_action"
    
    # Test your model
    tests = [
        {
            "cmd": f"python main.py ../scenarios/uni.v2.yaml --eval -load_model {model_path} {base_params}",
            "desc": "Evaluate Your Trained Model (Expected: ~19.5% success)"
        },
        {
            "cmd": f"python main.py ../scenarios/uni.v2.yaml --debug -load_model {model_path} {base_params}",
            "desc": "Enhanced Visualization of Your Model"
        },
        {
            "cmd": f"python main.py ../scenarios/uni.v2.yaml --eval -load_model trained_models/mlp.pt {base_params}",
            "desc": "Compare with Pre-trained MLP (Expected: ~80-90% success)"
        },
        {
            "cmd": f"python main.py ../scenarios/uni.v2.yaml --calc_baseline",
            "desc": "Baseline Performance (for reference)"
        }
    ]
    
    # Run tests
    for test in tests:
        run_command(test["cmd"], test["desc"])
    
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    print(f"🎯 Your model: {model_path}")
    print("📈 Expected performance: ~19.5% success rate, -8.05 reward")
    print("🎨 Enhanced visualization should show yellow/orange nodes")
    print("🔄 To resume training:")
    print(f"   cd NASimEmu-agents")
    print(f"   python main.py ../scenarios/uni.v2.yaml -load_model {model_path} {base_params} -max_epochs 100 -force_continue_epochs 30")

if __name__ == "__main__":
    main()