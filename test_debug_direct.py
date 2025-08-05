#!/usr/bin/env python3
"""
Direct test of the debug method to verify it works as expected.
"""

import sys
sys.path.insert(0, 'NASimEmu-agents')

from nasim_problem.nasim_debug import NASimDebug
import torch

class MockNet:
    def __class__(self): return MockNet()
    def clone_state(self, other): pass
    def eval(self): pass
    def train(self): pass
    def reset_state(self): pass
    def __call__(self, state, complete=False):
        if complete:
            return (torch.tensor([0.2, 0.8]), 
                   torch.tensor([0.3, 0.7]), 
                   torch.tensor(0.9), 
                   torch.tensor(1.1))
        return None, None, None, None

# Test the debug method directly
debug_instance = NASimDebug()
mock_net = MockNet()

print("Testing debug method directly...")
result = debug_instance.debug(mock_net, show=False)

print(f"Result keys: {list(result.keys())}")
print(f"Value: {result['value']}")
print(f"Q-val: {result['q_val']}")
print(f"Figure type: {type(result['figure'])}")

print("✅ Debug method works correctly!")