#!/usr/bin/env python3
"""
Integration test to verify _make_graph and _plot methods work together.
"""

import sys
import os
sys.path.append('NASimEmu-agents')
sys.path.append('src')

import numpy as np
import torch
from nasim_problem.nasim_debug import NASimDebug

def create_mock_state():
    """Create a mock environment state for testing"""
    # Create mock node features (host data)
    # Each row represents a host with various features
    node_feats = np.array([
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # Host (1, 0) - normal host
        [0, 1, 1, 0, 1, 0, 0, 0, 0, 5],  # Host (1, 1) - sensitive host with value 5
        [0, 2, 0, 0, 1, 0, 0, 0, 0, 0],  # Host (2, 0) - different subnet
    ])
    
    return node_feats

def test_integration():
    """Test that _make_graph and _plot work together"""
    print("Testing _make_graph and _plot integration...")
    
    debug_instance = NASimDebug()
    
    # Create mock state
    s = create_mock_state()
    
    # Create mock neural network outputs
    node_softmax = torch.tensor([0.1, 0.8, 0.1])  # High attention on second node
    action_softmax = torch.tensor([0.2, 0.7, 0.1])  # High action probability on second node
    
    try:
        # Test _make_graph
        G = debug_instance._make_graph(s, node_softmax, action_softmax)
        
        # Verify graph was created
        assert G is not None, "Graph should not be None"
        assert len(G.nodes) > 0, "Graph should have nodes"
        
        # Verify node attributes are set
        for node_id in G.nodes:
            node = G.nodes[node_id]
            assert 'pos' in node, f"Node {node_id} should have position"
            assert 'label' in node, f"Node {node_id} should have label"
            assert 'n_prob' in node, f"Node {node_id} should have node probability"
            assert 'a_prob' in node, f"Node {node_id} should have action probability"
        
        print("✓ _make_graph created graph with proper attributes")
        
        # Test _plot with the created graph
        value = 0.85
        q_val = 1.42
        
        fig = debug_instance._plot(G, value, q_val, None)
        
        # Verify figure was created
        assert fig is not None, "Figure should not be None"
        assert hasattr(fig, 'data'), "Figure should have data"
        
        print("✓ _plot created figure from _make_graph output")
        
        # Verify neural network values are displayed
        title_text = fig.layout.title.text
        assert f"{value:.4f}" in title_text, "Title should contain state value"
        assert f"{q_val:.4f}" in title_text, "Title should contain Q-value"
        
        print("✓ Neural network values properly displayed")
        
        # Check that probability information is preserved
        node_trace = fig.data[-1]  # Last trace should be nodes
        node_texts = node_trace.text
        
        # Should have probability information in node texts
        prob_texts = [text for text in node_texts if "N:" in text and "A:" in text]
        assert len(prob_texts) > 0, "Some nodes should display probability information"
        
        print("✓ Probability information preserved in visualization")
        
        print(f"\n🎉 Integration test passed!")
        print(f"  - Graph created with {len(G.nodes)} nodes and {len(G.edges)} edges")
        print(f"  - All node attributes properly set")
        print(f"  - Figure generated with neural network annotations")
        print(f"  - Probability information displayed correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\nTesting error handling...")
    
    debug_instance = NASimDebug()
    
    try:
        # Test with None probabilities
        s = create_mock_state()
        G = debug_instance._make_graph(s, None, None)
        fig = debug_instance._plot(G, 0.0, 0.0, None)
        
        assert fig is not None, "Should handle None probabilities gracefully"
        print("✓ Handles None probabilities correctly")
        
        # Test with mismatched probability dimensions
        node_softmax = torch.tensor([0.5])  # Too few values
        action_softmax = torch.tensor([0.3, 0.7, 0.1, 0.2])  # Too many values
        
        G = debug_instance._make_graph(s, node_softmax, action_softmax)
        fig = debug_instance._plot(G, 1.0, 2.0, None)
        
        assert fig is not None, "Should handle mismatched dimensions gracefully"
        print("✓ Handles mismatched probability dimensions correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running integration tests...\n")
    
    test1_passed = test_integration()
    test2_passed = test_error_handling()
    
    print(f"\nTest Results:")
    print(f"  Integration test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Error handling: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 All integration tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ Some integration tests failed.")
        sys.exit(1)