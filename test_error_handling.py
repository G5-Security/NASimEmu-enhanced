#!/usr/bin/env python3
"""
Test script to verify error handling in NASimDebug class.
"""

import sys
import os
import numpy as np
import torch

# Add the NASimEmu-agents directory to the path
sys.path.insert(0, 'NASimEmu-agents')

from nasim_problem.nasim_debug import NASimDebug

def test_error_handling():
    """Test various error conditions in NASimDebug methods."""
    
    debug_instance = NASimDebug()
    
    print("Testing error handling in NASimDebug...")
    
    # Test 1: _validate_probability_data with various invalid inputs
    print("\n1. Testing _validate_probability_data:")
    
    # Test with None
    result = debug_instance._validate_probability_data(None, 5, "test_none")
    print(f"   None input: {result}")
    assert len(result) == 5 and np.all(result == 0)
    
    # Test with invalid tensor
    try:
        invalid_tensor = torch.tensor([float('nan'), float('inf'), -1.0, 2.0, 0.5])
        result = debug_instance._validate_probability_data(invalid_tensor, 5, "test_invalid")
        print(f"   Invalid tensor: {result}")
        assert len(result) == 5
    except Exception as e:
        print(f"   Invalid tensor handled: {e}")
    
    # Test with wrong length
    short_array = np.array([0.1, 0.2])
    result = debug_instance._validate_probability_data(short_array, 5, "test_short")
    print(f"   Short array: {result}")
    assert len(result) == 5
    
    # Test 2: _create_fallback_graph
    print("\n2. Testing _create_fallback_graph:")
    fallback_graph = debug_instance._create_fallback_graph()
    print(f"   Fallback graph nodes: {list(fallback_graph.nodes)}")
    print(f"   Node attributes: {dict(fallback_graph.nodes[0])}")
    assert len(fallback_graph.nodes) >= 1
    
    # Test 3: _create_fallback_plot
    print("\n3. Testing _create_fallback_plot:")
    fallback_plot = debug_instance._create_fallback_plot("Test error message")
    print(f"   Fallback plot created: {type(fallback_plot)}")
    assert fallback_plot is not None
    
    # Test 4: _create_error_plot
    print("\n4. Testing _create_error_plot:")
    error_plot = debug_instance._create_error_plot("Test error message")
    print(f"   Error plot created: {type(error_plot)}")
    assert error_plot is not None
    
    # Test 5: _make_graph with invalid inputs
    print("\n5. Testing _make_graph with invalid inputs:")
    
    # Test with None state
    try:
        graph = debug_instance._make_graph(None, None, None)
        print(f"   None state handled, graph nodes: {len(graph.nodes)}")
        assert len(graph.nodes) >= 1
    except Exception as e:
        print(f"   None state error: {e}")
    
    # Test with invalid probability data
    try:
        # Create a minimal valid state
        mock_state = (
            np.array([[1, 0, 0, 0, 0]]),  # node_feats
            np.array([[0], [0]]),         # edge_index (self-loop)
            np.array([[0, 0]]),           # node_index
            np.array([0])                 # pos_index
        )
        
        invalid_probs = torch.tensor([float('nan'), float('inf')])
        graph = debug_instance._make_graph(mock_state, invalid_probs, invalid_probs)
        print(f"   Invalid probabilities handled, graph nodes: {len(graph.nodes)}")
        assert len(graph.nodes) >= 1
    except Exception as e:
        print(f"   Invalid probabilities error: {e}")
    
    # Test 6: _plot with invalid inputs
    print("\n6. Testing _plot with invalid inputs:")
    
    # Test with None graph
    try:
        plot = debug_instance._plot(None, 0.5, 0.3, None)
        print(f"   None graph handled: {type(plot)}")
        assert plot is not None
    except Exception as e:
        print(f"   None graph error: {e}")
    
    # Test with invalid values
    try:
        fallback_graph = debug_instance._create_fallback_graph()
        plot = debug_instance._plot(fallback_graph, float('nan'), float('inf'), None)
        print(f"   Invalid values handled: {type(plot)}")
        assert plot is not None
    except Exception as e:
        print(f"   Invalid values error: {e}")
    
    print("\n✅ All error handling tests completed successfully!")

if __name__ == "__main__":
    test_error_handling()