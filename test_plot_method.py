#!/usr/bin/env python3
"""
Test script to verify the _plot method implementation in NASimDebug class.
"""

import sys
import os
sys.path.append('NASimEmu-agents')
sys.path.append('src')

import numpy as np
import torch
import networkx as nx
from nasim_problem.nasim_debug import NASimDebug

def create_mock_graph():
    """Create a mock NetworkX graph with the expected node attributes"""
    G = nx.Graph()
    
    # Add some nodes and edges
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])
    
    # Set node positions
    pos = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}
    nx.set_node_attributes(G, pos, 'pos')
    
    # Set node attributes as expected by _plot method
    node_labels = {0: "Host (1, 0)", 1: "Host (1, 1)", 2: "Subnet 2", 3: "Host (1, 2)"}
    node_types = {0: "node", 1: "node", 2: "subnet", 3: "node"}
    node_colors = {0: "lightblue", 1: "red", 2: "grey", 3: "lightblue"}
    node_symbols = {0: "circle", 1: "circle", 2: "triangle-up", 3: "circle"}
    node_n_probs = {0: 0.1, 1: 0.8, 2: 0.0, 3: 0.3}
    node_a_probs = {0: 0.2, 1: 0.9, 2: 0.0, 3: 0.1}
    node_line_widths = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
    
    nx.set_node_attributes(G, node_labels, 'label')
    nx.set_node_attributes(G, node_types, 'type')
    nx.set_node_attributes(G, node_colors, 'color')
    nx.set_node_attributes(G, node_symbols, 'symbol')
    nx.set_node_attributes(G, node_n_probs, 'n_prob')
    nx.set_node_attributes(G, node_a_probs, 'a_prob')
    nx.set_node_attributes(G, node_line_widths, 'line_width')
    
    return G

def test_plot_method():
    """Test the _plot method with mock data"""
    print("Testing _plot method implementation...")
    
    # Create NASimDebug instance
    debug_instance = NASimDebug()
    
    # Create mock graph
    G = create_mock_graph()
    
    # Mock neural network values
    value = 0.75
    q_val = 1.23
    
    # Mock test environment (we don't actually use it in the current implementation)
    test_env = None
    
    try:
        # Call the _plot method
        fig = debug_instance._plot(G, value, q_val, test_env)
        
        # Verify the figure was created
        assert fig is not None, "Figure should not be None"
        
        # Check that the figure has the expected structure
        assert hasattr(fig, 'data'), "Figure should have data attribute"
        assert len(fig.data) >= 3, "Figure should have at least 3 traces (edges, subnets, nodes)"
        
        # Check that the layout includes our neural network values
        layout_title = fig.layout.title.text
        assert f"{value:.4f}" in layout_title, f"Title should contain value {value}"
        assert f"{q_val:.4f}" in layout_title, f"Title should contain q_val {q_val}"
        
        # Check annotations
        assert len(fig.layout.annotations) > 0, "Figure should have annotations"
        annotation_text = fig.layout.annotations[0].text
        assert f"V(s) = {value:.4f}" in annotation_text, "Annotation should contain state value"
        assert f"Q(s,a) = {q_val:.4f}" in annotation_text, "Annotation should contain Q-value"
        
        print("✓ _plot method test passed!")
        print(f"  - Figure created successfully")
        print(f"  - Title includes neural network values")
        print(f"  - Annotations include V(s) and Q(s,a) values")
        print(f"  - Figure has {len(fig.data)} traces")
        
        return True
        
    except Exception as e:
        print(f"✗ _plot method test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plot_with_probabilities():
    """Test that probability information is correctly displayed"""
    print("\nTesting probability information display...")
    
    debug_instance = NASimDebug()
    G = create_mock_graph()
    
    try:
        fig = debug_instance._plot(G, 0.5, 1.0, None)
        
        # Check node trace (should be the last trace)
        node_trace = fig.data[-1]
        
        # Verify that probability information is in the text
        node_texts = node_trace.text
        assert any("N:" in text and "A:" in text for text in node_texts), \
            "Node texts should contain probability information"
        
        # Verify hover text contains detailed probability info
        hover_texts = node_trace.hovertext
        assert any("Node Probability:" in text for text in hover_texts), \
            "Hover text should contain node probability information"
        assert any("Action Probability:" in text for text in hover_texts), \
            "Hover text should contain action probability information"
        
        # Check that high attention nodes are marked
        assert any("High Attention" in text for text in hover_texts), \
            "High probability nodes should be marked with High Attention"
        
        print("✓ Probability information test passed!")
        print(f"  - Node texts include probability values")
        print(f"  - Hover text includes detailed probability information")
        print(f"  - High attention nodes are properly marked")
        
        return True
        
    except Exception as e:
        print(f"✗ Probability information test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running _plot method tests...\n")
    
    test1_passed = test_plot_method()
    test2_passed = test_plot_with_probabilities()
    
    print(f"\nTest Results:")
    print(f"  Basic _plot functionality: {'PASS' if test1_passed else 'FAIL'}")
    print(f"  Probability information: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 All tests passed! The _plot method is working correctly.")
        sys.exit(0)
    else:
        print(f"\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)