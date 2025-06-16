import pytest
import torch
import networkx as nx
import random

from plstm.config.graph_block import (
    pLSTMGraphBlockConfig,
    pLSTMGraphEdgeBlockConfig,
)
from plstm.torch.graph_block import (
    pLSTMGraphBlock,
    pLSTMGraphEdgeBlock,
)
from plstm.graph import PreparedGraph
from plstm.graph import dagify


@pytest.mark.parametrize(
    "num_nodes,block_mode",
    [
        (3, "PD"),
        (5, "PD"),
        (3, "DP"),
        (5, "DP"),
    ],
)
def test_plstm_graph_block_forward(num_nodes, block_mode):
    """Test the forward pass of the pLSTMGraphBlock."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Create a random graph and convert to DAG
    extra_edge_prob = 0.2
    G = nx.random_labeled_tree(num_nodes)

    # Add extra random edges while ensuring the graph remains simple
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if not G.has_edge(u, v) and random.random() < extra_edge_prob:
                G.add_edge(u, v)

    DAG = dagify(G)

    # Set up dimensions
    input_dim = 16
    num_heads = 2
    max_edges = max(
        max([len(list(DAG.out_edges(n))) for n in DAG.nodes()], default=0),
        max([len(list(DAG.in_edges(n))) for n in DAG.nodes()], default=0),
    )
    max_edges = max(max_edges, 1)  # Ensure at least 1 to avoid empty tensors

    # Create block configuration
    config = pLSTMGraphBlockConfig(
        input_dim=input_dim,
        num_heads=num_heads,
        block_mode=block_mode,
        max_edges=max_edges,
        dtype="float64",
        param_dtype="float64",
    )

    # Initialize the block
    block = pLSTMGraphBlock(config)

    # Create input tensor
    node_features = torch.randn(num_nodes, input_dim, dtype=torch.float64)

    # Forward pass
    output = block(node_features, graph=DAG, deterministic=True)

    # Check output shape
    assert output.shape == (num_nodes, input_dim), f"Expected output shape {(num_nodes, input_dim)}, got {output.shape}"


@pytest.mark.parametrize(
    "num_nodes,block_mode",
    [
        (3, "PD"),
        (5, "DP"),
    ],
)
def test_plstm_graph_block_gradcheck(num_nodes, block_mode):
    """Test the gradient computation of the pLSTMGraphBlock using
    torch.autograd.gradcheck."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Create a random graph and convert to DAG
    extra_edge_prob = 0.2
    G = nx.random_labeled_tree(num_nodes)

    # Add extra random edges while ensuring the graph remains simple
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if not G.has_edge(u, v) and random.random() < extra_edge_prob:
                G.add_edge(u, v)

    DAG = dagify(G)

    # Set up dimensions
    input_dim = 8  # Smaller dimension for faster gradcheck
    num_heads = 2
    max_edges = max(
        max([len(list(DAG.out_edges(n))) for n in DAG.nodes()], default=0),
        max([len(list(DAG.in_edges(n))) for n in DAG.nodes()], default=0),
    )
    max_edges = max(max_edges, 1)  # Ensure at least 1 to avoid empty tensors

    # Create block configuration
    config = pLSTMGraphBlockConfig(
        input_dim=input_dim,
        num_heads=num_heads,
        block_mode=block_mode,
        max_edges=max_edges,
        dtype="float64",
        param_dtype="float64",
    )

    # Initialize the block
    block = pLSTMGraphBlock(config).to(dtype=torch.double)

    # Create input tensor with float precision for gradcheck
    node_features = torch.randn(num_nodes, input_dim, dtype=torch.double, requires_grad=True)

    # Prepare the graph
    prepared_graph = PreparedGraph.create(DAG)

    # Define a function for gradcheck
    def func(x):
        return block(x, graph=prepared_graph, deterministic=True)

    # Test gradients using gradcheck with relaxed tolerances
    try:
        result = torch.autograd.gradcheck(
            func,
            (node_features,),
            eps=1e-6,
            atol=1e-4,
        )
        assert result, "Gradcheck failed for pLSTMGraphBlock"
    except RuntimeError as e:
        pytest.fail(f"Gradcheck failed for pLSTMGraphBlock with error: {e}")


@pytest.mark.parametrize(
    "num_nodes,block_mode",
    [
        (3, "PD"),
        (5, "PD"),
        (3, "DP"),
        (5, "DP"),
    ],
)
def test_plstm_graph_edge_block_forward(num_nodes, block_mode):
    """Test the forward pass of the pLSTMGraphEdgeBlock."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Create a random graph and convert to DAG
    extra_edge_prob = 0.2
    G = nx.random_labeled_tree(num_nodes)

    # Add extra random edges while ensuring the graph remains simple
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if not G.has_edge(u, v) and random.random() < extra_edge_prob:
                G.add_edge(u, v)

    DAG = dagify(G)

    # Set up dimensions
    input_dim = 16
    num_heads = 2

    # Create block configuration
    config = pLSTMGraphEdgeBlockConfig(
        input_dim=input_dim,
        num_heads=num_heads,
        block_mode=block_mode,
        dtype="float64",
        param_dtype="float64",
    )

    # Initialize the block
    block = pLSTMGraphEdgeBlock(config)

    # Create input tensors
    node_features = torch.randn(num_nodes, input_dim, dtype=torch.float64)
    edge_features = torch.randn(DAG.number_of_edges(), input_dim, dtype=torch.float64)

    prepared_graph = PreparedGraph.create(DAG)

    # Forward pass
    output = block(node_features, edge_features=edge_features, graph=prepared_graph, deterministic=True)

    # Check output shape
    assert output.shape == (num_nodes, input_dim), f"Expected output shape {(num_nodes, input_dim)}, got {output.shape}"


@pytest.mark.parametrize(
    "num_nodes,block_mode",
    [
        (3, "PD"),
        (5, "DP"),
    ],
)
def test_plstm_graph_edge_block_gradcheck(num_nodes, block_mode):
    """Test the gradient computation of the pLSTMGraphEdgeBlock using
    torch.autograd.gradcheck."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    # Create a random graph and convert to DAG
    extra_edge_prob = 0.2
    G = nx.random_labeled_tree(num_nodes)

    # Add extra random edges while ensuring the graph remains simple
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if not G.has_edge(u, v) and random.random() < extra_edge_prob:
                G.add_edge(u, v)

    DAG = dagify(G)

    # Set up dimensions
    input_dim = 8  # Smaller dimension for faster gradcheck
    num_heads = 2

    # Create block configuration
    config = pLSTMGraphEdgeBlockConfig(
        input_dim=input_dim,
        num_heads=num_heads,
        block_mode=block_mode,
        dtype="float64",
        param_dtype="float64",
    )

    # Initialize the block
    block = pLSTMGraphEdgeBlock(config).to(dtype=torch.double)

    # Create input tensors with float precision for gradcheck
    node_features = torch.randn(num_nodes, input_dim, dtype=torch.double, requires_grad=True)
    edge_features = torch.randn(DAG.number_of_edges(), input_dim, dtype=torch.double, requires_grad=True)

    # Prepare the graph
    prepared_graph = PreparedGraph.create(DAG)

    # Define a function for gradcheck
    def func(x, y):
        return block(x, edge_features=y, graph=prepared_graph, deterministic=True)

    # Test gradients using gradcheck with relaxed tolerances
    try:
        result = torch.autograd.gradcheck(
            func,
            (node_features, edge_features),
            eps=1e-6,
            atol=1e-4,
        )
        assert result, "Gradcheck failed for pLSTMGraphEdgeBlock"
    except RuntimeError as e:
        pytest.fail(f"Gradcheck failed for pLSTMGraphEdgeBlock with error: {e}")
