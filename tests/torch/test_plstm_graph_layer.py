import pytest
import torch
import networkx as nx
import random

from plstm.config.plstm_graph_layer import (
    pLSTMGraphLayerConfig,
    pLSTMGraphEdgeLayerConfig,
)
from plstm.torch.plstm_graph_layer import (
    pLSTMGraphLayer,
    pLSTMGraphEdgeLayer,
)
from plstm.graph import PreparedGraph
from plstm.graph import dagify


@pytest.mark.parametrize(
    "num_nodes,mode",
    [
        (3, "P"),
        (5, "P"),
        (3, "D"),
        (5, "D"),
    ],
)
def test_plstm_graph_layer_forward(num_nodes, mode):
    """Test the forward pass of the pLSTMGraphLayer."""
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

    # Create layer configuration
    config = pLSTMGraphLayerConfig(
        mode=mode,
        input_dim=input_dim,
        output_dim=input_dim,
        num_heads=num_heads,
        qk_head_dim=input_dim // num_heads,
        hv_head_dim=input_dim // num_heads,
        max_edges=max_edges,
        bias=True,
        dtype="float64",
        param_dtype="float64",
    )

    # Initialize the layer
    layer = pLSTMGraphLayer(config)

    # Create input tensor
    node_features = torch.randn(num_nodes, input_dim, dtype=torch.float64)

    prepared_graph = PreparedGraph.create(DAG)

    # Forward pass
    output = layer(node_features, prepared_graph)

    # Check output shape
    assert output.shape == (num_nodes, input_dim), f"Expected output shape {(num_nodes, input_dim)}, got {output.shape}"


@pytest.mark.parametrize(
    "num_nodes,mode",
    [
        (3, "P"),
        (5, "D"),
    ],
)
def test_plstm_graph_layer_gradcheck(num_nodes, mode):
    """Test the gradient computation of the pLSTMGraphLayer using
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

    # Create layer configuration
    config = pLSTMGraphLayerConfig(
        mode=mode,
        input_dim=input_dim,
        output_dim=input_dim,
        num_heads=num_heads,
        qk_head_dim=input_dim // num_heads,
        hv_head_dim=input_dim // num_heads,
        max_edges=max_edges,
        bias=True,
        dtype="float64",
        param_dtype="float64",
    )

    # Initialize the layer
    layer = pLSTMGraphLayer(config).to(dtype=torch.double)

    # Create input tensor with double precision for gradcheck
    node_features = torch.randn(num_nodes, input_dim, dtype=torch.double, requires_grad=True)

    # Prepare the graph
    prepared_graph = PreparedGraph.create(DAG, mode=mode)

    # Define a function for gradcheck
    def func(x):
        return layer(x, prepared_graph)

    # Test gradients using gradcheck with relaxed tolerances
    try:
        result = torch.autograd.gradcheck(
            func,
            (node_features,),
            eps=1e-6,
            atol=1e-4,
        )
        assert result, "Gradcheck failed for pLSTMGraphLayer"
    except RuntimeError as e:
        pytest.fail(f"Gradcheck failed for pLSTMGraphLayer with error: {e}")


@pytest.mark.parametrize(
    "num_nodes,mode",
    [
        (3, "P"),
        (5, "D"),
    ],
)
def test_plstm_graph_edge_layer_forward(num_nodes, mode):
    """Test the forward pass of the pLSTMGraphEdgeLayer."""
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
    edge_input_dim = 8
    num_heads = 2
    max_edges = max(
        max([len(list(DAG.out_edges(n))) for n in DAG.nodes()], default=0),
        max([len(list(DAG.in_edges(n))) for n in DAG.nodes()], default=0),
    )
    max_edges = max(max_edges, 1)  # Ensure at least 1 to avoid empty tensors

    # Create layer configuration
    config = pLSTMGraphEdgeLayerConfig(
        mode=mode,
        input_dim=input_dim,
        edge_input_dim=edge_input_dim,
        output_dim=input_dim,
        num_heads=num_heads,
        qk_head_dim=input_dim // num_heads,
        hv_head_dim=input_dim // num_heads,
        max_edges=max_edges,
        bias=True,
        dtype="float64",
        param_dtype="float64",
    )

    # Initialize the layer
    layer = pLSTMGraphEdgeLayer(config)

    # Create input tensors
    node_features = torch.randn(num_nodes, input_dim, dtype=torch.float64)
    edge_features = torch.randn(DAG.number_of_edges(), edge_input_dim, dtype=torch.float64)

    prepared_graph = PreparedGraph.create(DAG)

    # Forward pass
    output = layer(node_features, edge_features, prepared_graph)

    # Check output shape
    assert output.shape == (num_nodes, input_dim), f"Expected output shape {(num_nodes, input_dim)}, got {output.shape}"


@pytest.mark.parametrize(
    "num_nodes,mode",
    [
        (3, "P"),
        (5, "D"),
    ],
)
def test_plstm_graph_edge_layer_gradcheck(num_nodes, mode):
    """Test the gradient computation of the pLSTMGraphEdgeLayer using
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
    edge_input_dim = 4
    num_heads = 2
    max_edges = max(
        max([len(list(DAG.out_edges(n))) for n in DAG.nodes()], default=0),
        max([len(list(DAG.in_edges(n))) for n in DAG.nodes()], default=0),
    )
    max_edges = max(max_edges, 1)  # Ensure at least 1 to avoid empty tensors

    # Create layer configuration
    config = pLSTMGraphEdgeLayerConfig(
        mode=mode,
        input_dim=input_dim,
        edge_input_dim=edge_input_dim,
        output_dim=input_dim,
        num_heads=num_heads,
        qk_head_dim=input_dim // num_heads,
        hv_head_dim=input_dim // num_heads,
        max_edges=max_edges,
        bias=True,
        dtype="float64",
        param_dtype="float64",
    )

    # Initialize the layer
    layer = pLSTMGraphEdgeLayer(config).to(dtype=torch.double)

    # Create input tensors with double precision for gradcheck
    node_features = torch.randn(num_nodes, input_dim, dtype=torch.double, requires_grad=True)
    edge_features = torch.randn(DAG.number_of_edges(), edge_input_dim, dtype=torch.double, requires_grad=True)

    # Prepare the graph
    prepared_graph = PreparedGraph.create(DAG, mode=mode)

    # Define a function for gradcheck
    def func(x, y):
        return layer(x, y, prepared_graph)

    # Test gradients using gradcheck with relaxed tolerances
    try:
        result = torch.autograd.gradcheck(
            func,
            (node_features, edge_features),
            eps=1e-6,
            atol=1e-4,
        )
        assert result, "Gradcheck failed for pLSTMGraphEdgeLayer"
    except RuntimeError as e:
        pytest.fail(f"Gradcheck failed for pLSTMGraphEdgeLayer with error: {e}")


# def test_plstm_graph_edge_layer_correctness():
#     """Test that ldag_idx_map is sorted in the topological order of the line
#     DAG."""
#     # Create a simple DAG
#     DAG = nx.DiGraph()
#     DAG.add_edges_from([(0, 2), (1, 2), (0, 1), (0, 3), (2, 3)])

#     input_dim = 8  # Smaller dimension for faster gradcheck
#     edge_input_dim = 4
#     num_heads = 2

#     config = pLSTMGraphEdgeLayerConfig(
#         mode="P",
#         input_dim=input_dim,
#         qk_head_dim=1,
#         edge_input_dim=edge_input_dim,
#         output_dim=input_dim,
#         num_heads=num_heads,
#         hv_head_dim=input_dim // num_heads,
#         bias=True,
#         dtype="float64",
#         param_dtype="float64",
#     )

#     layer = pLSTMGraphEdgeLayer(config)

#     inp = torch.zeros([4, input_dim], dtype=torch.float64)
#     inp[0] = 1.0
#     inp_edge = torch.zeros([5, edge_input_dim], dtype=torch.float64)
#     with torch.no_grad():
#         layer.query.bias.data.zero_()
#         layer.query.bias.data += 1.0
#         layer.key.bias.data.zero_()
#         layer.key.bias.data += 1.0
#         layer.value_.bias.data.zero_()
#         layer.value_.bias[:1] += 1.0
#         layer.transition.weight.data.zero_()
#         layer.transition.weight += 1.0
#         layer.transition.bias.data.zero_()
#         layer.source.weight.data.zero_()
#         layer.source.weight += 1.0
#         layer.mark.weight.data.zero_()
#         layer.mark.weight += 1.0
#         layer.direct.weight.data.zero_()
#         layer.direct.bias.data.zero_()
#         layer.direct.bias -= 20.0
#     inp_edge[3] = 1.0
#     import pudb

#     pudb.set_trace()

#     res = layer(inp, inp_edge, graph=DAG)
#     torch.all(res[3][0] > res[:3][0])
