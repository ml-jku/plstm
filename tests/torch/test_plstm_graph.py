import pytest
import torch
import networkx as nx
import random

from plstm.torch.plstm_graph import plstm_graph_nodewise, plstm_graph
from plstm.graph import dagify


@pytest.mark.parametrize(
    "num_nodes",
    [3, 5],
)
def test_plstm_graph_nodewise_gradcheck(num_nodes):
    """Test the gradient computation of the plstm_graph function using
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
    lDAG = nx.line_graph(DAG)

    # Set up dimensions
    H, N, K, V = 2, num_nodes, 3, 3

    # Create mappings for edges
    lDAG_idx_map = {edge: edge_idx for edge_idx, edge in enumerate(nx.topological_sort(lDAG))}

    # Create adjacency maps
    adjacency_forward_edgemap = {}
    adjacency_backward_edgemap = {}

    # For each node, map its outgoing edges to indices
    for node in DAG.nodes():
        for i, edge in enumerate(DAG.out_edges(node)):
            adjacency_forward_edgemap[edge] = i

    # For each node, map its incoming edges to indices
    for node in DAG.nodes():
        for i, edge in enumerate(DAG.in_edges(node)):
            adjacency_backward_edgemap[edge] = i

    # Maximum number of edges per node
    max_edges = max(
        max([len(list(DAG.out_edges(n))) for n in DAG.nodes()], default=0),
        max([len(list(DAG.in_edges(n))) for n in DAG.nodes()], default=0),
    )
    E = max(max_edges, 1)  # Ensure at least 1 to avoid empty tensors

    # Create input tensors with double precision for gradcheck
    q = torch.randn([H, N, K], dtype=torch.double, requires_grad=True)
    k = torch.randn([H, N, K], dtype=torch.double, requires_grad=True)
    v = torch.randn([H, N, V], dtype=torch.double, requires_grad=True)
    source = torch.randn([H, N, E], dtype=torch.double, requires_grad=True)
    transition = torch.randn([H, N, E, E], dtype=torch.double, requires_grad=True) / 10.0  # Scale down for stability
    mark = torch.randn([H, N, E], dtype=torch.double, requires_grad=True)
    direct = torch.randn([H, N], dtype=torch.double, requires_grad=True) / 10.0  # Scale down for stability

    # Test gradients using gradcheck
    try:
        result = torch.autograd.gradcheck(
            lambda q, k, v, s, t, m, d: plstm_graph_nodewise(
                q, k, v, s, t, m, d, adjacency_forward_edgemap, adjacency_backward_edgemap, lDAG, lDAG_idx_map
            ),
            (q, k, v, source, transition, mark, direct),
            eps=1e-6,
            atol=1e-4,
        )
        assert result, "Gradcheck failed for plstm_graph"
    except RuntimeError as e:
        pytest.fail(f"Gradcheck failed for plstm_graph with error: {e}")


@pytest.mark.parametrize(
    "num_nodes",
    [3, 5],
)
def test_plstm_graph_gradcheck(num_nodes):
    """Test the gradient computation of the plstm_graph function using
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
    lDAG = nx.line_graph(DAG)

    # Set up dimensions
    H, N, K, V = 2, num_nodes, 3, 3
    E, L = lDAG.number_of_nodes(), lDAG.number_of_edges()

    # Create mappings
    edge_list = {edge: edge_idx for edge_idx, edge in enumerate(nx.topological_sort(lDAG))}
    edge_comb_list = {edgepair: edgepair_idx for edgepair_idx, edgepair in enumerate(lDAG.edges)}

    # Create input tensors with double precision for gradcheck
    q = torch.randn([H, N, K], dtype=torch.double, requires_grad=True)
    k = torch.randn([H, N, K], dtype=torch.double, requires_grad=True)
    v = torch.randn([H, N, V], dtype=torch.double, requires_grad=True)
    s = torch.randn([H, E], dtype=torch.double, requires_grad=True)
    t = torch.randn([H, L], dtype=torch.double, requires_grad=True) / 10.0  # Scale down for numerical stability
    m = torch.randn([H, E], dtype=torch.double, requires_grad=True)
    d = torch.randn([H, N], dtype=torch.double, requires_grad=True) / 10.0  # Scale down for numerical stability

    # Test gradients using gradcheck
    try:
        result = torch.autograd.gradcheck(
            lambda q, k, v, s, t, m, d: plstm_graph(q, k, v, s, t, m, d, lDAG, edge_list, edge_comb_list),
            (q, k, v, s, t, m, d),
            eps=1e-6,
            atol=1e-4,
        )
        assert result, "Gradcheck failed for plstm_graph"
    except RuntimeError as e:
        pytest.fail(f"Gradcheck failed for plstm_graph with error: {e}")
