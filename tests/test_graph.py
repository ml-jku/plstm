import pytest
import networkx as nx
import random

from plstm.graph import (
    dagify,
    revert_dag,
    prune_dag_to_multitree,
    check_if_multitree,
    PreparedGraph,
)


def test_dagify():
    """Test that dagify correctly converts an undirected graph to a DAG."""
    # Create a simple undirected graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])  # Cycle 0-1-2-3-0

    # Convert to DAG
    DAG = dagify(G)

    # Check that the result is a DAG
    assert nx.is_directed_acyclic_graph(DAG)

    # Check that all nodes are preserved
    assert set(DAG.nodes()) == set(G.nodes())

    # Check that the number of edges is the same
    assert DAG.number_of_edges() == G.number_of_edges()

    # Check that edges are directed from lower to higher index
    for u, v in DAG.edges():
        assert u < v


def test_revert_dag():
    """Test that revert_dag correctly reverses a DAG."""
    # Create a simple DAG
    DAG = nx.DiGraph()
    DAG.add_edges_from([(0, 1), (1, 2), (2, 3)])

    # Reverse the DAG
    revDAG = revert_dag(DAG)

    # Check that the result is a DAG
    assert nx.is_directed_acyclic_graph(revDAG)

    # Check that all nodes are preserved (in reverse order)
    assert list(revDAG.nodes()) == list(reversed(list(DAG.nodes())))

    # Check that edges are reversed
    for u, v in DAG.edges():
        assert (v, u) in revDAG.edges()


def test_prune_dag_to_multitree():
    """Test that prune_dag_to_multitree correctly prunes a DAG to a
    multitree."""
    # Create a DAG that is not a multitree (has multiple paths between nodes)
    DAG = nx.DiGraph()
    DAG.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])  # Diamond shape with multiple paths from 0 to 3

    # Prune to multitree
    multitree = prune_dag_to_multitree(DAG)

    # Check that the result is a DAG
    assert nx.is_directed_acyclic_graph(multitree)

    # Check that all nodes are preserved
    assert set(multitree.nodes()) == set(DAG.nodes())

    # Check that it's a multitree (no multiple paths between any pair of nodes)
    assert check_if_multitree(multitree)

    # The pruned graph should have fewer edges than the original
    assert multitree.number_of_edges() < DAG.number_of_edges()


def test_check_if_multitree():
    """Test that check_if_multitree correctly identifies a multitree."""
    # Create a multitree (a tree is a special case of a multitree)
    multitree = nx.DiGraph()
    multitree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4)])

    # Check that it's identified as a multitree
    assert check_if_multitree(multitree)

    # Create a non-multitree (has multiple paths between nodes)
    non_multitree = nx.DiGraph()
    non_multitree.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])  # Diamond shape with multiple paths from 0 to 3

    # Check that it's not identified as a multitree
    assert not check_if_multitree(non_multitree)


def test_prepared_graph_create():
    """Test that PreparedGraph.create correctly creates a PreparedGraph from a
    graph."""
    # Create a simple undirected graph
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])

    # Create PreparedGraph
    pg = PreparedGraph.create(G)

    # Check that the DAG is created correctly
    assert nx.is_directed_acyclic_graph(pg.dag)
    assert set(pg.dag.nodes()) == set(G.nodes())
    assert pg.dag.number_of_edges() == G.number_of_edges()

    # Check that the line DAG is created correctly
    assert nx.is_directed_acyclic_graph(pg.ldag)

    # Check that mappings are created correctly
    assert len(pg.node_idx_map) == G.number_of_nodes()
    assert len(pg.dag_edge_orig_idx_map) == G.number_of_edges()
    assert len(pg.idx_node_map) == G.number_of_nodes()
    assert len(pg.orig_idx_dag_edge_map) == G.number_of_edges()

    # Check that edge indices are created correctly
    assert len(pg.topo_edge_incoming_nodeidx) == G.number_of_edges()
    assert len(pg.topo_edge_outgoing_nodeidx) == G.number_of_edges()
    assert len(pg.topo_edge_idx) == G.number_of_edges()


def test_prepared_graph_create_from_dag():
    """Test that PreparedGraph.create correctly creates a PreparedGraph from a
    DAG."""
    # Create a simple DAG
    DAG = nx.DiGraph()
    DAG.add_edges_from([(0, 1), (1, 2), (2, 3)])

    # Create PreparedGraph
    pg = PreparedGraph.create(DAG)

    # Check that the DAG is preserved
    assert pg.dag.nodes == DAG.nodes
    assert pg.dag.edges == DAG.edges

    # Check that the line DAG is created correctly
    assert nx.is_directed_acyclic_graph(pg.ldag)

    # Check that mappings are created correctly
    assert len(pg.node_idx_map) == DAG.number_of_nodes()
    assert len(pg.dag_edge_orig_idx_map) == DAG.number_of_edges()
    assert len(pg.idx_node_map) == DAG.number_of_nodes()
    assert len(pg.orig_idx_dag_edge_map) == DAG.number_of_edges()

    # Check that edge indices are created correctly
    assert len(pg.topo_edge_incoming_nodeidx) == DAG.number_of_edges()
    assert len(pg.topo_edge_outgoing_nodeidx) == DAG.number_of_edges()
    assert len(pg.topo_edge_idx) == DAG.number_of_edges()


def test_prepared_graph_create_with_mode_P():
    """Test that PreparedGraph.create correctly creates a PreparedGraph with
    mode='P'."""
    # Create a simple DAG
    DAG = nx.DiGraph()
    DAG.add_edges_from([(0, 1), (1, 2), (2, 3)])

    # Create PreparedGraph with mode='P'
    pg = PreparedGraph.create(DAG, mode="P")

    # Check that the DAG is preserved
    assert pg.dag.nodes == DAG.nodes
    assert pg.dag.edges == DAG.edges

    # Check that the line DAG is created correctly
    assert nx.is_directed_acyclic_graph(pg.ldag)

    # In mode 'P', the line DAG should not be pruned to a multitree
    # So it should be the same as the line graph of the DAG
    assert set(pg.ldag.nodes()) == set(nx.line_graph(DAG).nodes())
    assert set(pg.ldag.edges()) == set(nx.line_graph(DAG).edges())


def test_prepared_graph_reverse():
    """Test that PreparedGraph.reverse correctly reverses a PreparedGraph."""
    # Create a simple DAG
    DAG = nx.DiGraph()
    DAG.add_edges_from([(0, 1), (1, 2), (2, 3)])

    # Create PreparedGraph
    pg = PreparedGraph.create(DAG)

    # Reverse the PreparedGraph
    rev_pg = PreparedGraph.reverse(pg)

    # Check that the DAG is reversed
    assert set(rev_pg.dag.nodes()) == set(pg.dag.nodes())
    for u, v in pg.dag.edges():
        assert (v, u) in rev_pg.dag.edges()

    # Check that the line DAG is created correctly for the reversed DAG
    assert nx.is_directed_acyclic_graph(rev_pg.ldag)


def test_prepared_graph_with_random_graph():
    """Test PreparedGraph with a random graph."""
    # Set random seed for reproducibility
    random.seed(42)

    # Create a random graph
    num_nodes = 10
    G = nx.gnp_random_graph(num_nodes, 0.3)

    # Create PreparedGraph
    pg = PreparedGraph.create(G)

    # Check that the DAG is created correctly
    assert nx.is_directed_acyclic_graph(pg.dag)
    assert set(pg.dag.nodes()) == set(G.nodes())
    assert pg.dag.number_of_edges() == G.number_of_edges()

    # Check that the line DAG is created correctly
    assert nx.is_directed_acyclic_graph(pg.ldag)

    # Check that mappings are created correctly
    assert len(pg.node_idx_map) == G.number_of_nodes()
    assert len(pg.dag_edge_orig_idx_map) == G.number_of_edges()
    assert len(pg.idx_node_map) == G.number_of_nodes()
    assert len(pg.orig_idx_dag_edge_map) == G.number_of_edges()

    # Check that edge indices are created correctly
    assert len(pg.topo_edge_incoming_nodeidx) == G.number_of_edges()
    assert len(pg.topo_edge_outgoing_nodeidx) == G.number_of_edges()
    assert len(pg.topo_edge_idx) == G.number_of_edges()


def test_ldag_idx_map_topological_order():
    """Test that ldag_idx_map is sorted in the topological order of the line
    DAG."""
    # Create a simple DAG
    DAG = nx.DiGraph()
    DAG.add_edges_from([(0, 2), (1, 2), (0, 1), (0, 3), (3, 2)])

    # Create PreparedGraph
    pg = PreparedGraph.create(DAG)

    # Get the topological sort of the line DAG
    ldag_topo_sort = list(nx.topological_sort(pg.ldag))

    # Check that the indices in ldag_idx_map correspond to the topological order
    for i, edge in enumerate(ldag_topo_sort):
        assert pg.ldag_idx_map[edge] == i, f"Edge {edge} should have index {i} but has {pg.ldag_idx_map[edge]}"

    # Verify that edges earlier in the topological sort have lower indices
    for u, v in pg.ldag.edges():
        assert pg.ldag_idx_map[u] < pg.ldag_idx_map[v], f"Edge {u} should have lower index than {v}"


def test_prepared_graph_with_cyclic_graph():
    """Test that PreparedGraph.create raises an error for a cyclic directed
    graph."""
    # Create a cyclic directed graph
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Cycle 0->1->2->0

    # Creating a PreparedGraph should raise an error
    with pytest.raises(NotImplementedError):
        PreparedGraph.create(G)
