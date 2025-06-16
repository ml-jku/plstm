from collections import defaultdict
from typing import Literal
from collections.abc import Iterable
from dataclasses import dataclass
import networkx as nx


def dagify(G: nx.Graph) -> nx.DiGraph:
    DAG = nx.DiGraph()
    node_idx_map = {n: idx for idx, n in enumerate(G.nodes())}
    DAG.add_nodes_from(G.nodes())

    for u, v in G.edges():
        if node_idx_map[u] < node_idx_map[v]:
            DAG.add_edge(u, v)
        else:
            DAG.add_edge(v, u)
    return DAG


def revert_dag(dag: nx.DiGraph) -> nx.DiGraph:
    revDAG = nx.DiGraph()
    revDAG.add_nodes_from(reversed(list(dag.nodes())))

    for u, v in reversed(list(dag.edges())):
        revDAG.add_edge(v, u)

    return revDAG


def prune_dag_to_multitree(dag: nx.DiGraph):
    assert nx.is_directed_acyclic_graph(dag), "Input must be a DAG."

    topo_order = list(nx.topological_sort(dag))
    multitree = nx.DiGraph()
    multitree.add_nodes_from(dag.nodes)

    # For fast access:
    # ancestor_map[b] = set(a)  means a -> b has been visited (a is an ancestor of b)
    # descendant_map[a] = set(b) means a -> b has been visited (b is a descendant of a)
    ancestor_map = defaultdict(set)
    descendant_map = defaultdict(set)

    for current in topo_order:
        predecessors = list(dag.predecessors(current))

        # Sort predecessors based on topological order
        predecessors.sort(key=topo_order.index)

        for pred in predecessors:
            prior_ancestors = ancestor_map[pred]
            if any(current in descendant_map[ancestor] for ancestor in prior_ancestors):
                continue  # Skip edge

            # Otherwise, add the edge
            multitree.add_edge(pred, current)
            ancestor_map[current].add(pred)
            descendant_map[pred].add(current)
            for ancestor in prior_ancestors:
                ancestor_map[current].add(ancestor)
                descendant_map[ancestor].add(current)

    return multitree


def check_if_multitree(multitree: nx.DiGraph) -> bool:
    if not nx.is_directed_acyclic_graph(multitree):
        return False

    for node1 in multitree.nodes:
        for node2 in multitree.nodes:
            if len(list(nx.all_simple_paths(multitree, node1, node2))) > 1:
                return False
    return True


@dataclass
class PreparedGraph:
    node_idx_map: dict[int, int]
    dag_edge_orig_idx_map: dict[tuple[int, int], int]
    idx_node_map: dict[int, int]
    orig_idx_dag_edge_map: dict[int, tuple[int, int]]

    dag: nx.DiGraph
    ldag: nx.DiGraph

    ldag_sorted: Iterable[tuple[int, int]]
    ldag_idx_map: dict[tuple[int, int], int]
    ldag_edge_idx_map: dict[tuple[tuple[int, int], tuple[int, int]], int]
    incidence_forward_edgemap: dict[tuple[int, int], int]
    incidence_backward_edgemap: dict[tuple[int, int], int]

    incoming_edge_nums: dict[int, int]
    outgoing_edge_nums: dict[int, int]

    topo_edge_incoming_nodeidx: list[int]
    topo_edge_outgoing_nodeidx: list[int]
    topo_edge_idx: list[int]

    ldag_edge_incoming_edgeidx: list[int]
    ldag_edge_outgoing_edgeidx: list[int]
    ldag_edge_nodeidx: list[int]

    @staticmethod
    def reverse(graph: "PreparedGraph") -> "PreparedGraph":
        dag = revert_dag(graph.dag)
        return PreparedGraph.create(dag)

    # def number_of_edges(self):
    #    return self.dag.number_of_edges()

    @staticmethod
    def create(graph: nx.Graph | nx.DiGraph, mode: Literal["P", "D"] = "D") -> "PreparedGraph":
        node_idx_map = {node: idx for idx, node in enumerate(graph.nodes)}
        idx_node_map = {idx: node for idx, node in enumerate(graph.nodes)}

        edge_idx_map = {edge: idx for idx, edge in enumerate(graph.edges)}
        if not isinstance(graph, nx.DiGraph) and isinstance(graph, nx.Graph):
            dag = dagify(graph)
        elif isinstance(graph, nx.DiGraph):
            if nx.is_directed_acyclic_graph(graph):
                dag = dagify(graph)
            else:
                raise NotImplementedError

        dag_edge_orig_idx_map = {
            dedge: edge_idx_map[dedge] if dedge in edge_idx_map else edge_idx_map[dedge[1], dedge[0]]
            for dedge in dag.edges
        }
        orig_idx_dag_edge_map = {idx: dedge for dedge, idx in dag_edge_orig_idx_map.items()}
        edge_idx_map = None  # this should not be used anymore
        ldag = nx.line_graph(dag)

        if mode == "D":
            ldag = prune_dag_to_multitree(ldag)
        elif mode == "P":
            pass
        else:
            raise NotImplementedError

        ldag_sorted = list(nx.topological_sort(ldag))

        ldag_idx_map = {edge: idx for idx, edge in enumerate(ldag_sorted)}
        ldag_edge_idx_map = {ledge: idx for idx, ledge in enumerate(ldag.edges)}

        incidence_forward_edgemap = {}
        incidence_backward_edgemap = {}
        incoming_edge_nums = {}
        outgoing_edge_nums = {}
        for node in dag.nodes:
            pred = list(dag.predecessors(node))
            for idx, pn in enumerate(pred):
                incidence_backward_edgemap[(pn, node)] = idx
            incoming_edge_nums[node] = len(pred)

        for node in dag.nodes:
            succ = list(dag.successors(node))
            for idx, sn in enumerate(succ):
                incidence_forward_edgemap[(node, sn)] = idx
            outgoing_edge_nums[node] = len(succ)

        topo_edge_incoming_nodeidx = [
            node_idx_map[idx] for idx, _ in ldag_sorted
        ]  # inc. nodeidx of edges topologically sorted
        topo_edge_outgoing_nodeidx = [
            node_idx_map[idx] for _, idx in ldag_sorted
        ]  # outg. nodeidx of edges topologically sorted

        topo_edge_idx = [dag_edge_orig_idx_map[edge] for edge in ldag_sorted]

        ldag_edge_incoming_edgeidx = [dag_edge_orig_idx_map[ldag_edge] for ldag_edge, _ in ldag.edges]
        ldag_edge_outgoing_edgeidx = [dag_edge_orig_idx_map[ldag_edge] for _, ldag_edge in ldag.edges]
        ldag_edge_nodeidx = [node_idx_map[ldag_edge[0]] for ldag_edge, _ in ldag.edges]

        return PreparedGraph(
            node_idx_map=node_idx_map,
            dag_edge_orig_idx_map=dag_edge_orig_idx_map,
            idx_node_map=idx_node_map,
            orig_idx_dag_edge_map=orig_idx_dag_edge_map,
            dag=dag,
            ldag=ldag,
            ldag_sorted=ldag_sorted,
            ldag_idx_map=ldag_idx_map,
            ldag_edge_idx_map=ldag_edge_idx_map,
            incidence_forward_edgemap=incidence_forward_edgemap,
            incidence_backward_edgemap=incidence_backward_edgemap,
            incoming_edge_nums=incoming_edge_nums,
            outgoing_edge_nums=outgoing_edge_nums,
            topo_edge_incoming_nodeidx=topo_edge_incoming_nodeidx,
            topo_edge_outgoing_nodeidx=topo_edge_outgoing_nodeidx,
            topo_edge_idx=topo_edge_idx,
            ldag_edge_incoming_edgeidx=ldag_edge_incoming_edgeidx,
            ldag_edge_outgoing_edgeidx=ldag_edge_outgoing_edgeidx,
            ldag_edge_nodeidx=ldag_edge_nodeidx,
        )
