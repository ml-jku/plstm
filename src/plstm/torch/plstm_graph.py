import torch
import networkx as nx


def plstm_graph_nodewise(
    query: torch.Tensor,  # [H, N, K] queries sorted by orig. graph node idx
    key: torch.Tensor,  # [H, N, K] keys sorted by orig. graph node idx
    value: torch.Tensor,  # [H, N, V] values sorted by orig. graph node idx
    source: torch.Tensor,  # [H, N, E] source sorted by orig. graph node idx
    transition: torch.Tensor,  # [H, N, E, E]  transition sorted by orig. graph node idx
    mark: torch.Tensor,  # [H, N, E] mark sorted by orig. graph node idx
    direct: torch.Tensor,  # [H, N]  direct sorted by orig. graph node idx
    incidence_forward_edgemap: dict[tuple[int, int], int],  # mapping edge to idx in node-out-edge list
    incidence_backward_edgemap: dict[tuple[int, int], int],  # mapping edge to idx in node-in-edge list
    ldag: nx.DiGraph,  # line graph (DAG)
    ldag_idx_map: dict[
        tuple[int, int], int
    ],  # line graph edge map to topologically sorted idx (assuming an ordered dict)
    recompute_cell_states: bool = True,
):
    """Computes the pLSTM on a graph with line graph ldag and ldag_idx_map the
    topologically ordered dictionary of edge to edge index. The
    incidence_forward_edgemap and incidence_backward_edgemap contain the
    mapping from edge to idx for the respective outgoing or incoming node of
    the edge. The source, transition and mark connect node-to-edges, edge-to-
    edge and edge-to-node. The direct is the direct connection for one node.

    Note that there is no "batch_size" as batching in graphs happen via multiple
    connected components to enable flexibility. Here, this means that multiple graphs
    cannot be processed in parallel.

    Note that this code does not limit the number of (in/out) edges per node,
    however the tensor shapes have to enable (E is the maximal in/out-degree)
    the real maximal edge number. The (out/in)-edges for a node are indexed in
    incidence_forward_edgemap and incidence_backward_edgemap.
    This could be made more efficient at the cost of potentially sub-optimal pre-
    computation of source, transition and mark (i.e. source and mark would be
    [H, E] and transition [H, L] with L the number of line edges).

    Args:
        query      (torch.Tensor): Query matrix         [H, N, K]
        key        (torch.Tensor): Key matrix           [H, N, K]
        value      (torch.Tensor): Value matrix         [H, N, V]
        source     (torch.Tensor): Source array         [H, N, E]
        transition (torch.Tensor): Transition array     [H, N, E, E]
        mark       (torch.Tensor): Mark array           [H, N, E]
        direct     (torch.Tensor): Direct               [H, N]
    """

    class pLSTMGraph(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            query,
            key,
            value,
            source,
            transition,
            mark,
            direct,
            incidence_forward_edgemap,
            incidence_backward_edgemap,
            ldag,
            ldag_idx_map,
        ):
            num_heads, num_nodes, qk_head_dim, v_head_dim = *query.shape, value.shape[-1]
            num_edges = ldag.number_of_nodes()
            cell_states = query.new_zeros([num_heads, num_edges, qk_head_dim, v_head_dim])
            outputs = query.new_zeros([num_heads, num_nodes, v_head_dim])

            for idx_edge, edge in enumerate(ldag_idx_map):
                in_node = edge[0]
                out_node = edge[1]
                for pred_edge in ldag.predecessors(edge):
                    cell_states[:, idx_edge] += (
                        transition[
                            :,
                            in_node,
                            incidence_forward_edgemap[pred_edge],
                            incidence_backward_edgemap[edge],
                            None,
                            None,
                        ]
                        * cell_states[:, ldag_idx_map[pred_edge]]
                    )
                cell_states[:, idx_edge] += source[
                    :, in_node, incidence_backward_edgemap[edge], None, None
                ] * torch.einsum("hk,hv->hkv", key[:, in_node], value[:, in_node])
                outputs[:, out_node] += mark[:, out_node, incidence_forward_edgemap[edge], None] * torch.einsum(
                    "hk,hkv->hv", query[:, out_node], cell_states[:, idx_edge]
                )

            outputs += direct[:, :, None] * torch.sum(key * query, dim=-1, keepdim=True) * value

            ctx.save_for_backward(
                query, key, value, source, transition, mark, direct, cell_states if not recompute_cell_states else None
            )
            ctx.ldag = ldag
            ctx.ldag_idx_map = ldag_idx_map
            ctx.incidence_forward_edgemap = incidence_forward_edgemap
            ctx.incidence_backward_edgemap = incidence_backward_edgemap

            return outputs

        @staticmethod
        def backward(
            ctx, doutputs
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            None,
            None,
            None,
            None,
        ]:
            query, key, value, source, transition, mark, direct, cell_states = ctx.saved_tensors
            num_edges = ldag.number_of_nodes()
            num_heads, _, qk_head_dim, v_head_dim = *query.shape, value.shape[-1]

            dquery = torch.zeros_like(query)
            dkey = torch.zeros_like(key)
            dvalue = torch.zeros_like(value)
            dsource = torch.zeros_like(source)
            dtransition = torch.zeros_like(transition)
            dmark = torch.zeros_like(mark)
            ddirect = torch.zeros_like(direct)

            if not cell_states:
                cell_states = query.new_zeros([num_heads, num_edges, qk_head_dim, v_head_dim])
                for idx_edge, edge in enumerate(ldag_idx_map):
                    in_node = edge[0]
                    out_node = edge[1]
                    for pred_edge in ldag.predecessors(edge):
                        cell_states[:, idx_edge] += (
                            transition[
                                :,
                                in_node,
                                ctx.incidence_forward_edgemap[pred_edge],
                                ctx.incidence_backward_edgemap[edge],
                                None,
                                None,
                            ]
                            * cell_states[:, ldag_idx_map[pred_edge]]
                        )
                    cell_states[:, idx_edge] += source[
                        :, in_node, ctx.incidence_backward_edgemap[edge], None, None
                    ] * torch.einsum("ha,hb->hab", key[:, in_node], value[:, in_node])

            dcell_states = torch.zeros_like(cell_states)

            for revidx_edge, edge in enumerate(reversed(ctx.ldag_idx_map)):
                idx_edge = num_edges - revidx_edge - 1
                in_node = edge[0]
                out_node = edge[1]
                for succ_edge in ldag.successors(edge):
                    dcell_states[:, idx_edge] += (
                        transition[
                            :,
                            out_node,
                            ctx.incidence_forward_edgemap[edge],
                            ctx.incidence_backward_edgemap[succ_edge],
                            None,
                            None,
                        ]
                        * dcell_states[:, ldag_idx_map[succ_edge]]
                    )
                    dtransition[
                        :, out_node, ctx.incidence_forward_edgemap[edge], incidence_backward_edgemap[succ_edge]
                    ] += torch.einsum("hkv,hkv->h", cell_states[:, idx_edge], dcell_states[:, ldag_idx_map[succ_edge]])
                dcell_states[:, idx_edge] += mark[
                    :, out_node, ctx.incidence_forward_edgemap[edge], None, None
                ] * torch.einsum("hk,hv->hkv", query[:, out_node], doutputs[:, out_node])

                dquery[:, out_node] += mark[:, out_node, ctx.incidence_forward_edgemap[edge], None] * torch.einsum(
                    "hkv,hv->hk", cell_states[:, idx_edge], doutputs[:, out_node]
                )
                dmark[:, out_node, ctx.incidence_forward_edgemap[edge]] += torch.einsum(
                    "hk,hkv,hv->h", query[:, out_node], cell_states[:, idx_edge], doutputs[:, out_node]
                )

                dkey[:, in_node] += source[:, in_node, ctx.incidence_backward_edgemap[edge], None] * torch.einsum(
                    "hkv,hv->hk", dcell_states[:, idx_edge], value[:, in_node]
                )
                dvalue[:, in_node] += source[:, in_node, ctx.incidence_backward_edgemap[edge], None] * torch.einsum(
                    "hkv,hk->hv", dcell_states[:, idx_edge], key[:, in_node]
                )
                dsource[:, in_node, ctx.incidence_backward_edgemap[edge]] += torch.einsum(
                    "hkv,hk,hv->h", dcell_states[:, idx_edge], key[:, in_node], value[:, in_node]
                )

            dquery += torch.einsum("hn,hnk,hnv,hnv->hnk", direct, key, value, doutputs)
            dkey += torch.einsum("hn,hnk,hnv,hnv->hnk", direct, query, value, doutputs)
            dvalue += torch.einsum("hn,hnk,hnk,hnv->hnv", direct, query, key, doutputs)
            ddirect += torch.einsum("hnk,hnk,hnv,hnv->hn", query, key, value, doutputs)

            return dquery, dkey, dvalue, dsource, dtransition, dmark, ddirect, None, None, None, None

    return pLSTMGraph.apply(
        query,
        key,
        value,
        source,
        transition,
        mark,
        direct,
        incidence_forward_edgemap,
        incidence_backward_edgemap,
        ldag,
        ldag_idx_map,
    )


def plstm_graph(
    query: torch.Tensor,  # [H, N, K] # queries sorted by orig node idx
    key: torch.Tensor,  # [H, N, K] # keys sorted by orig node idx
    value: torch.Tensor,  # [H, N, V] # values sorted by orig node idx
    source: torch.Tensor,  # [H, E]  # source sorted by ldag idx (edge) in topo order of edges
    transition: torch.Tensor,  # [H, L] # source sorted by ldag edge idx (edge2edge)
    mark: torch.Tensor,  # [H, E] # mark sorted by ldag idx (edge) in topo order of edges
    direct: torch.Tensor,  # [H, N] # direct sorted by orig node idx
    ldag: nx.DiGraph,  # line graph (DAG)
    ldag_idx_map: dict[
        tuple[int, int], int
    ],  # ldag_idx_map of dag edge (ldag node) to idx in topo order (assuming an ordered dict)
    ldag_edge_idx_map: dict[
        tuple[tuple[int, int], tuple[int, int]], int
    ],  # ldag_edge_idx_map of ldag edge to transition index
    recompute_cell_states: bool = True,
):
    """Computes the pLSTM on a graph with line graph ldag and ldag_idx_map the
    topologically ordered dictionary of edge to edge index. The source,
    transition and mark connect node-to-edges, edge-to-edge and edge-to-node.
    The direct is the direct connection for one node.

    Note that there is no "batch_size" as batching in graphs happen via multiple
    connected components to enable flexibility. Here, this means that multiple graphs
    cannot be processed in parallel.

    This implementation is the most memory efficient variant, storing source and mark
    for exactly every edge and transition for every line graph edge.
    This means the respective values have to be pre-computed in potentially sub-optimal fashion.

    Args:
        query              (torch.Tensor): Query matrix         [H, N, K]
        key                (torch.Tensor): Key matrix           [H, N, K]
        value              (torch.Tensor): Value matrix         [H, N, V]
        source             (torch.Tensor): Source array         [H, E]
        transition         (torch.Tensor): Transition array     [H, L]
        mark               (torch.Tensor): Mark array           [H, E]
        direct             (torch.Tensor): Direct               [H, N]
        ldag               (nx.DiGraph):   Line Graph
        ldag_idx_map       (dict):         Dict mapping edges (node tuple) to edge index
        ldag_edge_idx_map  (dict):         Dict mapping linegraph edges (edge-edge-tuple)
                                           an index (transition entry)

    Returns:
       (torch.Tensor): Output of the pLSTM layer of the shape of the value matrix.
    """

    class pLSTMGraph(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            query,
            key,
            value,  # values sorted by orig node idx
            source,
            transition,
            mark,
            direct,
            ldag,
            ldag_idx_map,
            ldag_edge_idx_map,
        ):
            num_heads, num_nodes, qk_head_dim, v_head_dim = *query.shape, value.shape[-1]
            num_edges = ldag.number_of_nodes()
            cell_states = query.new_zeros([num_heads, num_edges, qk_head_dim, v_head_dim])
            outputs = query.new_zeros([num_heads, num_nodes, v_head_dim])

            for idx_edge, edge in enumerate(ldag_idx_map):
                in_node = edge[0]
                out_node = edge[1]
                for pred_edge in ldag.predecessors(edge):
                    cell_states[:, idx_edge] += (
                        transition[
                            :,
                            ldag_edge_idx_map[(pred_edge, edge)],
                            None,
                            None,
                        ]
                        * cell_states[:, ldag_idx_map[pred_edge]]
                    )
                cell_states[:, idx_edge] += source[:, ldag_idx_map[edge], None, None] * torch.einsum(
                    "hk,hv->hkv", key[:, in_node], value[:, in_node]
                )
                outputs[:, out_node] += mark[:, ldag_idx_map[edge], None] * torch.einsum(
                    "hk,hkv->hv", query[:, out_node], cell_states[:, idx_edge]
                )

            outputs += direct[:, :, None] * torch.sum(key * query, dim=-1, keepdim=True) * value

            ctx.save_for_backward(
                query, key, value, source, transition, mark, direct, cell_states if not recompute_cell_states else None
            )
            ctx.ldag = ldag
            ctx.ldag_idx_map = ldag_idx_map

            return outputs

        @staticmethod
        def backward(
            ctx, doutputs
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            None,
            None,
            None,
        ]:
            query, key, value, source, transition, mark, direct, cell_states = ctx.saved_tensors
            num_edges = ldag.number_of_nodes()
            num_heads, _, qk_head_dim, v_head_dim = *query.shape, value.shape[-1]

            dquery = torch.zeros_like(query)
            dkey = torch.zeros_like(key)
            dvalue = torch.zeros_like(value)
            dsource = torch.zeros_like(source)
            dtransition = torch.zeros_like(transition)
            dmark = torch.zeros_like(mark)
            ddirect = torch.zeros_like(direct)

            if not cell_states:
                cell_states = query.new_zeros([num_heads, num_edges, qk_head_dim, v_head_dim])
                for idx_edge, edge in enumerate(ldag_idx_map):
                    in_node = edge[0]
                    out_node = edge[1]
                    for pred_edge in ldag.predecessors(edge):
                        cell_states[:, idx_edge] += (
                            transition[
                                :,
                                ldag_edge_idx_map[(pred_edge, edge)],
                                None,
                                None,
                            ]
                            * cell_states[:, ldag_idx_map[pred_edge]]
                        )
                    cell_states[:, idx_edge] += source[:, ldag_idx_map[edge], None, None] * torch.einsum(
                        "ha,hb->hab", key[:, in_node], value[:, in_node]
                    )

            dcell_states = torch.zeros_like(cell_states)

            for revidx_edge, edge in enumerate(reversed(ctx.ldag_idx_map)):
                idx_edge = num_edges - revidx_edge - 1
                in_node = edge[0]
                out_node = edge[1]
                for succ_edge in ldag.successors(edge):
                    dcell_states[:, idx_edge] += (
                        transition[
                            :,
                            ldag_edge_idx_map[(edge, succ_edge)],
                            None,
                            None,
                        ]
                        * dcell_states[:, ldag_idx_map[succ_edge]]
                    )
                    dtransition[
                        :,
                        ldag_edge_idx_map[(edge, succ_edge)],
                    ] += torch.einsum(
                        "hkv,hkv->h",
                        cell_states[:, idx_edge],
                        dcell_states[
                            :,
                            ldag_idx_map[succ_edge],
                        ],
                    )
                dcell_states[:, idx_edge] += mark[:, ldag_idx_map[edge], None, None] * torch.einsum(
                    "hk,hv->hkv", query[:, out_node], doutputs[:, out_node]
                )

                dquery[:, out_node] += mark[:, ldag_idx_map[edge], None] * torch.einsum(
                    "hkv,hv->hk", cell_states[:, idx_edge], doutputs[:, out_node]
                )
                dmark[:, ldag_idx_map[edge]] += torch.einsum(
                    "hk,hkv,hv->h", query[:, out_node], cell_states[:, idx_edge], doutputs[:, out_node]
                )

                dkey[:, in_node] += source[:, ldag_idx_map[edge], None] * torch.einsum(
                    "hkv,hv->hk", dcell_states[:, idx_edge], value[:, in_node]
                )
                dvalue[:, in_node] += source[:, ldag_idx_map[edge], None] * torch.einsum(
                    "hkv,hk->hv", dcell_states[:, idx_edge], key[:, in_node]
                )
                dsource[:, ldag_idx_map[edge]] += torch.einsum(
                    "hkv,hk,hv->h", dcell_states[:, idx_edge], key[:, in_node], value[:, in_node]
                )

            dquery += torch.einsum("hn,hnk,hnv,hnv->hnk", direct, key, value, doutputs)
            dkey += torch.einsum("hn,hnk,hnv,hnv->hnk", direct, query, value, doutputs)
            dvalue += torch.einsum("hn,hnk,hnk,hnv->hnv", direct, query, key, doutputs)
            ddirect += torch.einsum("hnk,hnk,hnv,hnv->hn", query, key, value, doutputs)

            return dquery, dkey, dvalue, dsource, dtransition, dmark, ddirect, None, None, None

    return pLSTMGraph.apply(
        query,
        key,
        value,
        source,
        transition,
        mark,
        direct,
        ldag,
        ldag_idx_map,
        ldag_edge_idx_map,
    )


if __name__ == "__main__":
    import random
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.split(__file__)[0]) + "/../")
    from graph import dagify

    num_nodes = 5
    extra_edge_prob = 0.2

    G = nx.random_labeled_tree(num_nodes)

    # Step 2: Add extra random edges while ensuring the graph remains simple
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if not G.has_edge(u, v) and random.random() < extra_edge_prob:
                G.add_edge(u, v)

    dag = dagify(G)
    ldag = nx.line_graph(dag)

    H, N, K, V = 2, num_nodes, 3, 3
    E, L = ldag.number_of_nodes(), ldag.number_of_edges()

    edge_list = {edge: edge_idx for edge_idx, edge in enumerate(nx.topological_sort(ldag))}
    edge_comb_list = {edgepair: edgepair_idx for edgepair_idx, edgepair in enumerate(ldag.edges)}

    q = torch.randn([H, N, K], dtype=torch.double, requires_grad=True)
    k = torch.randn([H, N, K], dtype=torch.double, requires_grad=True)
    v = torch.randn([H, N, V], dtype=torch.double, requires_grad=True)
    s = torch.randn([H, E], dtype=torch.double, requires_grad=True)
    t = torch.randn([H, L], dtype=torch.double, requires_grad=True) / 10.0
    m = torch.randn([H, E], dtype=torch.double, requires_grad=True)
    d = torch.randn([H, N], dtype=torch.double, requires_grad=True)

    res = plstm_graph(q, k, v, s, t, m, d, ldag, edge_list, edge_comb_list)
    res.sum().backward()

    try:
        torch.autograd.gradcheck(plstm_graph, (q, k, v, s, t, m, d, ldag, edge_list, edge_comb_list))
    except torch.autograd.gradcheck.GradcheckError as e:
        print(e)
