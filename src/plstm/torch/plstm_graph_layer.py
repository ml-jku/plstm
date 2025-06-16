import torch
import networkx as nx
from torch import nn
from ..graph import PreparedGraph
from .plstm_graph import plstm_graph_nodewise, plstm_graph
from .initialization import InitInterface
from .norm import NormInterface
from .interfaces import ResidualModule
from compoconf import register
from ..config.plstm_graph_layer import pLSTMGraphLayerConfig, pLSTMGraphEdgeLayerConfig
from .dtype import str_dtype_to_torch


@register
class pLSTMGraphLayer(ResidualModule):
    config: pLSTMGraphLayerConfig

    def __init__(self, config: pLSTMGraphLayerConfig):
        super().__init__(config)
        self.config = config

        # Create parameters with param_dtype
        param_dtype = str_dtype_to_torch(config.param_dtype)

        self.query = nn.Linear(
            config.input_dim, config.num_heads * config.qk_head_dim, bias=config.bias, dtype=param_dtype
        )
        self.key = nn.Linear(
            config.input_dim, config.num_heads * config.qk_head_dim, bias=config.bias, dtype=param_dtype
        )
        self.value_ = nn.Linear(
            config.input_dim, config.num_heads * config.hv_head_dim, bias=config.bias, dtype=param_dtype
        )

        self.source = nn.Linear(config.input_dim, config.num_heads * config.max_edges, dtype=param_dtype)
        self.transition = nn.Linear(
            config.input_dim, config.num_heads * config.max_edges * config.max_edges, dtype=param_dtype
        )
        self.mark = nn.Linear(config.input_dim, config.num_heads * config.max_edges, dtype=param_dtype)
        self.direct = nn.Linear(config.input_dim, config.num_heads, dtype=param_dtype)

        self.mhnorm = config.mhnorm.instantiate(NormInterface)

        if self.config.out:
            self.out = nn.Linear(
                config.num_heads * config.hv_head_dim, config.input_dim, bias=config.bias, dtype=param_dtype
            )

        self.reset_parameters()

    def _smd_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def _pmode_transition_normalization(self, t: torch.Tensor, graph: PreparedGraph):
        H, N, E, _ = t.shape
        edge_nums = torch.tensor([v for v in graph.incoming_edge_nums.values()], dtype=torch.int, device=t.device)
        edge_mask = edge_nums[:, None] - torch.arange(E, device=t.device)[None, :] > 0.5
        t = edge_mask[None, :, :, None] * t
        alpha = 0.5
        return t / (alpha + torch.sum(torch.abs(t), dim=3, keepdim=True))

    def reset_parameters(self):
        if self.config.bias:
            self.config.qkvo_bias_init.instantiate(InitInterface)(self.query.bias)
            self.config.qkvo_bias_init.instantiate(InitInterface)(self.key.bias)
            self.config.qkvo_bias_init.instantiate(InitInterface)(self.value_.bias)
            if self.config.out:
                self.config.qkvo_bias_init.instantiate(InitInterface)(self.out.bias)
        self.config.qkvo_weight_init.instantiate(InitInterface)(self.query.weight)
        self.config.qkvo_weight_init.instantiate(InitInterface)(self.key.weight)
        self.config.qkvo_weight_init.instantiate(InitInterface)(self.value_.weight)
        if self.config.out:
            self.config.qkvo_weight_init.instantiate(InitInterface)(self.out.weight)

        self.config.source_bias_init.instantiate(InitInterface)(self.source.bias)
        self.config.source_weight_init.instantiate(InitInterface)(self.source.weight)
        self.config.mark_bias_init.instantiate(InitInterface)(self.mark.bias)
        self.config.mark_weight_init.instantiate(InitInterface)(self.mark.weight)
        self.config.direct_bias_init.instantiate(InitInterface)(self.direct.bias)
        self.config.direct_weight_init.instantiate(InitInterface)(self.direct.weight)

        self.config.transition_weight_init.instantiate(InitInterface)(self.transition.weight)
        bias_inited = torch.zeros_like(self.transition.bias).reshape(
            self.config.num_heads, self.config.max_edges, self.config.max_edges
        )
        self.config.transition_bias_init.instantiate(InitInterface)(bias_inited)
        with torch.no_grad():
            self.transition.bias.copy_(bias_inited.reshape(self.transition.bias.shape))

        self.mhnorm.reset_parameters()

    def forward(self, node_features: torch.Tensor, graph: nx.Graph | nx.DiGraph | PreparedGraph) -> torch.Tensor:
        N, H, E = node_features.shape[0], self.config.num_heads, self.config.max_edges
        if not isinstance(graph, PreparedGraph):
            graph = PreparedGraph.create(graph, mode=self.config.mode)

        # Cast input to computation dtype
        dtype = str_dtype_to_torch(self.config.dtype)

        with torch.autocast(device_type=node_features.device.type, dtype=dtype):
            q, k, v = self.query(node_features), self.key(node_features), self.value_(node_features)
            s, t, m, d = (
                self._smd_activation(self.source(node_features)),
                self.transition(node_features),
                self._smd_activation(self.mark(node_features)),
                self._smd_activation(self.direct(node_features)),
            )

            q, k, v = (
                q.reshape(N, H, -1).transpose(0, 1),
                k.reshape(N, H, -1).transpose(0, 1),
                v.reshape(N, H, -1).transpose(0, 1),
            )
            s, m, d = (
                s.reshape(N, H, E).transpose(0, 1),
                m.reshape(N, H, E).transpose(0, 1),
                d.reshape(N, H).transpose(0, 1),
            )
            t = t.reshape(N, H, E, E).transpose(0, 1)

            if self.config.mode == "P":
                t = self._pmode_transition_normalization(t, graph=graph)
            elif self.config.mode == "D":
                t = torch.tanh(t)
            else:
                raise ValueError

            # Ensure all tensors are in computation dtype
            q, k, v = q.to(dtype=dtype), k.to(dtype=dtype), v.to(dtype=dtype)
            s, t, m, d = s.to(dtype=dtype), t.to(dtype=dtype), m.to(dtype=dtype), d.to(dtype=dtype)

            out = plstm_graph_nodewise(
                q,
                k,
                v,
                s,
                t,
                m,
                d,
                graph.incidence_forward_edgemap,
                graph.incidence_backward_edgemap,
                graph.ldag,
                graph.ldag_idx_map,
            )

            # Apply normalization and reshape
            res = self.mhnorm(out).transpose(0, 1).reshape(N, -1)
            if self.config.out:
                res = self.out(res)

        # Return result in the original dtype of node_features
        return res


@register
class pLSTMGraphEdgeLayer(ResidualModule):
    config: pLSTMGraphEdgeLayerConfig

    def __init__(self, config: pLSTMGraphEdgeLayerConfig):
        super().__init__(config)
        self.config = config

        param_dtype = str_dtype_to_torch(config.param_dtype)
        self.query = nn.Linear(config.input_dim, config.num_heads * config.qk_head_dim, dtype=param_dtype)
        self.key = nn.Linear(config.input_dim, config.num_heads * config.qk_head_dim, dtype=param_dtype)
        self.value_ = nn.Linear(config.input_dim, config.num_heads * config.hv_head_dim, dtype=param_dtype)
        self.source = nn.Linear(config.input_dim + config.edge_input_dim, config.num_heads, dtype=param_dtype)
        self.transition = nn.Linear(
            config.input_dim + config.edge_input_dim + config.edge_input_dim, config.num_heads, dtype=param_dtype
        )
        self.mark = nn.Linear(config.input_dim + config.edge_input_dim, config.num_heads, dtype=param_dtype)
        self.direct = nn.Linear(config.input_dim, config.num_heads, dtype=param_dtype)

        self.mhnorm = self.config.mhnorm.instantiate(NormInterface)

        if self.config.out:
            self.out = nn.Linear(config.num_heads * config.hv_head_dim, config.input_dim, dtype=param_dtype)

    def reset_parameters(self):
        if self.config.bias:
            self.config.qkvo_bias_init.instantiate(InitInterface)(self.query.bias)
            self.config.qkvo_bias_init.instantiate(InitInterface)(self.key.bias)
            self.config.qkvo_bias_init.instantiate(InitInterface)(self.value_.bias)
            if self.config.out:
                self.ocnfig.qkvo_bias_init.instantiate(InitInterface)(self.out.bias)
        self.config.qkvo_weight_init.instantiate(InitInterface)(self.query.weight)
        self.config.qkvo_weight_init.instantiate(InitInterface)(self.key.weight)
        self.config.qkvo_weight_init.instantiate(InitInterface)(self.value_.weight)
        if self.config.out:
            self.config.qkvo_weight_init.instantiate(InitInterface)(self.out.weight)

        self.config.source_bias_init.instantiate(InitInterface)(self.source.bias)
        self.config.source_weight_init.instantiate(InitInterface)(self.source.weight)
        self.config.mark_bias_init.instantiate(InitInterface)(self.mark.bias)
        self.config.mark_weight_init.instantiate(InitInterface)(self.mark.weight)
        self.config.direct_bias_init.instantiate(InitInterface)(self.direct.bias)
        self.config.direct_weight_init.instantiate(InitInterface)(self.direct.weight)

        self.config.transition_weight_init.instantiate(InitInterface)(self.transition.weight)

        self.config.transition_bias_init.instantiate(InitInterface)(self.transition.bias)

        self.mhnorm.reset_parameters()

    def _smd_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def _pmode_transition_normalization(self, t: torch.Tensor, graph: PreparedGraph):
        # H, N, E, _ = t.shape
        # edge_nums = torch.tensor([v for v in graph.incoming_edge_nums.values()], dtype=torch.int)
        # edge_mask = edge_nums[:, None] - torch.arange(E)[None, :] > 0.5
        # t = edge_mask[None, :, :, None] * t
        # alpha = 0.5
        # return t / (1 + alpha * torch.sum(torch.abs(t), dim=3, keepdim=True))
        H = t.shape[0]
        outedge_idx = torch.tensor(graph.ldag_edge_outgoing_edgeidx, dtype=torch.long, device=t.device)

        if len(outedge_idx) == 0:
            max_idx = 1
        else:
            max_idx = outedge_idx.max() + 1

        row_norms = t.new_zeros([t.shape[0], max_idx])
        row_norms.scatter_add_(1, outedge_idx[None, :].expand(H, -1), torch.abs(t))
        alpha = 0.5
        row_norm_factor = 1.0 / (alpha + row_norms)
        t = t * torch.take_along_dim(row_norm_factor, outedge_idx[None, :].expand(H, -1), dim=1)
        return t

    def forward(
        self,
        node_features: torch.Tensor,  # [N, I]
        edge_features: torch.Tensor,  # [E, J]
        graph: nx.Graph | nx.DiGraph | PreparedGraph,
    ) -> torch.Tensor:
        N, H, E = node_features.shape[0], self.config.num_heads, graph.dag.number_of_edges()

        if not isinstance(graph, PreparedGraph):
            graph = PreparedGraph.create(graph)

        dtype = str_dtype_to_torch(self.config.dtype)
        with torch.autocast(device_type=node_features.device.type, dtype=dtype):
            q, k, v = self.query(node_features), self.key(node_features), self.value_(node_features)
            q, k, v = (
                q.reshape(N, H, -1).transpose(0, 1),
                k.reshape(N, H, -1).transpose(0, 1),
                v.reshape(N, H, -1).transpose(0, 1),
            )

            source_node_features = node_features[torch.tensor(graph.topo_edge_incoming_nodeidx)]
            mark_node_features = node_features[torch.tensor(graph.topo_edge_outgoing_nodeidx)]

            s = self._smd_activation(
                self.source(torch.cat((source_node_features, edge_features[torch.tensor(graph.topo_edge_idx)]), dim=-1))
                .reshape(E, H)
                .transpose(0, 1)
            )
            m = self._smd_activation(
                self.mark(torch.cat((mark_node_features, edge_features[torch.tensor(graph.topo_edge_idx)]), dim=-1))
                .reshape(E, H)
                .transpose(0, 1)
            )
            d = self._smd_activation(self.direct(node_features).reshape(N, H).transpose(0, 1))

            t = self.transition(
                torch.cat(
                    (
                        edge_features[torch.tensor(graph.ldag_edge_incoming_edgeidx, dtype=torch.long)],
                        edge_features[torch.tensor(graph.ldag_edge_outgoing_edgeidx, dtype=torch.long)],
                        node_features[torch.tensor(graph.ldag_edge_nodeidx, dtype=torch.long)],
                    ),
                    axis=-1,
                )
            ).transpose(0, 1)

            if self.config.mode == "P":
                t = self._pmode_transition_normalization(t, graph=graph)
            elif self.config.mode == "D":
                t = torch.tanh(t)
            else:
                raise ValueError

            out = plstm_graph(q, k, v, s, t, m, d, graph.ldag, graph.ldag_idx_map, graph.ldag_edge_idx_map)

            res = self.mhnorm(out).transpose(0, 1).reshape(N, -1)
            if self.config.out:
                res = self.out(res)
        return res
