from dataclasses import dataclass
from typing import Literal
from .interfaces import ResidualModuleConfig
from .blocks import PreUpProjectionBlockConfig, PostUpProjectionBlockConfig
from .plstm_graph_layer import pLSTMGraphLayerConfig, pLSTMGraphEdgeLayerConfig
from ..util import assert_check_literals
from .dtype import DTYPES


@dataclass
class pLSTMGraphBlockConfig(ResidualModuleConfig):
    input_dim: int = 32
    num_heads: int = 8
    block_type: Literal["pre_up", "post_up"] = "post_up"
    block_mode: Literal["PD", "DP"] = "PD"
    # ideally the type annotation should be sth like Block.interfaces, maybe it is necessary to reflect
    # the common structure of interfaces with dummy modules without backend
    block0: PreUpProjectionBlockConfig | PostUpProjectionBlockConfig | None = None
    block1: PreUpProjectionBlockConfig | PostUpProjectionBlockConfig | None = None

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"
    max_edges: int = -1

    def __post_init__(self):
        assert_check_literals(self)
        assert self.input_dim > 0
        assert self.num_heads > 0
        assert self.max_edges > 0
        if self.block0 is None:
            self.block0 = PostUpProjectionBlockConfig(
                input_dim=self.input_dim,
                interaction_module_name="pLSTMGraph",
                interaction_module=pLSTMGraphLayerConfig(
                    mode=self.block_mode[0],
                    input_dim=self.input_dim,
                    num_heads=self.num_heads,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    max_edges=self.max_edges,
                ),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                gated=False,
            )

        if self.block1 is None:
            self.block1 = PostUpProjectionBlockConfig(
                input_dim=self.input_dim,
                interaction_module_name="pLSTMGraph",
                interaction_module=pLSTMGraphLayerConfig(
                    mode=self.block_mode[1],
                    input_dim=self.input_dim,
                    num_heads=self.num_heads,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    max_edges=self.max_edges,
                ),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                gated=False,
            )
        assert isinstance(self.block0.interaction_module, pLSTMGraphLayerConfig)
        assert isinstance(self.block1.interaction_module, pLSTMGraphLayerConfig)


@dataclass
class pLSTMGraphEdgeBlockConfig(ResidualModuleConfig):
    input_dim: int = 32
    num_heads: int = 8
    block_type: Literal["pre_up", "post_up"] = "post_up"
    block_mode: Literal["PD", "DP"] = "PD"
    # ideally the type annotation should be sth like Block.interfaces, maybe it is necessary to reflect
    # the common structure of interfaces with dummy modules without backend
    block0: PreUpProjectionBlockConfig | PostUpProjectionBlockConfig | None = None
    block1: PreUpProjectionBlockConfig | PostUpProjectionBlockConfig | None = None

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        assert_check_literals(self)
        assert self.input_dim > 0
        assert self.num_heads > 0
        if self.block0 is None:
            self.block0 = PostUpProjectionBlockConfig(
                input_dim=self.input_dim,
                interaction_module_name="pLSTMGraphEdge",
                interaction_module=pLSTMGraphEdgeLayerConfig(
                    mode=self.block_mode[0],
                    input_dim=self.input_dim,
                    edge_input_dim=self.input_dim,
                    num_heads=self.num_heads,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                gated=False,
            )

        if self.block1 is None:
            self.block1 = PostUpProjectionBlockConfig(
                input_dim=self.input_dim,
                interaction_module_name="pLSTMGraphEdge",
                interaction_module=pLSTMGraphEdgeLayerConfig(
                    mode=self.block_mode[1],
                    input_dim=self.input_dim,
                    edge_input_dim=self.input_dim,
                    num_heads=self.num_heads,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                ),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                gated=False,
            )
        assert isinstance(self.block0.interaction_module, pLSTMGraphEdgeLayerConfig)
        assert isinstance(self.block1.interaction_module, pLSTMGraphEdgeLayerConfig)
