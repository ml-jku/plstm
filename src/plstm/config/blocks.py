from dataclasses import dataclass, field
from typing import Literal
from .dtype import DTYPES

from .plstm_1d_layer import pLSTM1DLayerConfig
from .plstm_2d_layer import pLSTM2DLayerConfig
from .plstm_2d_layer_fused import pLSTM2DLayerFusedConfig
from .plstm_graph_layer import pLSTMGraphLayerConfig, pLSTMGraphEdgeLayerConfig
from .norm import NormConfig, RMSNormConfig
from .scale import ScaleLayerConfig
from ..util import assert_check_literals
from .interfaces import ResidualModuleConfig

from .initialization import WeightInitConfig, BiasInitConfig, SmallInitConfig, ZerosInitConfig


InteractionModuleConfig = (
    pLSTM1DLayerConfig
    | pLSTM2DLayerFusedConfig
    | pLSTM2DLayerConfig
    | pLSTMGraphLayerConfig
    | pLSTMGraphEdgeLayerConfig
)

InteractionModuleName = Literal["pLSTM1D", "pLSTM2DFused", "pLSTM2D", "pLSTMGraph", "pLSTMGraphEdge"]

InteractionModuleDict = {
    "pLSTM1D": pLSTM1DLayerConfig,
    "pLSTM2DFused": pLSTM2DLayerFusedConfig,
    "pLSTM2D": pLSTM2DLayerConfig,
    "pLSTMGraph": pLSTMGraphLayerConfig,
    "pLSTMGraphEdge": pLSTMGraphEdgeLayerConfig,
}


@dataclass
class PreUpProjectionBlockConfig(ResidualModuleConfig):
    input_dim: int = -1
    output_dim: int = -1
    upproj_weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)
    downproj_weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)
    upproj_bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)
    downproj_bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)

    bias: bool = True
    skip: bool = True

    projection_scaling: float = 2.0
    projection_round: int = 64
    gated: bool = True
    gating_function: Literal["silu"] = "silu"

    inner_input_dim: int = -1

    interaction_module_name: InteractionModuleName = "pLSTM1D"
    interaction_module: InteractionModuleConfig | None = None

    norm: NormConfig | None = None

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    @property
    def _inner_input_dim(self):
        return (
            int(round((self.projection_scaling * self.input_dim - 1) // self.projection_round) + 1)
            * self.projection_round
        )

    def __post_init__(self):
        assert self.input_dim > 0
        assert_check_literals(self)
        if self.output_dim <= 0:
            self.output_dim = self.input_dim
        if self.inner_input_dim < 0:
            self.inner_input_dim = self._inner_input_dim
        if self.norm is None:
            self.norm = RMSNormConfig(
                input_dim=self.input_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

        if self.interaction_module is None:
            if self.interaction_module_name == "pLSTM1D":
                self.interaction_module = pLSTM1DLayerConfig(
                    input_dim=self.inner_input_dim,
                    num_heads=4,
                    sub_heads=self.inner_input_dim // 4,
                    outprojection=False,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            elif self.interaction_module_name == "pLSTM2D":
                self.interaction_module = pLSTM2DLayerConfig(
                    input_dim=self.inner_input_dim,
                    num_heads=4,
                    sub_heads=self.inner_input_dim // 4,
                    orientation_combinations=[0, 1, 2, 3],
                    mode="D",
                    outprojection=False,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            elif self.interaction_module_name == "pLSTM2DFused":
                self.interaction_module = pLSTM2DLayerConfig(
                    input_dim=self.inner_input_dim,
                    num_heads=4,
                    sub_heads=self.inner_input_dim // 4,
                    outprojection=False,
                    mode="D",
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            elif self.interaction_module_name == "pLSTMGraph":
                self.interaction_module = pLSTMGraphLayerConfig(
                    input_dim=self.input_dim,
                    mode="D",
                    num_heads=self.input_dim // 32,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            elif self.interaction_module_name == "pLSTMGraphEdge":
                self.interaction_module = pLSTMGraphEdgeLayerConfig(
                    input_dim=self.input_dim,
                    edge_input_dim=self.input_dim,
                    mode="D",
                    num_heads=self.input_dim // 32,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
        else:
            assert isinstance(self.interaction_module, InteractionModuleDict[self.interaction_module_name])
        assert self._inner_input_dim == self.inner_input_dim


@dataclass
class PostUpProjectionBlockConfig(ResidualModuleConfig):
    input_dim: int = -1
    output_dim: int = -1

    upproj_weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)
    downproj_weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)
    upproj_bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)
    downproj_bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)

    bias: bool = True
    projection_scaling: float = 4  # 8 / 3
    projection_round: int = 64
    gated: bool = False
    # potentially used as activation function in MLP if not gated
    gating_function: Literal["silu", "gelu"] = "gelu"

    drop_path_rate: float = 0.0

    interaction_module_name: InteractionModuleName = "pLSTM1D"
    interaction_module: InteractionModuleConfig | None = None
    skip: bool = True
    use_scale: bool = False
    scale: ScaleLayerConfig | None = None

    norm: NormConfig | None = None

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    @property
    def inner_input_dim(self):
        return (
            int(round((self.projection_scaling * self.input_dim - 1) // self.projection_round) + 1)
            * self.projection_round
        )

    def __post_init__(self):
        assert_check_literals(self)
        assert self.input_dim > 0
        assert 1.0 >= self.drop_path_rate >= 0.0
        if self.norm is None:
            self.norm = RMSNormConfig(
                input_dim=self.input_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        if self.output_dim <= 0:
            self.output_dim = self.input_dim
        if self.use_scale and self.scale is None:
            self.scale = ScaleLayerConfig(input_dim=self.input_dim, scale=1.0)
        if self.interaction_module is None:
            if self.interaction_module_name == "pLSTM1D":
                self.interaction_module = pLSTM1DLayerConfig(
                    input_dim=self.input_dim,
                    num_heads=4,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            elif self.interaction_module_name == "pLSTM2D":
                self.interaction_module = pLSTM2DLayerConfig(
                    input_dim=self.input_dim,
                    mode="D",
                    num_heads=4,
                    orientation_combinations=[0, 1, 2, 3],
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            elif self.interaction_module_name == "pLSTM2DFused":
                self.interaction_module = pLSTM2DLayerFusedConfig(
                    input_dim=self.input_dim,
                    mode="D",
                    num_heads=4,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            elif self.interaction_module_name == "pLSTMGraph":
                self.interaction_module = pLSTMGraphLayerConfig(
                    input_dim=self.input_dim,
                    mode="D",
                    num_heads=self.input_dim // 32,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
            elif self.interaction_module_name == "pLSTMGraphEdge":
                self.interaction_module = pLSTMGraphEdgeLayerConfig(
                    input_dim=self.input_dim,
                    edge_input_dim=self.input_dim,
                    mode="D",
                    num_heads=self.input_dim // 32,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                )
        else:
            assert isinstance(self.interaction_module, InteractionModuleDict[self.interaction_module_name])
