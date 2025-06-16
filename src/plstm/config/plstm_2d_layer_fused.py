from dataclasses import dataclass, field
from typing import Literal
from .dtype import DTYPES
from ..util import assert_check_literals
from .initialization import BiasInitConfig, WeightInitConfig, ZerosInitConfig, LinspaceInitConfig, SmallInitConfig
from .query_key_value import KeyLayerConfig, QueryLayerConfig, ValueLayerConfig
from .source_mark_layer import (
    SourceLayerConfig,
    MarkLayerConfig,
    DirectLayerConfig,
)
from .longrange_transition_layer import LongRangeTransitionLayerConfig
from .passthrough import PassthroughLayerConfig
from .convolution import Convolution2DLayerConfig
from .norm import MultiHeadRMSNormConfig, MultiHeadLayerNormConfig
from .interfaces import ResidualModuleConfig


@dataclass
class pLSTM2DLayerFusedConfig(ResidualModuleConfig):
    _shortname: str = "pLSTM2DFused"
    class_name: str = "pLSTM2DLayerFused"
    mode: Literal["D", "P"] = "P"
    pmode_orientation_bias_init: BiasInitConfig = field(
        default_factory=lambda: LinspaceInitConfig(low=-2.0, high=2.0, axis=-1)
    )  # headwise_distributed will be handled in implementation
    pmode_orientation_weight_init: WeightInitConfig = field(default_factory=ZerosInitConfig)
    pmode_orientation_scale: float = 0.5
    dmode_chirality: Literal["L", "R"] = "L"

    input_dim: int = -1
    output_dim: int = -1
    num_heads: int = 1

    # state tracking dimensions
    JK: int = 1  # key state-tracking dimension
    JQ: int = 1  # query state-tracking dimension
    JV: int = 1  # value state-tracking dimension
    JT: int = 1  # transition state-tracking dimension (most important)
    JO: int = 1  # output state-tracking dimension

    DK: int = -1  # query / key dimension
    DV: int = -1  # value / output dimension

    source: SourceLayerConfig | None = None
    transition: LongRangeTransitionLayerConfig | None = None
    mark: MarkLayerConfig | None = None
    direct: DirectLayerConfig | None = None
    query: QueryLayerConfig | None = None
    key: KeyLayerConfig | None = None
    value: ValueLayerConfig | None = None
    passthrough: PassthroughLayerConfig | None = None
    convolution: Convolution2DLayerConfig | None = None
    sub_heads: int = -1
    additional_passthrough: bool = False
    additional_convolution: bool = False

    outprojection: bool = True
    outprojection_bias: bool = False
    outprojection_weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)
    outprojection_bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)

    convolution_inputs: str = "STMDQKsO"  # options "STMDQKVstPO", orientation O

    mhnorm: MultiHeadRMSNormConfig | MultiHeadLayerNormConfig | None = None

    levels: int = 6

    tie_query_key: bool = False

    dtype: DTYPES = "bfloat16"
    param_dtype: DTYPES = "float32"

    def __post_init__(self):
        if self.output_dim < 0:
            self.output_dim = self.input_dim

        if self.DV < 0:
            self.DV = self.input_dim // self.JV // self.num_heads
        if self.DK < 0:
            self.DK = self.input_dim // self.JK // self.num_heads
        if self.sub_heads < 0:
            self.sub_heads = self.num_heads

        if self.source is None:
            self.source = SourceLayerConfig(
                num_heads=4 * self.num_heads,
                JK=self.JK,
                JV=self.JV,
                JT=self.JT,
                input_dim=self.input_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        else:
            self.source.num_heads = 4 * self.num_heads
            self.source.__post_init__()
        if self.transition is None:
            self.transition = LongRangeTransitionLayerConfig(
                num_heads=4 * self.num_heads,
                input_dim=self.input_dim,
                transition_dim=self.JT,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        else:
            self.transition.num_heads = 4 * self.num_heads
            self.transition.__post_init__()
        if self.mark is None:
            self.mark = MarkLayerConfig(
                num_heads=4 * self.num_heads,
                JO=self.JO,
                JQ=self.JQ,
                JT=self.JT,
                input_dim=self.input_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        else:
            self.mark.num_heads = 4 * self.num_heads
            self.mark.__post_init__()
        if self.direct is None:
            self.direct = DirectLayerConfig(
                num_heads=self.num_heads,
                JO=self.JO,
                JQ=self.JQ,
                JK=self.JK,
                JV=self.JV,
                input_dim=self.input_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

        if self.passthrough is None and self.additional_passthrough:
            self.passthrough = PassthroughLayerConfig(
                input_dim=self.input_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        if self.convolution is None and self.additional_convolution:
            self.convolution = Convolution2DLayerConfig(
                input_dim=self.input_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

        if self.key is None and not self.tie_query_key:
            self.key = KeyLayerConfig(
                input_dim=self.input_dim,
                sub_heads=self.sub_heads,
                num_heads=self.num_heads,
                DK=self.DK,
                JK=self.JK,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        if self.query is None:
            self.query = QueryLayerConfig(
                input_dim=self.input_dim,
                sub_heads=self.sub_heads,
                num_heads=self.num_heads,
                DK=self.DK,
                JQ=self.JQ,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        if self.value is None:
            self.value = ValueLayerConfig(
                input_dim=self.input_dim,
                sub_heads=self.sub_heads,
                num_heads=self.num_heads,
                DV=self.DV,
                JV=self.JV,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

        if self.mhnorm is None:
            self.mhnorm = MultiHeadRMSNormConfig(
                input_dim=self.input_dim,
                num_heads=self.num_heads,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

        assert self.JT == self.transition.transition_dim, "Transition dimension has to be JT state tracking dimension"

        assert self.JT == self.mark.JT
        assert self.JT == self.source.JT
        assert self.JO == self.mark.JO
        assert self.JQ == self.mark.JQ
        assert self.JK == self.source.JK
        assert self.JV == self.source.JV
        assert self.input_dim == self.source.input_dim
        assert self.input_dim == self.transition.input_dim
        assert self.input_dim == self.mark.input_dim

        assert self.query.DK == self.key.DK == self.DK
        assert self.value.DV == self.DV
        assert self.query.input_dim == self.input_dim
        assert self.key.input_dim == self.input_dim
        assert self.value.input_dim == self.input_dim

        assert self.output_dim == self.num_heads * self.DV * self.JO

        assert_check_literals(self)
