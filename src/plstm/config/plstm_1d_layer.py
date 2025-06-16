from dataclasses import dataclass, field
from ..util import assert_check_literals
from .dtype import DTYPES

from .source_mark_layer import (
    SourceLayerConfig,
    MarkLayerConfig,
    DirectLayerConfig,
)
from .longrange_transition_layer import (
    LongRangeTransitionLayerConfig,
)
from .query_key_value import QueryLayerConfig, KeyLayerConfig, ValueLayerConfig
from .passthrough import PassthroughLayerConfig
from .convolution import Convolution1DLayerConfig
from .norm import NormConfig, MultiHeadRMSNormConfig
from .interfaces import ResidualModuleConfig
from .initialization import BiasInitConfig, WeightInitConfig, ZerosInitConfig, SmallInitConfig


@dataclass
class pLSTM1DLayerConfig(ResidualModuleConfig):
    _shortname: str = "pLSTM1D"
    input_dim: int = -1
    output_dim: int = -1
    num_heads: int = 1

    # state tracking dimensions
    JK: int = 1  # key state-tracking dimension
    JQ: int = 1  # query state-tracking dimension
    JV: int = 4  # value state-tracking dimension
    JT: int = 4  # transition state-tracking dimension (most important)
    JO: int = 4  # output state-tracking dimension

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
    sub_heads: int = -1
    additional_passthrough: bool = True
    additional_convolution: bool = True
    convolution: Convolution1DLayerConfig | None = None
    convolution_inputs: str = "STMDQKstP"  # all would be "STMDQKVstP"

    levels: int = 8

    mhnorm: NormConfig | None = None

    outprojection: bool = True
    outprojection_bias: bool = False
    outprojection_weight_init: WeightInitConfig = field(default_factory=SmallInitConfig)
    outprojection_bias_init: BiasInitConfig = field(default_factory=ZerosInitConfig)

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

        if self.mhnorm is None:
            self.mhnorm = MultiHeadRMSNormConfig(
                input_dim=self.input_dim,
                num_heads=self.num_heads,
                eps=1e-5,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

        if self.source is None:
            self.source = SourceLayerConfig(
                num_heads=self.num_heads,
                JK=self.JK,
                JV=self.JV,
                JT=self.JT,
                input_dim=self.input_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        if self.transition is None:
            self.transition = LongRangeTransitionLayerConfig(
                num_heads=self.num_heads,
                input_dim=self.input_dim,
                transition_dim=self.JT,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
        if self.mark is None:
            self.mark = MarkLayerConfig(
                num_heads=self.num_heads,
                JO=self.JO,
                JQ=self.JQ,
                JT=self.JT,
                input_dim=self.input_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )
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
        if self.sub_heads < 0:
            self.sub_heads = self.num_heads
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

        if self.passthrough is None and self.additional_passthrough:
            self.passthrough = PassthroughLayerConfig(
                input_dim=self.input_dim,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
            )

        if self.convolution is None and self.additional_convolution:
            self.convolution = Convolution1DLayerConfig(
                input_dim=self.input_dim,
                output_dim=self.input_dim,
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

        assert self.output_dim == self.DV * self.JO * self.num_heads

        assert_check_literals(self)
