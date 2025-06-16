import jax
from plstm.nnx_dummy import nnx
import logging

from ..config.lm_model import pLSTMLMModelConfig
from .interfaces import ResidualModule
from .norm import NormInterface
from .scalar import ScalarFunctionLayer
from .dtype import str_dtype_to_jax
from .initialization import InitInterface

LOGGER = logging.getLogger(__name__)


class pLSTMLMModel(nnx.Module):
    """Language model using pLSTM blocks."""

    config: pLSTMLMModelConfig
    token_embedding: nnx.Embed
    post_blocks_norm: NormInterface
    lm_head: nnx.Linear
    logit_soft_cap: ScalarFunctionLayer

    def __init__(self, config: pLSTMLMModelConfig, rngs: nnx.Rngs):
        nnx.Module.__init__(self)
        self.config = config

        # Initialize embeddings
        self.token_embedding = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.embedding_dim,
            dtype=str_dtype_to_jax(self.config.dtype),
            embedding_init=config.embed_init.instantiate(InitInterface),
            param_dtype=str_dtype_to_jax(self.config.param_dtype),
            rngs=rngs,
        )

        self.block_stack = config.block_stack.instantiate(ResidualModule, rngs=rngs)
        # Initialize normalization and output layers
        self.post_blocks_norm = config.post_blocks_norm.instantiate(NormInterface, rngs=rngs)

        if not self.config.tie_weights:
            self.lm_head = nnx.Linear(
                in_features=config.embedding_dim,
                out_features=config.vocab_size,
                use_bias=False,
                kernel_init=config.head_init.instantiate(InitInterface),
                dtype=str_dtype_to_jax(self.config.dtype),
                param_dtype=str_dtype_to_jax(self.config.param_dtype),
                rngs=rngs,
            )
        self.logit_soft_cap = self.config.logit_soft_cap.instantiate(ScalarFunctionLayer)

    def __call__(self, token_ids: jax.Array, deterministic: bool = False) -> jax.Array:
        x = self.token_embedding(token_ids)
        x = self.block_stack(x, deterministic=deterministic)
        y = self.post_blocks_norm(x)
        if self.config.tie_weights:
            y = self.token_embedding.attend(y)
        else:
            y = self.lm_head(y)
        y = self.logit_soft_cap(y)
        return y
