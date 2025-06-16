from flax import linen as nn
import jax

from ..config.lm_model import pLSTMLMModelConfig
from .interfaces import ResidualModule
from .norm import NormInterface
from .scalar import ScalarFunctionLayer
from .dtype import str_dtype_to_jax
from .initialization import InitInterface


class pLSTMLMModel(nn.Module):
    """Language model using pLSTM blocks."""

    config: pLSTMLMModelConfig

    def setup(self):
        # Initialize embeddings
        self.token_embedding = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.embedding_dim,
            dtype=str_dtype_to_jax(self.config.dtype),
            param_dtype=str_dtype_to_jax(self.config.param_dtype),
            embedding_init=self.config.embed_init.instantiate(InitInterface),
        )

        # Initialize block stack
        self.block_stack = self.config.block_stack.instantiate(ResidualModule)

        # Initialize normalization and output layers
        self.post_blocks_norm = self.config.post_blocks_norm.instantiate(NormInterface)

        # Initialize output head if not using tied weights
        if not self.config.tie_weights:
            self.lm_head = nn.Dense(
                features=self.config.vocab_size,
                use_bias=False,
                kernel_init=self.config.head_init.instantiate(InitInterface),
                dtype=str_dtype_to_jax(self.config.dtype),
                param_dtype=str_dtype_to_jax(self.config.param_dtype),
            )

        # Initialize logit soft cap
        self.logit_soft_cap = self.config.logit_soft_cap.instantiate(ScalarFunctionLayer)

    def __call__(self, token_ids: jax.Array, deterministic: bool = False) -> jax.Array:
        # Embed tokens
        x = self.token_embedding(token_ids)

        # Process through block stack
        x = self.block_stack(x, deterministic=deterministic)

        # Apply normalization
        y = self.post_blocks_norm(x)

        # Apply output projection
        if self.config.tie_weights:
            # Use transposed embedding weights for output projection
            y = y @ self.token_embedding.embedding.T
        else:
            # Use separate output projection
            y = self.lm_head(y)

        # Apply soft cap to logits
        y = self.logit_soft_cap(y)

        return y
