import jax.numpy as jnp
import jax
from plstm.config.norm import MultiHeadLayerNormConfig
from plstm.nnx.norm import MultiHeadLayerNorm
from flax import nnx

mhn = MultiHeadLayerNorm(MultiHeadLayerNormConfig(input_dim=64, num_heads=4), rngs=nnx.Rngs())

inp = jnp.ones([3, 4, 64]) + jax.random.normal(jax.random.PRNGKey(42), [3, 4, 64])

nnx.bridge.lazy_init(mhn, inp)
out = mhn(inp)
