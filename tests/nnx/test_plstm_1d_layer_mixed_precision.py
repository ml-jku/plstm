import pytest
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx

from plstm.config.plstm_1d_layer import pLSTM1DLayerConfig
from plstm.nnx.plstm_1d_layer import pLSTM1DLayer
from plstm.nnx.util import module_named_params


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize(
    "dtype,param_dtype",
    [
        ("bfloat16", "float32"),  # Mixed precision: BF16 compute, FP32 params
        ("float32", "bfloat16"),  # Mixed precision: FP32 compute, BF16 params
        ("bfloat16", "bfloat16"),  # Pure BF16
    ],
)
@pytest.mark.parametrize(
    "batch_size,seq_len,input_dim,num_heads",
    [
        (2, 128, 128, 4),  # Standard size
        (1, 64, 64, 2),  # Small size
    ],
)
def test_plstm_1d_layer_mixed_precision(
    batch_size,
    seq_len,
    input_dim,
    num_heads,
    dtype,
    param_dtype,
    seed,
    rng,
):
    """Test pLSTM1DLayer with mixed precision configurations.

    Args:
        batch_size: Number of samples in batch
        seq_len: Sequence length
        input_dim: Input dimension
        num_heads: Number of attention heads
        dtype: Data type for computation
        param_dtype: Data type for parameters
        seed: Random seed
        rng: JAX RNG key
    """
    # Set random seed
    rng = jax.random.PRNGKey(seed)

    # Create config with specified dtypes
    config = pLSTM1DLayerConfig(
        input_dim=input_dim,
        num_heads=num_heads,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    # Create model
    model = pLSTM1DLayer(config, rngs=nnx.Rngs(rng))

    # Create input
    x = np.random.randn(batch_size, seq_len, input_dim)
    jax_x = jnp.array(x, dtype=getattr(jnp, dtype))

    # Initialize and run forward pass
    model = nnx.bridge.lazy_init(model, jax_x)
    output = model(jax_x)

    # Verify output dtype matches computation dtype
    assert output.dtype == getattr(jnp, dtype), "Bad output dtype"

    # Verify parameter dtypes match param_dtype
    for _, param in module_named_params(model):
        assert param.dtype == getattr(jnp, param_dtype), "Bad param dtype"

    # Basic shape checks
    expected_shape = (batch_size, seq_len, input_dim)
    assert output.shape == expected_shape, "Bad output shape"

    # Test that the output contains no NaN values
    assert not jnp.any(jnp.isnan(output)), "NaN output"
