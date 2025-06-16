"""Test JAX scale layer implementation."""

import pytest
import jax
import jax.numpy as jnp
from plstm.nnx_dummy import nnx

from plstm.nnx.scale import ScaleLayer
from plstm.config.initialization import ConstantInitConfig
from plstm.config.scale import ScaleLayerConfig
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
from plstm.nnx.util import module_named_params


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


def test_scale_layer_config():
    """Test ScaleLayerConfig validation."""
    # Test valid config
    config = ScaleLayerConfig(input_dim=4, scale_init=0.5)
    assert config.output_dim == 4  # Should be set to input_dim

    # Test invalid input_dim
    with pytest.raises(AssertionError):
        ScaleLayerConfig(input_dim=-1)

    # Test invalid output_dim
    with pytest.raises(AssertionError):
        ScaleLayerConfig(input_dim=4, output_dim=8)


def test_scale_layer_init(rng, request):
    """Test ScaleLayer initialization."""
    test_name = request_pytest_filepath(request, __file__)
    config = ScaleLayerConfig(input_dim=4, scale_init=ConstantInitConfig(0.5))
    layer = ScaleLayer(config, rngs=nnx.Rngs(rng))

    # Check scale parameter shape
    assert layer.scale.shape == (4,)  # input_dim

    # Check scale parameter initialization
    expected_scale = 0.5 * jnp.ones(4)
    assert_allclose_with_plot(layer.scale, expected_scale, rtol=1e-5, base_path=f"{test_name}_{0}")


@pytest.mark.parametrize(
    "shape",
    [
        (2, 4),  # batch_size, input_dim
        (3, 2, 4),  # batch_size, seq_len, input_dim
        (2, 3, 4),  # batch_size, seq_len, input_dim
    ],
)
def test_scale_layer_forward(shape, rng, request):
    """Test ScaleLayer forward pass with different input shapes."""
    input_dim = shape[-1]
    config = ScaleLayerConfig(input_dim=input_dim, scale_init=ConstantInitConfig(0.5))
    layer = ScaleLayer(config, rngs=nnx.Rngs(rng))

    # Create input
    x = jnp.ones(shape)

    # Compute output
    y = layer(x)

    # Check output shape
    assert y.shape == x.shape

    # Check scaling
    expected_output = 0.5 * x  # Since scale_init=0.5
    test_name = request_pytest_filepath(request, __file__)
    assert_allclose_with_plot(y, expected_output, rtol=1e-5, base_path=f"{test_name}")


def test_scale_layer_params(rng):
    """Test ScaleLayer parameter access."""
    config = ScaleLayerConfig(input_dim=4)
    layer = ScaleLayer(config, rngs=nnx.Rngs(rng))

    # Get parameters
    params = dict(module_named_params(layer))

    # Verify scale parameter is found
    assert set(params.keys()) == {"scale"}
    assert params["scale"].shape == (4,)  # input_dim
