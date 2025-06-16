import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx

from plstm.config.convolution import Convolution1DLayerConfig, Convolution2DLayerConfig
from plstm.nnx.convolution import Convolution1DLayer as NNXConvolution1DLayer
from plstm.nnx.convolution import Convolution2DLayer as NNXConvolution2DLayer
from plstm.torch.convolution import Convolution1DLayer as TorchConvolution1DLayer
from plstm.torch.convolution import Convolution2DLayer as TorchConvolution2DLayer
from plstm.linen.convolution import Convolution1DLayer as LinenConvolution1DLayer
from plstm.linen.convolution import Convolution2DLayer as LinenConvolution2DLayer
from plstm.conversion import convert_parameters_nnx_to_torch
from plstm.conversion import (
    convert_parameters_nnx_to_linen,
)
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,input_dim,kernel_size,bias",
    [
        (2, 16, 32, 4, True),  # Basic case
        (2, 16, 32, 4, False),  # Without bias
        (1, 8, 16, 3, True),  # Minimal dimensions
        (8, 32, 64, 5, True),  # Large dimensions
    ],
    ids=[
        "basic",
        "no-bias",
        "minimal-dims",
        "large-dims",
    ],
)
def test_convolution_1d_layer(batch_size, seq_len, input_dim, kernel_size, bias, seed, rng, request):
    """Test Convolution1DLayer with various configurations and verify NNX,
    Linen, and PyTorch implementations match.

    Args:
        batch_size: Number of samples in batch
        seq_len: Length of sequence
        num_heads: Number of attention heads
        input_dim: Input dimension per head
        kernel_size: Size of convolution kernel
        bias: Whether to use bias
        seed: Random seed
        rng: JAX RNG key
    """
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create config
    config = Convolution1DLayerConfig(
        input_dim=input_dim,
        kernel_size=kernel_size,
        bias=bias,
        param_dtype="float32",
        dtype="float32",
    )

    # Create models
    nnx_model = NNXConvolution1DLayer(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchConvolution1DLayer(config)
    linen_model = LinenConvolution1DLayer(config)

    # Create input
    x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Copy parameters using conversion system
    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.cpu().detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Convert parameters from NNX to Linen
    updated_linen_variables = convert_parameters_nnx_to_linen(
        nnx_model, linen_model, linen_variables, exmp_input=linen_x
    )

    # Test forward pass for Linen
    linen_out = linen_model.apply(updated_linen_variables, linen_x)

    # Compare NNX and Linen outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        np.array(linen_out),
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test gradients
    # NNX gradient
    def nnx_loss(x):
        return jnp.mean(nnx_model(x) ** 2)

    nnx_grad = jax.grad(nnx_loss)(nnx_x)

    # Linen gradient
    def linen_loss(x):
        return jnp.mean(linen_model.apply(updated_linen_variables, x) ** 2)

    linen_grad = jax.grad(linen_loss)(linen_x)

    # PyTorch gradient
    torch_x = torch.from_numpy(x).requires_grad_()
    torch_out = torch_model(torch_x)
    torch_loss = torch.mean(torch_out**2)
    torch_loss.backward()
    torch_grad = torch_x.grad

    # Compare gradients
    assert_allclose_with_plot(
        np.array(nnx_grad),
        torch_grad.cpu().detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(nnx_grad),
        np.array(linen_grad),
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(linen_grad),
        torch_grad.cpu().detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,height,width,input_dim,kernel_size,bias",
    [
        (2, 16, 16, 32, (3, 3), True),  # Basic case
        (2, 16, 16, 32, (3, 3), False),  # Without bias
        (1, 8, 8, 16, (3, 3), True),  # Minimal dimensions
        (8, 32, 32, 64, (5, 5), True),  # Large dimensions
        (2, 16, 24, 32, (3, 5), True),  # Different height/width and kernel sizes
    ],
    ids=[
        "basic",
        "no-bias",
        "minimal-dims",
        "large-dims",
        "asymmetric",
    ],
)
def test_convolution_2d_layer(batch_size, height, width, input_dim, kernel_size, bias, seed, rng, request):
    """Test Convolution2DLayer with various configurations and verify NNX,
    Linen, and PyTorch implementations match.

    Args:
        batch_size: Number of samples in batch
        height: Height of input
        width: Width of input
        num_heads: Number of attention heads
        input_dim: Input dimension per head
        kernel_size: Size of convolution kernel (height, width)
        bias: Whether to use bias
        seed: Random seed
        rng: JAX RNG key
    """
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create config
    config = Convolution2DLayerConfig(
        input_dim=input_dim,
        kernel_size=kernel_size,
        bias=bias,
        param_dtype="float32",
        dtype="float32",
    )

    # Create models
    nnx_model = NNXConvolution2DLayer(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchConvolution2DLayer(config)
    linen_model = LinenConvolution2DLayer(config)

    # Create input
    x = np.random.randn(batch_size, height, width, input_dim).astype(np.float32)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Copy parameters using conversion system
    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.cpu().detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Convert parameters from NNX to Linen
    updated_linen_variables = convert_parameters_nnx_to_linen(
        nnx_model, linen_model, linen_variables, exmp_input=linen_x
    )

    # Test forward pass for Linen
    linen_out = linen_model.apply(updated_linen_variables, linen_x)

    # Compare NNX and Linen outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        np.array(linen_out),
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test gradients
    # NNX gradient
    def nnx_loss(x):
        return jnp.mean(nnx_model(x) ** 2)

    nnx_grad = jax.grad(nnx_loss)(nnx_x)

    # Linen gradient
    def linen_loss(x):
        return jnp.mean(linen_model.apply(updated_linen_variables, x) ** 2)

    linen_grad = jax.grad(linen_loss)(linen_x)

    # PyTorch gradient
    torch_x = torch.from_numpy(x).requires_grad_()
    torch_out = torch_model(torch_x)
    torch_loss = torch.mean(torch_out**2)
    torch_loss.backward()
    torch_grad = torch_x.grad

    # Compare gradients
    assert_allclose_with_plot(
        np.array(nnx_grad),
        torch_grad.cpu().detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(nnx_grad),
        np.array(linen_grad),
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(linen_grad),
        torch_grad.cpu().detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.parametrize("layer_type", ["1d", "2d"])
def test_invalid_config(layer_type, request):
    """Test that invalid configurations raise appropriate errors.

    Args:
        layer_type: Type of convolution layer ('1d' or '2d')
    """
    if layer_type == "1d":
        config_class = Convolution1DLayerConfig
        nnx_class = NNXConvolution1DLayer
        torch_class = TorchConvolution1DLayer
        linen_class = LinenConvolution1DLayer
        valid_kernel = 3
    else:
        config_class = Convolution2DLayerConfig
        nnx_class = NNXConvolution2DLayer
        torch_class = TorchConvolution2DLayer
        linen_class = LinenConvolution2DLayer
        valid_kernel = (3, 3)

    # Test invalid input_dim for PyTorch
    with pytest.raises(AssertionError):
        config = config_class(input_dim=-1, kernel_size=valid_kernel)
        torch_class(config)

    if layer_type == "1d":
        # Test invalid kernel size for 1D PyTorch
        with pytest.raises((AssertionError, ValueError)):
            config = config_class(input_dim=32, kernel_size=-1)
            torch_class(config)
    else:
        # Test invalid kernel size for 2D PyTorch
        with pytest.raises((AssertionError, ValueError)):
            config = config_class(input_dim=32, kernel_size=(-1, -1))
            torch_class(config)

    # Test invalid input_dim for NNX
    with pytest.raises(AssertionError):
        config = config_class(input_dim=-1, kernel_size=valid_kernel)
        nnx_class(config, rngs=nnx.Rngs(jax.random.PRNGKey(0)))

    if layer_type == "1d":
        # Test invalid kernel size for 1D NNX
        with pytest.raises((AssertionError, ValueError)):
            config = config_class(input_dim=32, kernel_size=-1)
            nnx_class(config, rngs=nnx.Rngs(jax.random.PRNGKey(0)))
    else:
        # Test invalid kernel size for 2D NNX
        with pytest.raises((AssertionError, ValueError)):
            config = config_class(input_dim=32, kernel_size=(-1, -1))
            nnx_class(config, rngs=nnx.Rngs(jax.random.PRNGKey(0)))

    # Test invalid input_dim for Linen
    with pytest.raises((AssertionError, ValueError)):
        config = config_class(input_dim=-1, kernel_size=valid_kernel)
        linen_model = linen_class(config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 32)))

    if layer_type == "1d":
        # Test invalid kernel size for 1D Linen
        with pytest.raises((AssertionError, ValueError)):
            config = config_class(input_dim=32, kernel_size=-1)
            linen_model = linen_class(config)
            linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 32)))
    else:
        # Test invalid kernel size for 2D Linen
        with pytest.raises((AssertionError, ValueError)):
            config = config_class(input_dim=32, kernel_size=(-1, -1))
            linen_model = linen_class(config)
            linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 10, 32)))
