import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools

from plstm.config.plstm_2d_layer import pLSTM2DLayerConfig
from plstm.nnx.plstm_2d_layer import pLSTM2DLayer as NNXpLSTM2DLayer
from plstm.torch.plstm_2d_layer import pLSTM2DLayer as TorchPLSTM2DLayer
from plstm.linen.plstm_2d_layer import pLSTM2DLayer as LinenpLSTM2DLayer
from plstm.conversion import convert_parameters_nnx_to_torch
from plstm.conversion import (
    convert_parameters_nnx_to_linen,
)
from plstm.conversion.test import assert_parameters_match


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    (
        "batch_size,height,width,input_dim,mode,num_heads,additional_convolution,"
        "additional_passthrough,outprojection,dtype"
    ),
    [
        (2, 16, 16, 128, "P", 4, True, True, True, "float32"),  # Full features P mode
        (2, 16, 16, 128, "D", 4, True, True, True, "float32"),  # Full features D mode
        (2, 16, 16, 128, "P", 4, False, False, False, "float32"),  # Minimal features P mode
        (2, 16, 16, 128, "D", 4, False, False, False, "float32"),  # Minimal features D mode
        (1, 8, 8, 64, "P", 2, True, True, True, "float32"),  # Small dimensions
        (4, 32, 32, 256, "D", 8, True, True, True, "float32"),  # Large dimensions
    ],
    ids=[
        "p-mode-full",
        "d-mode-full",
        "p-mode-minimal",
        "d-mode-minimal",
        "small-dims",
        "large-dims",
    ],
)
def test_plstm_2d_layer(
    batch_size,
    height,
    width,
    input_dim,
    mode,
    num_heads,
    additional_convolution,
    additional_passthrough,
    outprojection,
    dtype,
    seed,
    rng,
    request,
):
    """Test pLSTM2DLayer with various configurations and verify NNX, Linen, and
    PyTorch implementations match.

    Args:
        batch_size: Number of samples in batch
        height: Height of input
        width: Width of input
        input_dim: Input dimension
        mode: Mode of operation ('P' or 'D')
        num_heads: Number of attention heads
        additional_convolution: Whether to use additional convolution
        additional_passthrough: Whether to use additional passthrough
        outprojection: Whether to use output projection
        dtype: Data type for computation
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
    config = pLSTM2DLayerConfig(
        mode=mode,
        input_dim=input_dim,
        num_heads=num_heads,
        additional_convolution=additional_convolution,
        additional_passthrough=additional_passthrough,
        outprojection=outprojection,
        dtype="float32",
        param_dtype="float32",
    )

    # Create models
    nnx_model = NNXpLSTM2DLayer(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchPLSTM2DLayer(config)
    linen_model = LinenpLSTM2DLayer(config)

    # Create input
    x = np.random.randn(batch_size, height, width, input_dim).astype(dtype)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Convert parameters from NNX to PyTorch
    convert_parameters_nnx_to_torch(nnx_model, torch_model)
    assert_parameters_match(nnx_model, torch_model)

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.cpu().detach().numpy(),
        rtol=1.5e-2,
        atol=1.5e-2,
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
        rtol=1.5e-2,
        atol=1.5e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=1.5e-2,
        atol=1.5e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test gradients if using float32
    if dtype == "float32":
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
            rtol=8e-2,
            atol=8e-6,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(nnx_grad),
            np.array(linen_grad),
            rtol=8e-2,
            atol=8e-6,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(linen_grad),
            torch_grad.cpu().detach().numpy(),
            rtol=8e-2,
            atol=8e-6,
            base_path=f"{test_name}_{next(counter)}",
        )


@pytest.mark.parametrize("mode", ["P", "D"])
def test_invalid_config(mode, request):
    """Test that invalid configurations raise appropriate errors.

    Args:
        mode: Mode of operation ('P' or 'D')
    """
    # Test invalid input_dim
    with pytest.raises((AssertionError, ValueError)):
        config = pLSTM2DLayerConfig(mode=mode, input_dim=-1, num_heads=4)
        TorchPLSTM2DLayer(config)
        linen_model = LinenpLSTM2DLayer(config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 8, 8, 32)))

    # Test invalid num_heads
    with pytest.raises((AssertionError, ValueError)):
        config = pLSTM2DLayerConfig(mode=mode, input_dim=32, num_heads=-1)
        TorchPLSTM2DLayer(config)
        linen_model = LinenpLSTM2DLayer(config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 8, 8, 32)))


def test_numerical_stability(request):
    """Test numerical stability with small values."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    rng = jax.random.PRNGKey(0)
    rng_nnx, rng_linen = jax.random.split(rng)

    config = pLSTM2DLayerConfig(
        mode="P",
        input_dim=32,
        num_heads=4,
        dtype="float32",
        param_dtype="float32",
    )

    nnx_model = NNXpLSTM2DLayer(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchPLSTM2DLayer(config)
    linen_model = LinenpLSTM2DLayer(config)

    # Test with very small values
    x_small = (np.random.randn(2, 16, 16, 32) * 1e-6).astype(np.float32)
    nnx_x = jnp.array(x_small)
    linen_x = jnp.array(x_small)
    torch_x = torch.from_numpy(x_small)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Convert parameters
    convert_parameters_nnx_to_torch(nnx_model, torch_model)
    assert_parameters_match(nnx_model, torch_model)

    updated_linen_variables = convert_parameters_nnx_to_linen(
        nnx_model, linen_model, linen_variables, exmp_input=linen_x
    )

    # Test forward pass
    nnx_out_small = nnx_model(nnx_x)
    torch_out_small = torch_model(torch_x)
    linen_out_small = linen_model.apply(updated_linen_variables, linen_x)

    # Compare outputs
    assert_allclose_with_plot(
        np.array(nnx_out_small),
        torch_out_small.cpu().detach().numpy(),
        rtol=1.5e-2,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(nnx_out_small),
        np.array(linen_out_small),
        rtol=1.5e-2,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(linen_out_small),
        torch_out_small.cpu().detach().numpy(),
        rtol=1.5e-2,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )
