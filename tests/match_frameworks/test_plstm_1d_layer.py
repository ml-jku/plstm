import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx

from plstm.config.plstm_1d_layer import pLSTM1DLayerConfig
from plstm.nnx.plstm_1d_layer import pLSTM1DLayer as NNXpLSTM1DLayer
from plstm.torch.plstm_1d_layer import pLSTM1DLayer as TorchPLSTM1DLayer
from plstm.linen.plstm_1d_layer import pLSTM1DLayer as LinenpLSTM1DLayer
from plstm.conversion import convert_parameters_nnx_to_torch
from plstm.conversion import convert_parameters_nnx_to_linen
from plstm.conversion.test import assert_parameters_match, assert_linen_nnx_parameters_match

from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    ("batch_size,seq_length,input_dim,num_heads,additional_convolution,additional_passthrough,outprojection,dtype"),
    [
        (2, 32, 96, 3, True, True, True, "float32"),  # Full features
        (2, 32, 128, 4, False, False, False, "float32"),  # Minimal features
        (1, 16, 16, 2, True, False, True, "float32"),  # Small dimensions
        (4, 1024, 128, 8, True, True, True, "float32"),  # Large dimensions
        (2, 32, 128, 4, True, False, False, "float32"),  # Mixed features 1
        (2, 32, 128, 4, False, True, True, "float32"),  # Mixed features 2
    ],
    ids=[
        "full-features",
        "minimal-features",
        "small-dims",
        "large-dims",
        "mixed-features-1",
        "mixed-features-2",
    ],
)
def test_plstm_1d_layer(
    batch_size,
    seq_length,
    input_dim,
    num_heads,
    additional_convolution,
    additional_passthrough,
    outprojection,
    dtype,
    seed,
    rng,
    request,
):
    """Test pLSTM1DLayer with various configurations and verify NNX, Linen, and
    PyTorch implementations match.

    Args:
        batch_size: Number of samples in batch
        seq_length: Length of input sequence
        input_dim: Input dimension
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
    config = pLSTM1DLayerConfig(
        input_dim=input_dim,
        num_heads=num_heads,
        additional_convolution=additional_convolution,
        additional_passthrough=additional_passthrough,
        outprojection=outprojection,
        param_dtype="float32",
        dtype="float32",
    )

    # Create models
    nnx_model = NNXpLSTM1DLayer(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchPLSTM1DLayer(config)
    linen_model = LinenpLSTM1DLayer(config)

    # Create input
    x = np.random.randn(batch_size, seq_length, input_dim).astype(dtype)
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
    assert_linen_nnx_parameters_match(linen_model, nnx_model, variables=updated_linen_variables)
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
        if nnx_grad is not None and torch_grad is not None:
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


def test_invalid_config(request):
    """Test that invalid configurations raise appropriate errors."""
    # Test invalid input_dim
    with pytest.raises((AssertionError, ValueError)):
        config = pLSTM1DLayerConfig(input_dim=-1, num_heads=4)
        TorchPLSTM1DLayer(config)

    # Test invalid num_heads
    with pytest.raises((AssertionError, ValueError)):
        config = pLSTM1DLayerConfig(input_dim=32, num_heads=-1)
        TorchPLSTM1DLayer(config)

    # Test invalid input_dim for Linen
    with pytest.raises((AssertionError, ValueError)):
        config = pLSTM1DLayerConfig(input_dim=-1, num_heads=4)
        linen_model = LinenpLSTM1DLayer(config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 32)))

    # Test invalid num_heads for Linen
    with pytest.raises((AssertionError, ValueError)):
        config = pLSTM1DLayerConfig(input_dim=32, num_heads=-1)
        linen_model = LinenpLSTM1DLayer(config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 32)))


def test_numerical_stability(request):
    """Test numerical stability with small values."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    rng = jax.random.PRNGKey(0)
    rng_nnx, rng_linen = jax.random.split(rng)

    config = pLSTM1DLayerConfig(
        input_dim=32,
        num_heads=4,
        param_dtype="float32",
        dtype="float32",
    )

    nnx_model = NNXpLSTM1DLayer(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchPLSTM1DLayer(config)
    linen_model = LinenpLSTM1DLayer(config)

    # Test with very small values
    x_small = (np.random.randn(2, 32, 32) * 1e-6).astype(dtype=np.float32)
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
