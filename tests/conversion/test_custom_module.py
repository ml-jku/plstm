"""Test parameter conversion between custom JAX (nnx/linen) and PyTorch
modules."""

import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx
from flax import linen as nn
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools

from plstm.conversion import (
    convert_parameters_nnx_to_torch,
    convert_parameters_torch_to_nnx,
    convert_parameters_linen_to_torch,
    convert_parameters_torch_to_linen,
    convert_parameters_nnx_to_linen,
    convert_parameters_linen_to_nnx,
)

from plstm.nnx_dummy import _NNX_IS_DUMMY

has_nnx = not _NNX_IS_DUMMY


class CustomTorchModule(torch.nn.Module):
    """Simple custom PyTorch module for testing parameter conversion."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * (x @ self.weight.T + self.bias)


class CustomNNXModule(nnx.Module):
    """Simple custom NNX module for testing parameter conversion."""

    def __init__(self, input_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        super().__init__()
        self.weight = nnx.Param(nnx.initializers.normal()(rngs.params(), (hidden_dim, input_dim), dtype=np.float32))
        self.bias = nnx.Param(nnx.initializers.zeros_init()(rngs.params(), (hidden_dim,), dtype=np.float32))
        self.scale = nnx.Param(jnp.ones((1,)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.scale * (x @ self.weight.T + self.bias)


class CustomLinenModule(nn.Module):
    """Simple custom Linen module for testing parameter conversion."""

    input_dim: int
    hidden_dim: int

    def setup(self):
        self.weight = self.param("weight", nn.initializers.normal(), (self.hidden_dim, self.input_dim))
        self.bias = self.param("bias", nn.initializers.zeros, (self.hidden_dim,))
        self.scale = self.param("scale", lambda *args, **kwargs: jnp.ones((1,)), (1,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.scale * (x @ self.weight.T + self.bias)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,input_dim,hidden_dim,dtype",
    [
        (2, 32, 64, "float32"),  # Standard case
        (1, 16, 16, "float32"),  # Square dimensions
        (4, 128, 256, "float32"),  # Large dimensions
    ],
    ids=[
        "standard",
        "square",
        "large",
    ],
)
def test_custom_module_conversion_nnx_torch(batch_size, input_dim, hidden_dim, dtype, seed, rng, request):
    """Test conversion between custom NNX and PyTorch modules.

    Args:
        batch_size: Number of samples in batch
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        dtype: Data type for computation
        seed: Random seed
        rng: JAX RNG key
    """
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create models
    nnx_model = CustomNNXModule(input_dim, hidden_dim, rngs=nnx.Rngs(rng))
    torch_model = CustomTorchModule(input_dim, hidden_dim)

    # Create input
    x = np.random.randn(batch_size, input_dim).astype(dtype)
    nnx_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize NNX model
    nnx.bridge.lazy_init(nnx_model, nnx_x)

    # Test NNX to PyTorch conversion
    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    # Compare outputs after NNX to PyTorch conversion
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test PyTorch to NNX conversion
    # First modify torch parameters to ensure we're testing actual conversion
    with torch.no_grad():
        torch_model.weight.mul_(2.0)
        torch_model.bias.add_(1.0)
        torch_model.scale.mul_(0.5)

    convert_parameters_torch_to_nnx(torch_model, nnx_model)

    # Compare outputs after PyTorch to NNX conversion
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,input_dim,hidden_dim,dtype",
    [
        (2, 32, 64, "float32"),  # Standard case
        (1, 16, 16, "float32"),  # Square dimensions
        (4, 128, 256, "float32"),  # Large dimensions
    ],
    ids=[
        "standard",
        "square",
        "large",
    ],
)
def test_custom_module_conversion_linen_torch(batch_size, input_dim, hidden_dim, dtype, seed, rng, request):
    """Test conversion between custom Linen and PyTorch modules.

    Args:
        batch_size: Number of samples in batch
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        dtype: Data type for computation
        seed: Random seed
        rng: JAX RNG key
    """
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create models
    linen_model = CustomLinenModule(input_dim=input_dim, hidden_dim=hidden_dim)
    torch_model = CustomTorchModule(input_dim, hidden_dim)

    # Create input
    x = np.random.randn(batch_size, input_dim).astype(dtype)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize Linen model
    variables = linen_model.init(rng, linen_x)

    # Test Linen to PyTorch conversion
    convert_parameters_linen_to_torch(linen_model, torch_model, variables, exmp_input=linen_x)

    # Compare outputs after Linen to PyTorch conversion
    linen_out = linen_model.apply(variables, linen_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test PyTorch to Linen conversion
    # First modify torch parameters to ensure we're testing actual conversion
    with torch.no_grad():
        torch_model.weight.mul_(2.0)
        torch_model.bias.add_(1.0)
        torch_model.scale.mul_(0.5)

    updated_variables = convert_parameters_torch_to_linen(torch_model, linen_model, variables, exmp_input=linen_x)

    # Compare outputs after PyTorch to Linen conversion
    linen_out = linen_model.apply(updated_variables, linen_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,input_dim,hidden_dim,dtype",
    [
        (2, 32, 64, "float32"),  # Standard case
        (1, 16, 16, "float32"),  # Square dimensions
        (4, 128, 256, "float32"),  # Large dimensions
    ],
    ids=[
        "standard",
        "square",
        "large",
    ],
)
def test_custom_module_conversion_nnx_linen(batch_size, input_dim, hidden_dim, dtype, seed, rng, request):
    """Test conversion between custom NNX and Linen modules.

    Args:
        batch_size: Number of samples in batch
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        dtype: Data type for computation
        seed: Random seed
        rng: JAX RNG key
    """
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create models
    nnx_model = CustomNNXModule(input_dim, hidden_dim, rngs=nnx.Rngs(rng))
    linen_model = CustomLinenModule(input_dim=input_dim, hidden_dim=hidden_dim)

    # Create input
    x = np.random.randn(batch_size, input_dim).astype(dtype)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    variables = linen_model.init(rng, linen_x)

    # Test NNX to Linen conversion
    updated_variables = convert_parameters_nnx_to_linen(nnx_model, linen_model, variables, exmp_input=linen_x)

    # Compare outputs after NNX to Linen conversion
    nnx_out = nnx_model(nnx_x)
    linen_out = linen_model.apply(updated_variables, linen_x)
    assert_allclose_with_plot(
        np.array(nnx_out),
        np.array(linen_out),
        rtol=1e-5,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test Linen to NNX conversion
    # First modify linen parameters to ensure we're testing actual conversion
    updated_variables2 = updated_variables.copy()
    updated_variables2["params"]["weight"] = updated_variables2["params"]["weight"] * 2.0
    updated_variables2["params"]["bias"] = updated_variables2["params"]["bias"] + 1.0
    updated_variables2["params"]["scale"] = updated_variables2["params"]["scale"] * 0.5

    convert_parameters_linen_to_nnx(linen_model, nnx_model, updated_variables2, exmp_input=linen_x)

    # Compare outputs after Linen to NNX conversion
    nnx_out = nnx_model(nnx_x)
    linen_out = linen_model.apply(updated_variables2, linen_x)
    assert_allclose_with_plot(
        np.array(nnx_out),
        np.array(linen_out),
        rtol=1e-5,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
def test_numerical_stability_nnx_torch(request):
    """Test conversion with very small values between NNX and PyTorch."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    rng = jax.random.PRNGKey(0)

    # Create models
    nnx_model = CustomNNXModule(32, 64, rngs=nnx.Rngs(rng))
    torch_model = CustomTorchModule(32, 64)

    # Test with very small input
    x_small = (np.random.randn(2, 32) * 1e-6).astype(dtype=np.float32)
    nnx_x = jnp.array(x_small)
    torch_x = torch.from_numpy(x_small)

    # Initialize NNX model
    nnx.bridge.lazy_init(nnx_model, nnx_x)

    # Convert parameters
    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    # Compare outputs
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )


def test_numerical_stability_linen_torch(request):
    """Test conversion with very small values between Linen and PyTorch."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    rng = jax.random.PRNGKey(0)

    # Create models
    linen_model = CustomLinenModule(input_dim=32, hidden_dim=64)
    torch_model = CustomTorchModule(32, 64)

    # Test with very small input
    x_small = (np.random.randn(2, 32) * 1e-6).astype(dtype=np.float32)
    linen_x = jnp.array(x_small)
    torch_x = torch.from_numpy(x_small)

    # Initialize Linen model
    variables = linen_model.init(rng, linen_x)

    # Convert parameters
    convert_parameters_linen_to_torch(linen_model, torch_model, variables, exmp_input=linen_x)

    # Compare outputs
    linen_out = linen_model.apply(variables, linen_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
def test_numerical_stability_nnx_linen(request):
    """Test conversion with very small values between NNX and Linen."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    rng = jax.random.PRNGKey(0)

    # Create models
    nnx_model = CustomNNXModule(32, 64, rngs=nnx.Rngs(rng))
    linen_model = CustomLinenModule(input_dim=32, hidden_dim=64)

    # Test with very small input
    x_small = (np.random.randn(2, 32) * 1e-6).astype(dtype=np.float32)
    nnx_x = jnp.array(x_small)
    linen_x = jnp.array(x_small)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    variables = linen_model.init(rng, linen_x)

    # Convert parameters
    updated_variables = convert_parameters_nnx_to_linen(nnx_model, linen_model, variables, exmp_input=linen_x)

    # Compare outputs
    nnx_out = nnx_model(nnx_x)
    linen_out = linen_model.apply(updated_variables, linen_x)
    assert_allclose_with_plot(
        np.array(nnx_out),
        np.array(linen_out),
        rtol=1e-5,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )
