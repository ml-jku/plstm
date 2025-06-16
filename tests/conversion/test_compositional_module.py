"""Test parameter conversion between compositional JAX (nnx/linen) and PyTorch
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


class TorchLinear(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# PyTorch compositional module
class TorchCompositionalModule(torch.nn.Module):
    """Compositional PyTorch module with linear, norm, and scale parameter."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.linear = TorchLinear(input_dim, hidden_dim)
        self.norm = torch.nn.LayerNorm(hidden_dim, eps=1e-5)
        self.output_linear = torch.nn.Linear(hidden_dim, output_dim)
        self.scale = torch.nn.Parameter(2.0 * torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.output_linear(x)
        return self.scale * x


class NNXLinear(nnx.Module):
    def __init__(self, input_dim, hidden_dim, rngs):
        self.linear = nnx.Linear(input_dim, hidden_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


# NNX compositional module
class NNXCompositionalModule(nnx.Module):
    """Compositional NNX module with linear, norm, and scale parameter."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear = NNXLinear(input_dim, hidden_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(num_features=hidden_dim, epsilon=1e-5, rngs=rngs)
        self.output_linear = nnx.Linear(in_features=hidden_dim, out_features=output_dim, rngs=rngs)
        self.scale = nnx.Param(2.0 * jnp.ones((1,)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.linear(x)
        x = self.norm(x)
        x = self.output_linear(x)
        return self.scale * x


class LinenLinear(nn.Module):
    output_dim: int

    def setup(self):
        self.linear = nn.Dense(self.output_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


# Linen compositional module
class LinenCompositionalModule(nn.Module):
    """Compositional Linen module with dense, norm, and scale parameter."""

    input_dim: int
    hidden_dim: int
    output_dim: int

    def setup(self):
        self.linear = LinenLinear(self.hidden_dim)
        self.norm = nn.LayerNorm(
            epsilon=1e-5,
        )
        self.output_linear = nn.Dense(features=self.output_dim)
        self.scale = self.param("scale", lambda *args, **kwargs: jnp.ones((1,)), (1,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.linear(x)
        x = self.norm(x)
        x = self.output_linear(x)
        return self.scale * x


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,input_dim,hidden_dim,output_dim,dtype",
    [
        (2, 32, 64, 16, "float32"),  # Standard case
        (1, 16, 32, 8, "float32"),  # Small dimensions
        (4, 128, 256, 64, "float32"),  # Large dimensions
    ],
    ids=[
        "standard",
        "small",
        "large",
    ],
)
def test_compositional_module_conversion_nnx_torch(
    batch_size, input_dim, hidden_dim, output_dim, dtype, seed, rng, request
):
    """Test conversion between NNX and PyTorch compositional modules."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create models
    nnx_model = NNXCompositionalModule(input_dim, hidden_dim, output_dim, rngs=nnx.Rngs(rng))
    torch_model = TorchCompositionalModule(input_dim, hidden_dim, output_dim)

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
        rtol=2e-4,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test PyTorch to NNX conversion
    # First modify torch parameters to ensure we're testing actual conversion
    with torch.no_grad():
        torch_model.linear.linear.weight.mul_(1.5)
        torch_model.linear.linear.bias.add_(0.5)
        torch_model.norm.weight.mul_(1.2)
        torch_model.norm.bias.add_(0.2)
        torch_model.output_linear.weight.mul_(1.3)
        torch_model.output_linear.bias.add_(0.3)
        torch_model.scale.mul_(2.0)

    convert_parameters_torch_to_nnx(torch_model, nnx_model)

    # Compare outputs after PyTorch to NNX conversion
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.detach().cpu().numpy(),
        rtol=2e-4,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,input_dim,hidden_dim,output_dim,dtype",
    [
        (2, 32, 64, 16, "float32"),  # Standard case
        (1, 16, 32, 8, "float32"),  # Small dimensions
        (4, 128, 256, 64, "float32"),  # Large dimensions
    ],
    ids=[
        "standard",
        "small",
        "large",
    ],
)
def test_compositional_module_conversion_linen_torch(
    batch_size, input_dim, hidden_dim, output_dim, dtype, seed, rng, request
):
    """Test conversion between Linen and PyTorch compositional modules."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create models
    linen_model = LinenCompositionalModule(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    torch_model = TorchCompositionalModule(input_dim, hidden_dim, output_dim)

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
        rtol=2e-4,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test PyTorch to Linen conversion
    # First modify torch parameters to ensure we're testing actual conversion
    with torch.no_grad():
        torch_model.linear.linear.weight.mul_(1.5)
        torch_model.linear.linear.bias.add_(0.5)
        torch_model.norm.weight.mul_(1.2)
        torch_model.norm.bias.add_(0.2)
        torch_model.output_linear.weight.mul_(1.3)
        torch_model.output_linear.bias.add_(0.3)
        torch_model.scale.mul_(2.0)

    updated_variables = convert_parameters_torch_to_linen(torch_model, linen_model, variables, exmp_input=linen_x)

    # Compare outputs after PyTorch to Linen conversion
    linen_out = linen_model.apply(updated_variables, linen_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.detach().cpu().numpy(),
        rtol=2e-4,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,input_dim,hidden_dim,output_dim,dtype",
    [
        (2, 32, 64, 16, "float32"),  # Standard case
        (1, 16, 32, 8, "float32"),  # Small dimensions
        (4, 128, 256, 64, "float32"),  # Large dimensions
    ],
    ids=[
        "standard",
        "small",
        "large",
    ],
)
def test_compositional_module_conversion_nnx_linen(
    batch_size, input_dim, hidden_dim, output_dim, dtype, seed, rng, request
):
    """Test conversion between NNX and Linen compositional modules."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create models
    nnx_model = NNXCompositionalModule(input_dim, hidden_dim, output_dim, rngs=nnx.Rngs(rng))
    linen_model = LinenCompositionalModule(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

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
        rtol=2e-4,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test Linen to NNX conversion
    # First modify linen parameters to ensure we're testing actual conversion
    updated_variables2 = updated_variables.copy()
    params = updated_variables2["params"].copy()

    # Modify dense parameters
    for k in params:
        if k.startswith("dense/"):
            params[k] = params[k] * 1.5
        elif k.startswith("norm/"):
            params[k] = params[k] * 1.2
        elif k.startswith("output_dense/"):
            params[k] = params[k] * 1.3

    # Modify scale parameter
    params["scale"] = params["scale"] * 2.0

    updated_variables2["params"] = params

    convert_parameters_linen_to_nnx(linen_model, nnx_model, updated_variables2, exmp_input=linen_x)

    # Compare outputs after Linen to NNX conversion
    nnx_out = nnx_model(nnx_x)
    linen_out = linen_model.apply(updated_variables2, linen_x)
    assert_allclose_with_plot(
        np.array(nnx_out),
        np.array(linen_out),
        rtol=2e-4,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
def test_numerical_stability_compositional(request):
    """Test conversion with very small values for compositional modules."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    rng = jax.random.PRNGKey(0)

    # Create models
    input_dim, hidden_dim, output_dim = 32, 64, 16
    nnx_model = NNXCompositionalModule(input_dim, hidden_dim, output_dim, rngs=nnx.Rngs(rng))
    torch_model = TorchCompositionalModule(input_dim, hidden_dim, output_dim)
    linen_model = LinenCompositionalModule(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Test with very small input
    x_small = (np.random.randn(2, input_dim) * 1e-6).astype(dtype=np.float32)
    nnx_x = jnp.array(x_small)
    torch_x = torch.from_numpy(x_small)
    linen_x = jnp.array(x_small)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    variables = linen_model.init(rng, linen_x)

    # Test NNX to PyTorch conversion
    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    # Compare outputs
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.detach().cpu().numpy(),
        rtol=2e-4,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test NNX to Linen conversion
    updated_variables = convert_parameters_nnx_to_linen(nnx_model, linen_model, variables, exmp_input=linen_x)

    # Compare outputs
    nnx_out = nnx_model(nnx_x)
    linen_out = linen_model.apply(updated_variables, linen_x)
    assert_allclose_with_plot(
        np.array(nnx_out),
        np.array(linen_out),
        rtol=2e-4,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test Linen to PyTorch conversion
    convert_parameters_linen_to_torch(linen_model, torch_model, updated_variables, exmp_input=linen_x)

    # Compare outputs
    linen_out = linen_model.apply(updated_variables, linen_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.detach().cpu().numpy(),
        rtol=2e-4,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )
