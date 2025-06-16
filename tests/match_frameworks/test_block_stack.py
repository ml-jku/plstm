"""Tests for BlockStack JAX/PyTorch/Linen compatibility."""

import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx

from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools

from plstm.config.blocks import PreUpProjectionBlockConfig, PostUpProjectionBlockConfig
from plstm.config.block_stack import BlockStackConfig
from plstm.nnx.block_stack import BlockStack as NNXBlockStack
from plstm.torch.block_stack import BlockStack as TorchBlockStack
from plstm.linen.block_stack import BlockStack as LinenBlockStack
from plstm.conversion import convert_parameters_nnx_to_torch
from plstm.conversion import (
    convert_parameters_nnx_to_linen,
)
from plstm.conversion.test import assert_parameters_match


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,input_dim,num_blocks,block_type,interaction_module_name",
    [
        (2, 16, 64, 2, "pre", "pLSTM1D"),  # Basic PreUpProjection with pLSTM1D
        (2, 16, 64, 2, "post", "pLSTM1D"),  # Basic PostUpProjection with pLSTM1D
        (1, 1, 32, 1, "pre", "pLSTM1D"),  # Minimal dimensions
        (8, 32, 128, 4, "pre", "pLSTM1D"),  # Large dimensions
    ],
    ids=[
        "pre-1d-basic",
        "post-1d-basic",
        "minimal-dims",
        "large-dims",
    ],
)
def test_block_stack_forward(
    batch_size, seq_len, input_dim, num_blocks, block_type, interaction_module_name, seed, rng, request
):
    """Test BlockStack with various configurations and verify NNX, Linen, and
    PyTorch implementations match.

    Args:
        batch_size: Number of samples in batch
        seq_len: Length of sequence
        input_dim: Input dimension per head
        num_blocks: Number of blocks in the stack
        block_type: Type of block ('pre' or 'post' up-projection)
        interaction_module_name: Type of wrapped model ('pLSTM1D' or 'pLSTM2D')
        seed: Random seed
        rng: JAX RNG key
    """
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create block config
    block_config_class = PreUpProjectionBlockConfig if block_type == "pre" else PostUpProjectionBlockConfig
    block_config = block_config_class(
        input_dim=input_dim,
        interaction_module_name=interaction_module_name,
        dtype="float32",
        param_dtype="float32",
    )

    # Create stack config
    config = BlockStackConfig(
        input_dim=input_dim,
        num_blocks=num_blocks,
        block=block_config,
    )

    # Create models
    nnx_model = NNXBlockStack(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchBlockStack(config)
    linen_model = LinenBlockStack(config)

    # Create input
    x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
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
        torch_out.detach().numpy(),
        rtol=5e-2,
        atol=5e-2,
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
        rtol=5e-2,
        atol=5e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.detach().numpy(),
        rtol=5e-2,
        atol=5e-2,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,seq_len,input_dim,num_blocks,block_type,interaction_module_name",
    [
        (2, 16, 64, 2, "pre", "pLSTM1D"),  # Basic configuration
        (2, 16, 64, 4, "post", "pLSTM1D"),  # Deeper stack
    ],
    ids=[
        "basic",
        "deep",
    ],
)
def test_block_stack_gradient(
    batch_size, seq_len, input_dim, num_blocks, block_type, interaction_module_name, seed, rng, request
):
    """Test BlockStack gradients match between NNX, Linen, and PyTorch."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create configs
    block_config_class = PreUpProjectionBlockConfig if block_type == "pre" else PostUpProjectionBlockConfig
    block_config = block_config_class(
        input_dim=input_dim,
        interaction_module_name=interaction_module_name,
        dtype="float32",
        param_dtype="float32",
    )
    config = BlockStackConfig(
        input_dim=input_dim,
        num_blocks=num_blocks,
        block=block_config,
    )

    # Create models
    nnx_model = NNXBlockStack(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchBlockStack(config)
    linen_model = LinenBlockStack(config)

    # Create input
    x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Convert parameters from NNX to PyTorch
    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    # Convert parameters from NNX to Linen
    updated_linen_variables = convert_parameters_nnx_to_linen(
        nnx_model, linen_model, linen_variables, exmp_input=linen_x
    )

    # NNX gradient
    def nnx_loss_fn(mod: nnx.Module, x: jax.Array) -> jax.Array:
        return jnp.mean(mod(x) ** 2)

    nnx_grad_fn = nnx.grad(nnx_loss_fn, argnums=1)
    nnx_grad = nnx_grad_fn(nnx_model, nnx_x)

    # Linen gradient
    def linen_loss_fn(x: jax.Array) -> jax.Array:
        return jnp.mean(linen_model.apply(updated_linen_variables, x) ** 2)

    linen_grad = jax.grad(linen_loss_fn)(linen_x)

    # PyTorch gradient
    torch_x = torch.from_numpy(x).requires_grad_()
    torch_out = torch_model(torch_x)
    torch_loss = torch.mean(torch_out**2)
    torch_loss.backward()
    torch_grad = torch_x.grad

    # Compare NNX and PyTorch gradients
    assert_allclose_with_plot(
        np.array(nnx_grad),
        torch_grad.numpy(),
        rtol=5e-2,
        atol=5e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare NNX and Linen gradients
    assert_allclose_with_plot(
        np.array(nnx_grad),
        np.array(linen_grad),
        rtol=5e-2,
        atol=5e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch gradients
    assert_allclose_with_plot(
        np.array(linen_grad),
        torch_grad.numpy(),
        rtol=5e-2,
        atol=5e-2,
        base_path=f"{test_name}_{next(counter)}",
    )


def test_numerical_stability(request):
    """Test numerical stability with small values."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    rng = jax.random.PRNGKey(0)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create configs
    block_config = PreUpProjectionBlockConfig(
        input_dim=64,
        interaction_module_name="pLSTM1D",
        dtype="float32",
        param_dtype="float32",
    )
    config = BlockStackConfig(
        input_dim=64,
        num_blocks=2,
        block=block_config,
    )

    # Create models
    nnx_model = NNXBlockStack(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchBlockStack(config)
    linen_model = LinenBlockStack(config)

    # Create input with small values
    x_small = np.random.randn(2, 16, 64) * 1e-6
    nnx_x = jnp.array(x_small)
    linen_x = jnp.array(x_small)
    torch_x = torch.from_numpy(x_small)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Convert parameters from NNX to PyTorch
    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    # Convert parameters from NNX to Linen
    updated_linen_variables = convert_parameters_nnx_to_linen(
        nnx_model, linen_model, linen_variables, exmp_input=linen_x
    )

    # Test forward pass
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)
    linen_out = linen_model.apply(updated_linen_variables, linen_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.detach().numpy(),
        rtol=1e-3,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare NNX and Linen outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        np.array(linen_out),
        rtol=1e-3,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.detach().numpy(),
        rtol=1e-3,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )
