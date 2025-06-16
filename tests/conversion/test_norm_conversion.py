"""Test parameter conversion between JAX (nnx/linen) and PyTorch norm
layers."""

import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx
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
from plstm.torch.norm import (
    MultiHeadLayerNorm as TorchMultiHeadLayerNorm,
    MultiHeadRMSNorm as TorchMultiHeadRMSNorm,
    LayerNorm as TorchLayerNorm,
    RMSNorm as TorchRMSNorm,
    Identity as TorchIdentity,
)
from plstm.nnx.norm import (
    MultiHeadLayerNorm as NNXMultiHeadLayerNorm,
    MultiHeadRMSNorm as NNXMultiHeadRMSNorm,
    LayerNorm as NNXLayerNorm,
    RMSNorm as NNXRMSNorm,
    Identity as NNXIdentity,
)
from plstm.linen.norm import (
    MultiHeadLayerNorm as LinenMultiHeadLayerNorm,
    MultiHeadRMSNorm as LinenMultiHeadRMSNorm,
    LayerNorm as LinenLayerNorm,
    RMSNorm as LinenRMSNorm,
)
from plstm.config.norm import (
    MultiHeadLayerNormConfig,
    MultiHeadRMSNormConfig,
    LayerNormConfig,
    RMSNormConfig,
    IdentityConfig,
)


from plstm.nnx_dummy import _NNX_IS_DUMMY

has_nnx = not _NNX_IS_DUMMY


# MultiHeadLayerNorm Tests
@pytest.mark.skipif(not has_nnx, reason="NNX not available")
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,num_heads,input_dim,bias,scale",
    [
        (2, 4, 32, False, True),  # Standard case with scale
        (1, 2, 16, True, True),  # With bias and scale
        (4, 8, 64, False, True),  # Without bias
        (2, 4, 32, False, False),  # Without scale
    ],
    ids=[
        "standard-with-scale",
        "with-bias-and-scale",
        "without-bias",
        "without-scale",
    ],
)
def test_multihead_layernorm_conversion_nnx_torch(batch_size, num_heads, input_dim, bias, scale, seed, request):
    """Test conversion between NNX and PyTorch MultiHeadLayerNorm."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = MultiHeadLayerNormConfig(
        num_heads=num_heads,
        input_dim=input_dim,
        bias=bias,
        scale=scale,
    )

    # Create models
    nnx_model = NNXMultiHeadLayerNorm(config, rngs=nnx.Rngs(rng))
    torch_model = TorchMultiHeadLayerNorm(config)

    # Create input with shape [batch_size, num_heads, input_dim]
    x = np.random.randn(batch_size, num_heads, input_dim // num_heads).astype(np.float32)
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
    if scale:
        with torch.no_grad():
            torch_model.scale.mul_(2.0)
    if bias:
        with torch.no_grad():
            torch_model.bias.add_(1.0)

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
    "batch_size,num_heads,input_dim,bias,scale",
    [
        (2, 4, 32, False, True),  # Standard case with scale
        (1, 2, 16, True, True),  # With bias and scale
        (4, 8, 64, False, True),  # Without bias
        (2, 4, 32, False, False),  # Without scale
    ],
    ids=[
        "standard-with-scale",
        "with-bias-and-scale",
        "without-bias",
        "without-scale",
    ],
)
def test_multihead_layernorm_conversion_linen_torch(batch_size, num_heads, input_dim, bias, scale, seed, request):
    """Test conversion between Linen and PyTorch MultiHeadLayerNorm."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = MultiHeadLayerNormConfig(
        num_heads=num_heads,
        input_dim=input_dim,
        bias=bias,
        scale=scale,
    )

    # Create models
    linen_model = LinenMultiHeadLayerNorm(config)
    torch_model = TorchMultiHeadLayerNorm(config)

    # Create input with shape [batch_size, num_heads, input_dim]
    x = np.random.randn(batch_size, num_heads, input_dim // num_heads).astype(np.float32)
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
    if scale:
        with torch.no_grad():
            torch_model.scale.mul_(2.0)
    if bias:
        with torch.no_grad():
            torch_model.bias.add_(1.0)

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
    "batch_size,num_heads,input_dim,bias,scale",
    [
        (2, 4, 32, False, True),  # Standard case with scale
        (1, 2, 16, True, True),  # With bias and scale
        (4, 8, 64, False, True),  # Without bias
        (2, 4, 32, False, False),  # Without scale
    ],
    ids=[
        "standard-with-scale",
        "with-bias-and-scale",
        "without-bias",
        "without-scale",
    ],
)
def test_multihead_layernorm_conversion_nnx_linen(batch_size, num_heads, input_dim, bias, scale, seed, request):
    """Test conversion between NNX and Linen MultiHeadLayerNorm."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = MultiHeadLayerNormConfig(
        num_heads=num_heads,
        input_dim=input_dim,
        bias=bias,
        scale=scale,
    )

    # Create models
    nnx_model = NNXMultiHeadLayerNorm(config, rngs=nnx.Rngs(rng))
    linen_model = LinenMultiHeadLayerNorm(config)

    # Create input with shape [batch_size, num_heads, input_dim]
    x = np.random.randn(batch_size, num_heads, input_dim // num_heads).astype(np.float32)
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
    if scale:
        updated_variables2["params"]["norm"]["scale"] = updated_variables2["params"]["norm"]["scale"] * 2.0
    if bias:
        updated_variables2["params"]["norm"]["bias"] = updated_variables2["params"]["norm"]["bias"] + 1.0

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


# MultiHeadRMSNorm Tests
@pytest.mark.skipif(not has_nnx, reason="NNX not available")
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,num_heads,input_dim,scale",
    [
        (2, 4, 32, True),  # Standard case with scale
        (1, 2, 16, True),  # With scale
        (4, 8, 64, True),  # With scale
        (2, 4, 32, False),  # Without scale
    ],
    ids=[
        "standard-with-scale",
        "with-scale-1",
        "with-scale-2",
        "without-scale",
    ],
)
def test_multihead_rmsnorm_conversion_nnx_torch(batch_size, num_heads, input_dim, scale, seed, request):
    """Test conversion between NNX and PyTorch MultiHeadRMSNorm."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = MultiHeadRMSNormConfig(
        num_heads=num_heads,
        input_dim=input_dim,
        scale=scale,
    )

    # Create models
    nnx_model = NNXMultiHeadRMSNorm(config, rngs=nnx.Rngs(rng))
    torch_model = TorchMultiHeadRMSNorm(config)

    # Create input with shape [batch_size, num_heads, input_dim]
    x = np.random.randn(batch_size, num_heads, input_dim // num_heads).astype(np.float32)
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
    if scale:
        with torch.no_grad():
            torch_model.scale.mul_(2.0)

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
    "batch_size,num_heads,input_dim,scale",
    [
        (2, 4, 32, True),  # Standard case with scale
        (1, 2, 16, True),  # With scale
        (4, 8, 64, True),  # With scale
        (2, 4, 32, False),  # Without scale
    ],
    ids=[
        "standard-with-scale",
        "with-scale-1",
        "with-scale-2",
        "without-scale",
    ],
)
def test_multihead_rmsnorm_conversion_linen_torch(batch_size, num_heads, input_dim, scale, seed, request):
    """Test conversion between Linen and PyTorch MultiHeadRMSNorm."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = MultiHeadRMSNormConfig(
        num_heads=num_heads,
        input_dim=input_dim,
        scale=scale,
    )

    # Create models
    linen_model = LinenMultiHeadRMSNorm(config)
    torch_model = TorchMultiHeadRMSNorm(config)

    # Create input with shape [batch_size, num_heads, input_dim]
    x = np.random.randn(batch_size, num_heads, input_dim // num_heads).astype(np.float32)
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
    if scale:
        with torch.no_grad():
            torch_model.scale.mul_(2.0)

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
    "batch_size,num_heads,input_dim,scale",
    [
        (2, 4, 32, True),  # Standard case with scale
        (1, 2, 16, True),  # With scale
        (4, 8, 64, True),  # With scale
        (2, 4, 32, False),  # Without scale
    ],
    ids=[
        "standard-with-scale",
        "with-scale-1",
        "with-scale-2",
        "without-scale",
    ],
)
def test_multihead_rmsnorm_conversion_nnx_linen(batch_size, num_heads, input_dim, scale, seed, request):
    """Test conversion between NNX and Linen MultiHeadRMSNorm."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = MultiHeadRMSNormConfig(
        num_heads=num_heads,
        input_dim=input_dim,
        scale=scale,
    )

    # Create models
    nnx_model = NNXMultiHeadRMSNorm(config, rngs=nnx.Rngs(rng))
    linen_model = LinenMultiHeadRMSNorm(config)

    # Create input with shape [batch_size, num_heads, input_dim]
    x = np.random.randn(batch_size, num_heads, input_dim // num_heads).astype(np.float32)
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


# Test Linen to NNX conversion for MultiHeadRMSNorm
@pytest.mark.skipif(not has_nnx, reason="NNX not available")
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,num_heads,input_dim,scale",
    [
        (2, 4, 32, True),  # Standard case with scale
        (1, 2, 16, True),  # With scale
        (4, 8, 64, True),  # With scale
        (2, 4, 32, False),  # Without scale
    ],
    ids=[
        "standard-with-scale",
        "with-scale-1",
        "with-scale-2",
        "without-scale",
    ],
)
def test_multihead_rmsnorm_conversion_linen_nnx(batch_size, num_heads, input_dim, scale, seed, request):
    """Test conversion between Linen and NNX MultiHeadRMSNorm."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = MultiHeadRMSNormConfig(
        num_heads=num_heads,
        input_dim=input_dim,
        scale=scale,
    )

    # Create models
    nnx_model = NNXMultiHeadRMSNorm(config, rngs=nnx.Rngs(rng))
    linen_model = LinenMultiHeadRMSNorm(config)

    # Create input with shape [batch_size, num_heads, input_dim]
    x = np.random.randn(batch_size, num_heads, input_dim // num_heads).astype(np.float32)
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
    if scale:
        updated_variables2["params"]["norm"]["scale"] = updated_variables2["params"]["norm"]["scale"] * 2.0

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


# LayerNorm Tests
@pytest.mark.skipif(not has_nnx, reason="NNX not available")
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,input_dim,bias,scale",
    [
        (2, 32, False, True),  # Standard case with scale
        (1, 16, True, True),  # With bias and scale
        (4, 64, False, True),  # Without bias
        (2, 32, True, False),  # Without scale
    ],
    ids=[
        "standard-with-scale",
        "with-bias-and-scale",
        "without-bias",
        "without-scale",
    ],
)
def test_layernorm_conversion_nnx_torch(batch_size, input_dim, bias, scale, seed, request):
    """Test conversion between NNX and PyTorch LayerNorm."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = LayerNormConfig(
        input_dim=input_dim,
        bias=bias,
        scale=scale,
    )

    # Create models
    nnx_model = NNXLayerNorm(config, rngs=nnx.Rngs(rng))
    torch_model = TorchLayerNorm(config)

    # Create input with shape [batch_size, input_dim]
    x = np.random.randn(batch_size, input_dim).astype(np.float32)
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
    if scale:
        with torch.no_grad():
            torch_model.scale.mul_(2.0)
    if bias:
        with torch.no_grad():
            torch_model.bias.add_(1.0)

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
    "batch_size,input_dim,bias,scale",
    [
        (2, 32, False, True),  # Standard case with scale
        (1, 16, True, True),  # With bias and scale
        (4, 64, False, True),  # Without bias
        (2, 32, True, False),  # Without scale
    ],
    ids=[
        "standard-with-scale",
        "with-bias-and-scale",
        "without-bias",
        "without-scale",
    ],
)
def test_layernorm_conversion_linen_torch(batch_size, input_dim, bias, scale, seed, request):
    """Test conversion between Linen and PyTorch LayerNorm."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = LayerNormConfig(
        input_dim=input_dim,
        bias=bias,
        scale=scale,
    )

    # Create models
    linen_model = LinenLayerNorm(config)
    torch_model = TorchLayerNorm(config)

    # Create input with shape [batch_size, input_dim]
    x = np.random.randn(batch_size, input_dim).astype(np.float32)
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
    if scale:
        with torch.no_grad():
            torch_model.scale.mul_(2.0)
    if bias:
        with torch.no_grad():
            torch_model.bias.add_(1.0)

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
    "batch_size,input_dim,bias,scale",
    [
        (2, 32, False, True),  # Standard case with scale
        (1, 16, True, True),  # With bias and scale
        (4, 64, False, True),  # Without bias
        (2, 32, True, False),  # Without scale
    ],
    ids=[
        "standard-with-scale",
        "with-bias-and-scale",
        "without-bias",
        "without-scale",
    ],
)
def test_layernorm_conversion_nnx_linen(batch_size, input_dim, bias, scale, seed, request):
    """Test conversion between NNX and Linen LayerNorm."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = LayerNormConfig(
        input_dim=input_dim,
        bias=bias,
        scale=scale,
    )

    # Create models
    nnx_model = NNXLayerNorm(config, rngs=nnx.Rngs(rng))
    linen_model = LinenLayerNorm(config)

    # Create input with shape [batch_size, input_dim]
    x = np.random.randn(batch_size, input_dim).astype(np.float32)
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
    if scale:
        updated_variables2["params"]["norm"]["scale"] = updated_variables2["params"]["norm"]["scale"] * 2.0
    if bias:
        updated_variables2["params"]["norm"]["bias"] = updated_variables2["params"]["norm"]["bias"] + 1.0

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


# RMSNorm Tests
@pytest.mark.skipif(not has_nnx, reason="NNX not available")
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,input_dim,scale",
    [
        (2, 32, True),  # Standard case with scale
        (1, 16, True),  # With scale
        (4, 64, True),  # With scale
        (2, 32, False),  # Without scale
    ],
    ids=[
        "standard-with-scale",
        "with-scale-1",
        "with-scale-2",
        "without-scale",
    ],
)
def test_rmsnorm_conversion_nnx_torch(batch_size, input_dim, scale, seed, request):
    """Test conversion between NNX and PyTorch RMSNorm."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = RMSNormConfig(
        input_dim=input_dim,
        scale=scale,
    )

    # Create models
    nnx_model = NNXRMSNorm(config, rngs=nnx.Rngs(rng))
    torch_model = TorchRMSNorm(config)

    # Create input with shape [batch_size, input_dim]
    x = np.random.randn(batch_size, input_dim).astype(np.float32)
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
    if scale:
        with torch.no_grad():
            torch_model.scale.mul_(2.0)

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
    "batch_size,input_dim,scale",
    [
        (2, 32, True),  # Standard case with scale
        (1, 16, True),  # With scale
        (4, 64, True),  # With scale
        (2, 32, False),  # Without scale
    ],
    ids=[
        "standard-with-scale",
        "with-scale-1",
        "with-scale-2",
        "without-scale",
    ],
)
def test_rmsnorm_conversion_linen_torch(batch_size, input_dim, scale, seed, request):
    """Test conversion between Linen and PyTorch RMSNorm."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = RMSNormConfig(
        input_dim=input_dim,
        scale=scale,
    )

    # Create models
    linen_model = LinenRMSNorm(config)
    torch_model = TorchRMSNorm(config)

    # Create input with shape [batch_size, input_dim]
    x = np.random.randn(batch_size, input_dim).astype(np.float32)
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
    if scale:
        with torch.no_grad():
            torch_model.scale.mul_(2.0)

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
    "batch_size,input_dim,scale",
    [
        (2, 32, True),  # Standard case with scale
        (1, 16, True),  # With scale
        (4, 64, True),  # With scale
        (2, 32, False),  # Without scale
    ],
    ids=[
        "standard-with-scale",
        "with-scale-1",
        "with-scale-2",
        "without-scale",
    ],
)
def test_rmsnorm_conversion(batch_size, input_dim, scale, seed, request):
    """Test conversion between JAX and PyTorch RMSNorm.

    Args:
        batch_size: Number of samples in batch
        input_dim: Input dimension
        scale: Whether to use scale
        seed: Random seed
    """
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = RMSNormConfig(
        input_dim=input_dim,
        scale=scale,
    )

    # Create models
    jax_model = NNXRMSNorm(config, rngs=nnx.Rngs(rng))
    torch_model = TorchRMSNorm(config)

    # Create input with shape [batch_size, input_dim]
    x = np.random.randn(batch_size, input_dim).astype(np.float32)
    jax_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize JAX model
    nnx.bridge.lazy_init(jax_model, jax_x)

    # Test JAX to PyTorch conversion
    convert_parameters_nnx_to_torch(jax_model, torch_model)

    # Compare outputs after JAX to PyTorch conversion
    jax_out = jax_model(jax_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(jax_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test PyTorch to JAX conversion
    # First modify torch parameters to ensure we're testing actual conversion
    if scale:
        with torch.no_grad():
            torch_model.scale.mul_(2.0)

    convert_parameters_torch_to_nnx(torch_model, jax_model)

    # Compare outputs after PyTorch to JAX conversion
    jax_out = jax_model(jax_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(jax_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,input_dim",
    [
        (2, 32),
        (1, 16),
        (4, 64),
    ],
    ids=[
        "case-1",
        "case-2",
        "case-3",
    ],
)
def test_identity_conversion(batch_size, input_dim, seed, request):
    """Test conversion between JAX and PyTorch Identity.

    Args:
        batch_size: Number of samples in batch
        input_dim: Input dimension
        seed: Random seed
    """
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = IdentityConfig()

    # Create models
    jax_model = NNXIdentity(config, rngs=nnx.Rngs(rng))
    torch_model = TorchIdentity(config)

    # Create input with shape [batch_size, input_dim]
    x = np.random.randn(batch_size, input_dim).astype(np.float32)
    jax_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize JAX model
    nnx.bridge.lazy_init(jax_model, jax_x)

    # Test JAX to PyTorch conversion
    convert_parameters_nnx_to_torch(jax_model, torch_model)

    # Compare outputs after JAX to PyTorch conversion
    jax_out = jax_model(jax_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(jax_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test PyTorch to JAX conversion
    convert_parameters_torch_to_nnx(torch_model, jax_model)

    # Compare outputs after PyTorch to JAX conversion
    jax_out = jax_model(jax_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(jax_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )
