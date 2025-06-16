"""Test parameter conversion between JAX (nnx/linen) and PyTorch modules."""

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


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,in_features,out_features,bias,dtype",
    [
        (2, 32, 64, True, "float32"),  # Standard case with bias
        (2, 32, 64, False, "float32"),  # Without bias
        (1, 16, 16, True, "float32"),  # Square matrix
        (4, 128, 256, True, "float32"),  # Large dimensions
    ],
    ids=[
        "standard",
        "no-bias",
        "square",
        "large",
    ],
)
def test_linear_conversion_nnx_torch(batch_size, in_features, out_features, bias, dtype, seed, rng, request):
    """Test conversion between torch.nn.Linear and flax.nnx.Linear."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create models
    nnx_model = nnx.Linear(
        in_features=in_features,
        out_features=out_features,
        use_bias=bias,
        rngs=nnx.Rngs(rng),
    )
    torch_model = torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
    )

    # Create input
    x = np.random.randn(batch_size, in_features).astype(dtype)
    nnx_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize JAX model
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
        if bias:
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
    "batch_size,in_features,out_features,bias,dtype",
    [
        (2, 32, 64, True, "float32"),  # Standard case with bias
        (2, 32, 64, False, "float32"),  # Without bias
        (1, 16, 16, True, "float32"),  # Square matrix
        (4, 128, 256, True, "float32"),  # Large dimensions
    ],
    ids=[
        "standard",
        "no-bias",
        "square",
        "large",
    ],
)
def test_linear_conversion_linen_torch(batch_size, in_features, out_features, bias, dtype, seed, rng, request):
    """Test conversion between torch.nn.Linear and flax.linen.Dense."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    linen_model = nn.Dense(features=out_features, use_bias=bias)
    torch_model = torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
    )

    # Create input
    x = np.random.randn(batch_size, in_features).astype(dtype)
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
        if bias:
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
    "batch_size,in_features,out_features,bias,dtype",
    [
        (2, 32, 64, True, "float32"),  # Standard case with bias
        (2, 32, 64, False, "float32"),  # Without bias
        (1, 16, 16, True, "float32"),  # Square matrix
        (4, 128, 256, True, "float32"),  # Large dimensions
    ],
    ids=[
        "standard",
        "no-bias",
        "square",
        "large",
    ],
)
def test_linear_conversion_nnx_linen(batch_size, in_features, out_features, bias, dtype, seed, rng, request):
    """Test conversion between flax.linen.Dense and flax.nnx.Linear."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create models
    nnx_model = nnx.Linear(
        in_features=in_features,
        out_features=out_features,
        use_bias=bias,
        rngs=nnx.Rngs(rng),
    )

    linen_model = nn.Dense(features=out_features, use_bias=bias)

    # Create input
    x = np.random.randn(batch_size, in_features).astype(dtype)
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
    updated_variables2["params"]["kernel"] = updated_variables2["params"]["kernel"] * 2.0
    if bias:
        updated_variables2["params"]["bias"] = updated_variables2["params"]["bias"] + 1.0

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
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,normalized_shape,elementwise_affine,bias,dtype",
    [
        (2, (64,), True, True, "float32"),  # Standard case with affine
        (2, (64,), False, False, "float32"),  # Without affine
        (1, (16,), True, False, "float32"),  # Small dimensions
        (4, (256,), True, False, "float32"),  # Large dimensions
    ],
    ids=[
        "standard",
        "no-affine",
        "small",
        "large",
    ],
)
def test_layernorm_conversion_nnx_torch(
    batch_size, normalized_shape, elementwise_affine, bias, dtype, seed, rng, request
):
    """Test conversion between torch.nn.LayerNorm and flax.nnx.LayerNorm."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create models
    nnx_model = nnx.LayerNorm(
        num_features=normalized_shape[0],
        use_bias=bias,
        use_scale=elementwise_affine,
        rngs=nnx.Rngs(rng),
        epsilon=1e-5,
    )
    torch_model = torch.nn.LayerNorm(
        normalized_shape=normalized_shape,
        elementwise_affine=elementwise_affine,
        bias=bias,
        eps=1e-5,
    )

    # Create input
    x = np.random.randn(batch_size, *normalized_shape).astype(dtype)
    nnx_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize JAX model
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
    if elementwise_affine:
        with torch.no_grad():
            torch_model.weight.mul_(2.0)

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
    "batch_size,normalized_shape,elementwise_affine,dtype",
    [
        (2, (64,), True, "float32"),  # Standard case with affine
        (2, (64,), False, "float32"),  # Without affine
        (1, (16,), True, "float32"),  # Small dimensions
        (4, (256,), True, "float32"),  # Large dimensions
    ],
    ids=[
        "standard",
        "no-affine",
        "small",
        "large",
    ],
)
def test_layernorm_conversion_linen_torch(batch_size, normalized_shape, elementwise_affine, dtype, seed, rng, request):
    """Test conversion between torch.nn.LayerNorm and flax.linen.LayerNorm."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    linen_model = nn.LayerNorm(
        epsilon=1e-5,
        use_bias=elementwise_affine,
        use_scale=elementwise_affine,
    )
    torch_model = torch.nn.LayerNorm(
        normalized_shape=normalized_shape,
        elementwise_affine=elementwise_affine,
        eps=1e-5,
    )

    # Create input
    x = np.random.randn(batch_size, *normalized_shape).astype(dtype)
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
    if elementwise_affine:
        with torch.no_grad():
            torch_model.weight.mul_(2.0)
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
    "batch_size,normalized_shape,use_scale,use_bias,dtype",
    [
        (2, (64,), True, True, "float32"),  # Standard case with scale and bias
        (2, (64,), True, False, "float32"),  # With scale, without bias
        (2, (64,), False, True, "float32"),  # Without scale, with bias
        (2, (64,), False, False, "float32"),  # Without scale and bias
    ],
    ids=[
        "scale-bias",
        "scale-no-bias",
        "no-scale-bias",
        "no-scale-no-bias",
    ],
)
def test_layernorm_conversion_nnx_linen(batch_size, normalized_shape, use_scale, use_bias, dtype, seed, rng, request):
    """Test conversion between flax.linen.LayerNorm and flax.nnx.LayerNorm."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create models
    nnx_model = nnx.LayerNorm(
        num_features=normalized_shape[0],
        use_bias=use_bias,
        use_scale=use_scale,
        rngs=nnx.Rngs(rng),
        epsilon=1e-5,
    )

    linen_model = nn.LayerNorm(
        epsilon=1e-5,
        use_bias=use_bias,
        use_scale=use_scale,
    )

    # Create input
    x = np.random.randn(batch_size, *normalized_shape).astype(dtype)
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
    if use_scale:
        updated_variables2["params"]["scale"] = updated_variables2["params"]["scale"] * 2.0
    if use_bias:
        updated_variables2["params"]["bias"] = updated_variables2["params"]["bias"] + 1.0

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
def test_numerical_stability(request):
    """Test conversion with very small values."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    rng = jax.random.PRNGKey(0)

    # Test Linear
    nnx_linear = nnx.Linear(in_features=32, out_features=64, rngs=nnx.Rngs(rng), dtype=jnp.float32)
    torch_linear = torch.nn.Linear(in_features=32, out_features=64).to(dtype=torch.float32)

    x_small = np.random.randn(2, 32).astype(dtype=np.float32) * 1e-6
    nnx_x = jnp.array(x_small)
    torch_x = torch.from_numpy(x_small)

    nnx.bridge.lazy_init(nnx_linear, nnx_x)
    convert_parameters_nnx_to_torch(nnx_linear, torch_linear)

    nnx_out = nnx_linear(nnx_x)
    torch_out = torch_linear(torch_x)
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test Linen Dense
    class LinenDense(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(features=64)(x)

    linen_dense = LinenDense()
    variables = linen_dense.init(rng, nnx_x)

    # Test Linen to PyTorch
    convert_parameters_linen_to_torch(linen_dense, torch_linear, variables, exmp_input=nnx_x)

    linen_out = linen_dense.apply(variables, nnx_x)
    torch_out = torch_linear(torch_x)
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        base_path=f"{test_name}_{next(counter)}",
    )
