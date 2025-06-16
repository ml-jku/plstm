"""Test parameter conversion between JAX (nnx/linen) and PyTorch transformer
blocks."""

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
from plstm.torch.transformer_block import TransformerBlock as TorchTransformerBlock
from plstm.nnx.transformer_block import TransformerBlock as NNXTransformerBlock
from plstm.linen.transformer_block import TransformerBlock as LinenTransformerBlock
from plstm.config.transformer_block import TransformerBlockConfig
from plstm.config.norm import RMSNormConfig
from plstm.config.scale import ScaleLayerConfig

from plstm.nnx_dummy import _NNX_IS_DUMMY

has_nnx = not _NNX_IS_DUMMY


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,seq_len,input_dim,num_heads,gated,bias",
    [
        (2, 3, 32, 4, True, True),  # Standard case with gated and bias
        (1, 3, 16, 2, True, False),  # With gated, without bias
        (4, 6, 64, 8, False, True),  # Without gated, with bias
        (2, 4, 32, 4, False, False),  # Without gated and bias
    ],
    ids=[
        "standard-gated-bias",
        "gated-no-bias",
        "no-gated-bias",
        "no-gated-no-bias",
    ],
)
def test_transformer_block_conversion_nnx_torch(batch_size, seq_len, input_dim, num_heads, gated, bias, seed, request):
    """Test conversion between NNX and PyTorch TransformerBlock."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = TransformerBlockConfig(
        input_dim=input_dim,
        num_heads=num_heads,
        gated=gated,
        bias=bias,
        norm=RMSNormConfig(input_dim=input_dim),
        scale=ScaleLayerConfig(input_dim=input_dim),
    )

    # Create models
    nnx_model = NNXTransformerBlock(config, rngs=nnx.Rngs(rng))
    torch_model = TorchTransformerBlock(config)

    # Create input with shape [batch_size, seq_len, input_dim]
    x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
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
        rtol=5e-2,
        atol=1e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test PyTorch to NNX conversion
    # First modify torch parameters to ensure we're testing actual conversion
    with torch.no_grad():
        torch_model.qkv.weight.mul_(1.1)
        torch_model.outproj.weight.mul_(1.1)
        if bias:
            torch_model.qkv.bias.add_(0.1)
            torch_model.outproj.bias.add_(0.1)

    convert_parameters_torch_to_nnx(torch_model, nnx_model)

    # Compare outputs after PyTorch to NNX conversion
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.detach().cpu().numpy(),
        rtol=5e-2,
        atol=1e-2,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,seq_len,input_dim,num_heads,gated,bias",
    [
        (2, 4, 32, 4, True, True),  # Standard case with gated and bias
        (1, 2, 16, 2, True, False),  # With gated, without bias
        (4, 8, 64, 8, False, True),  # Without gated, with bias
        (2, 4, 32, 4, False, False),  # Without gated and bias
    ],
    ids=[
        "standard-gated-bias",
        "gated-no-bias",
        "no-gated-bias",
        "no-gated-no-bias",
    ],
)
def test_transformer_block_conversion_linen_torch(
    batch_size, seq_len, input_dim, num_heads, gated, bias, seed, request
):
    """Test conversion between Linen and PyTorch TransformerBlock."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = TransformerBlockConfig(
        input_dim=input_dim,
        num_heads=num_heads,
        gated=gated,
        bias=bias,
        norm=RMSNormConfig(input_dim=input_dim),
        scale=ScaleLayerConfig(input_dim=input_dim),
    )

    # Create models
    linen_model = LinenTransformerBlock(config)
    torch_model = TorchTransformerBlock(config)

    # Create input with shape [batch_size, seq_len, input_dim]
    x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize Linen model
    variables = linen_model.init(rng, linen_x)

    # Test Linen to PyTorch conversion
    convert_parameters_linen_to_torch(linen_model, torch_model, variables=variables, exmp_input=linen_x)

    # Compare outputs after Linen to PyTorch conversion
    linen_out = linen_model.apply(variables, linen_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-2,
        atol=1e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test PyTorch to Linen conversion
    # First modify torch parameters to ensure we're testing actual conversion
    with torch.no_grad():
        torch_model.qkv.weight.mul_(1.1)
        torch_model.outproj.weight.mul_(1.1)
        if bias:
            torch_model.qkv.bias.add_(0.1)
            torch_model.outproj.bias.add_(0.1)

    updated_variables = convert_parameters_torch_to_linen(
        torch_model, linen_model, variables=variables, exmp_input=linen_x
    )

    # Compare outputs after PyTorch to Linen conversion
    linen_out = linen_model.apply(updated_variables, linen_x)
    torch_out = torch_model(torch_x)
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-2,
        atol=1e-2,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,seq_len,input_dim,num_heads,gated,bias",
    [
        (2, 4, 32, 4, True, True),  # Standard case with gated and bias
        (1, 2, 16, 2, True, False),  # With gated, without bias
        (4, 8, 64, 8, False, True),  # Without gated, with bias
        (2, 4, 32, 4, False, False),  # Without gated and bias
    ],
    ids=[
        "standard-gated-bias",
        "gated-no-bias",
        "no-gated-bias",
        "no-gated-no-bias",
    ],
)
def test_transformer_block_conversion_nnx_linen(batch_size, seq_len, input_dim, num_heads, gated, bias, seed, request):
    """Test conversion between NNX and Linen TransformerBlock."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = TransformerBlockConfig(
        input_dim=input_dim,
        num_heads=num_heads,
        gated=gated,
        bias=bias,
        norm=RMSNormConfig(input_dim=input_dim),
        scale=ScaleLayerConfig(input_dim=input_dim),
    )

    # Create models
    nnx_model = NNXTransformerBlock(config, rngs=nnx.Rngs(rng))
    linen_model = LinenTransformerBlock(config)

    # Create input with shape [batch_size, seq_len, input_dim]
    x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    variables = linen_model.init(rng, linen_x)

    # Test NNX to Linen conversion
    updated_variables = convert_parameters_nnx_to_linen(nnx_model, linen_model, variables=variables, exmp_input=nnx_x)

    # Compare outputs after NNX to Linen conversion
    nnx_out = nnx_model(nnx_x)
    linen_out = linen_model.apply(updated_variables, linen_x)
    assert_allclose_with_plot(
        np.array(nnx_out),
        np.array(linen_out),
        rtol=1e-2,
        atol=1e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test Linen to NNX conversion
    # First modify linen parameters to ensure we're testing actual conversion
    updated_variables2 = updated_variables.copy()
    params = updated_variables2["params"]

    # Modify multiheadattention parameters
    if "multiheadattention" in params:
        if "query" in params["multiheadattention"] and "kernel" in params["multiheadattention"]["query"]:
            params["multiheadattention"]["query"]["kernel"] = params["multiheadattention"]["query"]["kernel"] * 1.1
        if "key" in params["multiheadattention"] and "kernel" in params["multiheadattention"]["key"]:
            params["multiheadattention"]["key"]["kernel"] = params["multiheadattention"]["key"]["kernel"] * 1.1
        if "value" in params["multiheadattention"] and "kernel" in params["multiheadattention"]["value"]:
            params["multiheadattention"]["value"]["kernel"] = params["multiheadattention"]["value"]["kernel"] * 1.1
        if "out" in params["multiheadattention"] and "kernel" in params["multiheadattention"]["out"]:
            params["multiheadattention"]["out"]["kernel"] = params["multiheadattention"]["out"]["kernel"] * 1.1

    convert_parameters_linen_to_nnx(linen_model, nnx_model, variables=updated_variables2, exmp_input=nnx_x)

    # Compare outputs after Linen to NNX conversion
    nnx_out = nnx_model(nnx_x)
    linen_out = linen_model.apply(updated_variables2, linen_x)
    assert_allclose_with_plot(
        np.array(nnx_out),
        np.array(linen_out),
        rtol=1e-2,
        atol=1e-2,
        base_path=f"{test_name}_{next(counter)}",
    )
