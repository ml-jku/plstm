import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools

from plstm.config.vision_util import VitPatchEmbedConfig, VitPosEmbed2dConfig  # , DropPathConfig
from plstm.nnx.vision_util import (
    to_ntuple as nnx_to_ntuple,
    interpolate_sincos as nnx_interpolate_sincos,
    SequenceConv2d as NNXSequenceConv2d,
    VitPatchEmbed as NNXVitPatchEmbed,
    VitPosEmbed2d as NNXVitPosEmbed2d,
    # DropPath as NNXDropPath,
)
from plstm.torch.vision_util import (
    to_ntuple as torch_to_ntuple,
    interpolate_sincos as torch_interpolate_sincos,
    SequenceConv2d as TorchSequenceConv2d,
    VitPatchEmbed as TorchVitPatchEmbed,
    VitPosEmbed2d as TorchVitPosEmbed2d,
    # DropPath as TorchDropPath,
)
from plstm.linen.vision_util import (
    to_ntuple as linen_to_ntuple,
    interpolate_sincos as linen_interpolate_sincos,
    SequenceConv2d as LinenSequenceConv2d,
    VitPatchEmbed as LinenVitPatchEmbed,
    VitPosEmbed2d as LinenVitPosEmbed2d,
    # DropPath as LinenDropPath,
)
from plstm.conversion import (
    convert_parameters_nnx_to_linen,
    convert_parameters_nnx_to_torch,
)


def test_to_ntuple(request):
    """Test to_ntuple function equivalence."""
    # Test with integer input
    assert nnx_to_ntuple(3, 2) == torch_to_ntuple(3, 2) == linen_to_ntuple(3, 2)
    assert nnx_to_ntuple(5, 3) == torch_to_ntuple(5, 3) == linen_to_ntuple(5, 3)

    # Test with tuple input
    assert nnx_to_ntuple((2, 2), 2) == torch_to_ntuple((2, 2), 2) == linen_to_ntuple((2, 2), 2)
    assert nnx_to_ntuple((1, 2, 3), 3) == torch_to_ntuple((1, 2, 3), 3) == linen_to_ntuple((1, 2, 3), 3)

    # Test error cases
    with pytest.raises(AssertionError):
        nnx_to_ntuple((1, 2), 3)
    with pytest.raises(AssertionError):
        torch_to_ntuple((1, 2), 3)
    with pytest.raises(AssertionError):
        linen_to_ntuple((1, 2), 3)


@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,seqlens,embed_dim",
    [
        (1, (8, 8), 32),  # Basic case
        (1, (16, 16), 64),  # Larger dimensions
        (1, (7, 9), 32),  # Non-square dimensions
    ],
)
def test_interpolate_sincos(batch_size, seqlens, embed_dim, seed, request):
    """Test interpolate_sincos function equivalence."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create input embeddings
    nnx_embed = jax.random.normal(rng_nnx, (batch_size, 4, 4, embed_dim))
    torch_embed = torch.from_numpy(np.array(nnx_embed))

    linen_embed = nnx_embed

    # Test interpolation
    nnx_out = nnx_interpolate_sincos(nnx_embed, seqlens)
    linen_out = linen_interpolate_sincos(linen_embed, seqlens)
    torch_out = torch_interpolate_sincos(torch_embed, seqlens)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out), torch_out.numpy(), rtol=1e-1, atol=5e-1, base_path=f"{test_name}_{next(counter)}"
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out), torch_out.numpy(), rtol=1e-1, atol=5e-1, base_path=f"{test_name}_{next(counter)}"
    )

    # Compare NNX and Linen outputs
    assert_allclose_with_plot(
        np.array(nnx_out), np.array(linen_out), rtol=1e-1, atol=1e-1, base_path=f"{test_name}_{next(counter)}"
    )


@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,seq_len,in_channels,out_channels,kernel_size,stride,padding,bias",
    [
        (2, 16, 32, 64, 3, 1, 1, True),  # Basic case
        (2, 16, 32, 64, 3, 2, 1, False),  # Without bias
        (1, 64, 16, 32, 5, 1, 2, True),  # Different kernel and padding
        (4, 64, 64, 128, 3, 2, 1, True),  # Larger dimensions
    ],
)
def test_sequence_conv2d(
    batch_size,
    seq_len,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    bias,
    seed,
    request,
):
    """Test SequenceConv2d layer equivalence."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create models
    nnx_model = NNXSequenceConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        rngs=nnx.Rngs(rng_nnx),
        param_dtype="float32",
        dtype="float32",
    )
    torch_model = TorchSequenceConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    linen_model = LinenSequenceConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        param_dtype="float32",
        dtype="float32",
    )

    # Create input
    x = np.random.randn(batch_size, seq_len, in_channels).astype(np.float32)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize linen model
    linen_variables = linen_model.init(rng_linen, linen_x)

    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out), torch_out.detach().numpy(), rtol=5e-3, atol=5e-3, base_path=f"{test_name}_{next(counter)}"
    )

    # Convert parameters from NNX to Linen
    updated_linen_variables = convert_parameters_nnx_to_linen(
        nnx_model, linen_model, linen_variables, exmp_input=linen_x
    )

    # Test forward pass for Linen
    linen_out = linen_model.apply(updated_linen_variables, linen_x)

    # Compare NNX and Linen outputs
    assert_allclose_with_plot(
        np.array(nnx_out), np.array(linen_out), rtol=5e-3, atol=5e-3, base_path=f"{test_name}_{next(counter)}"
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out), torch_out.detach().numpy(), rtol=5e-3, atol=5e-3, base_path=f"{test_name}_{next(counter)}"
    )


@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,num_channels,resolution,patch_size,dim,channels_first",
    [
        (2, 3, (32, 32), 4, 96, True),  # Basic case
        (2, 3, (32, 32), 4, 96, False),  # Basic case
        (2, 3, (64, 64), 8, 128, True),  # Larger dimensions
        (1, 1, (16, 16), 2, 64, False),  # Minimal dimensions
        (4, 3, (48, 48), 6, 192, False),  # Different patch size
    ],
)
def test_vit_patch_embed(batch_size, num_channels, resolution, patch_size, dim, channels_first, seed, request):
    """Test VitPatchEmbed layer equivalence."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create config and models
    config = VitPatchEmbedConfig(
        dim=dim,
        num_channels=num_channels,
        resolution=resolution,
        patch_size=patch_size,
        channels_first=channels_first,
        param_dtype="float32",
        dtype="float32",
    )
    nnx_model = NNXVitPatchEmbed(config=config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchVitPatchEmbed(config=config)
    linen_model = LinenVitPatchEmbed(config=config)

    # Create input
    if channels_first:
        x = np.random.randn(batch_size, num_channels, *resolution).astype(np.float32)
    else:
        x = np.random.randn(batch_size, *resolution, num_channels).astype(np.float32)

    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize linen model
    linen_variables = linen_model.init(rng_linen, linen_x)
    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out), torch_out.detach().numpy(), rtol=5e-3, atol=5e-3, base_path=f"{test_name}_{next(counter)}"
    )

    # Convert parameters from NNX to Linen
    updated_linen_variables = convert_parameters_nnx_to_linen(
        nnx_model, linen_model, linen_variables, exmp_input=linen_x
    )

    # Test forward pass for Linen
    linen_out = linen_model.apply(updated_linen_variables, linen_x)

    # Compare NNX and Linen outputs
    assert_allclose_with_plot(
        np.array(nnx_out), np.array(linen_out), rtol=5e-3, atol=5e-3, base_path=f"{test_name}_{next(counter)}"
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out), torch_out.detach().numpy(), rtol=5e-3, atol=5e-3, base_path=f"{test_name}_{next(counter)}"
    )


@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,seqlens,dim",
    [
        (2, (8, 8), 96),  # Basic case
        (2, (16, 16), 128),  # Larger dimensions
        (1, (4, 4), 64),  # Minimal dimensions
        (4, (12, 12), 192),  # Different dimensions
    ],
)
def test_vit_pos_embed2d(batch_size, seqlens, dim, seed, request):
    """Test VitPosEmbed2d layer equivalence."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create config and models
    config = VitPosEmbed2dConfig(
        seqlens=seqlens,
        dim=dim,
        param_dtype="float32",
        dtype="float32",
    )
    nnx_model = NNXVitPosEmbed2d(config=config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchVitPosEmbed2d(config=config)
    linen_model = LinenVitPosEmbed2d(config=config)

    # Create input
    x = np.random.randn(batch_size, *seqlens, dim).astype(np.float32)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize linen model
    linen_variables = linen_model.init(rng_linen, linen_x)

    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out), torch_out.detach().numpy(), rtol=5e-3, atol=5e-3, base_path=f"{test_name}_{next(counter)}"
    )

    # Convert parameters from NNX to Linen
    updated_linen_variables = convert_parameters_nnx_to_linen(
        nnx_model, linen_model, linen_variables, exmp_input=linen_x
    )

    # Test forward pass for Linen
    linen_out = linen_model.apply(updated_linen_variables, linen_x)

    # Compare NNX and Linen outputs
    assert_allclose_with_plot(
        np.array(nnx_out), np.array(linen_out), rtol=5e-3, atol=5e-3, base_path=f"{test_name}_{next(counter)}"
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out), torch_out.detach().numpy(), rtol=5e-3, atol=5e-3, base_path=f"{test_name}_{next(counter)}"
    )


def test_invalid_configs(request):
    """Test invalid configurations raise appropriate errors."""
    # Test invalid patch size for VitPatchEmbed
    with pytest.raises(AssertionError):
        config = VitPatchEmbedConfig(
            dim=64,
            num_channels=3,
            resolution=(32, 32),
            param_dtype="float32",
            dtype="float32",
            patch_size=3,  # Invalid: not divisible into resolution
        )
        NNXVitPatchEmbed(config=config, rngs=nnx.Rngs(0))
        TorchVitPatchEmbed(config=config)
        linen_model = LinenVitPatchEmbed(config=config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))

    # Test invalid sequence length for SequenceConv2d
    with pytest.raises(AssertionError):
        x = np.random.randn(2, 15, 32)  # Non-square sequence length
        nnx_model = NNXSequenceConv2d(32, 64, 3, rngs=nnx.Rngs(0))
        torch_model = TorchSequenceConv2d(32, 64, 3)
        linen_model = LinenSequenceConv2d(32, 64, 3)
        nnx_model(jnp.array(x))
        torch_model(torch.from_numpy(x))
        linen_model.init(jax.random.PRNGKey(0), jnp.array(x))
