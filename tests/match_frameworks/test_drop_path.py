import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx

from plstm.config.blocks import PostUpProjectionBlockConfig
from plstm.config.transformer_block import TransformerBlockConfig
from plstm.config.vision_blocks import pLSTMVisionBlockConfig1

from plstm.nnx.blocks import PostUpProjectionBlock as NNXPostUpProjectionBlock
from plstm.nnx.transformer_block import TransformerBlock as NNXTransformerBlock
from plstm.nnx.vision_blocks import pLSTMVisionBlock1 as NNXpLSTMVisionBlock1

from plstm.torch.blocks import PostUpProjectionBlock as TorchPostUpProjectionBlock
from plstm.torch.transformer_block import TransformerBlock as TorchTransformerBlock
from plstm.torch.vision_blocks import pLSTMVisionBlock1 as TorchpLSTMVisionBlock1

from plstm.linen.blocks import PostUpProjectionBlock as LinenPostUpProjectionBlock
from plstm.linen.transformer_block import TransformerBlock as LinenTransformerBlock
from plstm.linen.vision_blocks import pLSTMVisionBlock1 as LinenpLSTMVisionBlock1

from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
from plstm.conversion.test import assert_parameters_match
import itertools
from plstm.conversion import convert_parameters_nnx_to_torch
from plstm.conversion import (
    convert_parameters_nnx_to_linen,
)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,input_dim,drop_path_rate,deterministic",
    [
        (2, 16, 32, 0.0, True),  # No drop_path, deterministic
        (2, 16, 32, 0.0, False),  # No drop_path, non-deterministic
        (2, 16, 32, 1.0, True),  # Full drop_path, deterministic (should be identity)
        (2, 16, 32, 1.0, False),  # Full drop_path, non-deterministic (should drop everything)
        (2, 16, 32, 0.5, True),  # Partial drop_path, deterministic
        (2, 16, 32, 0.5, False),  # Partial drop_path, non-deterministic
        (1, 1, 16, 0.0, True),  # Minimal dimensions, no drop_path
        (1, 1, 16, 1.0, True),  # Minimal dimensions, full drop_path
        (4, 32, 64, 0.2, True),  # Larger dimensions, small drop_path
        (4, 32, 64, 0.8, True),  # Larger dimensions, large drop_path
    ],
    ids=[
        "no-drop-det",
        "no-drop-nondet",
        "full-drop-det",
        "full-drop-nondet",
        "partial-drop-det",
        "partial-drop-nondet",
        "minimal-no-drop",
        "minimal-full-drop",
        "large-small-drop",
        "large-large-drop",
    ],
)
def test_post_up_projection_block_drop_path(
    batch_size, seq_len, input_dim, drop_path_rate, deterministic, seed, rng, request
):
    """Test PostUpProjectionBlock drop_path functionality across all
    frameworks."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create config with drop_path
    config = PostUpProjectionBlockConfig(
        input_dim=input_dim,
        gated=False,
        interaction_module_name="pLSTM1D",
        skip=True,
        drop_path_rate=drop_path_rate,
        dtype="float32",
        param_dtype="float32",
    )

    # Create models
    nnx_model = NNXPostUpProjectionBlock(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchPostUpProjectionBlock(config)
    linen_model = LinenPostUpProjectionBlock(config)

    # Create input
    x = np.random.randn(batch_size, seq_len, input_dim).astype("float32")
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    linen_variables = linen_model.init(rng_linen, linen_x, deterministic=True)

    # Convert parameters from NNX to PyTorch
    convert_parameters_nnx_to_torch(nnx_model, torch_model)
    assert_parameters_match(nnx_model, torch_model)

    # Convert parameters from NNX to Linen
    updated_linen_variables = convert_parameters_nnx_to_linen(
        nnx_model, linen_model, linen_variables, exmp_input=linen_x
    )

    # Test deterministic behavior
    if deterministic:
        # All frameworks should produce identical outputs when deterministic=True
        nnx_out = nnx_model(nnx_x, deterministic=True)
        torch_out = torch_model(torch_x, deterministic=True)
        linen_out = linen_model.apply(updated_linen_variables, linen_x, deterministic=True)

        # Compare outputs
        assert_allclose_with_plot(
            np.array(nnx_out),
            torch_out.cpu().detach().numpy(),
            rtol=2e-2,
            atol=2e-2,
            base_path=f"{test_name}_{next(counter)}_det_nnx_torch",
        )

        assert_allclose_with_plot(
            np.array(nnx_out),
            np.array(linen_out),
            rtol=2e-2,
            atol=2e-2,
            base_path=f"{test_name}_{next(counter)}_det_nnx_linen",
        )

        assert_allclose_with_plot(
            np.array(linen_out),
            torch_out.cpu().detach().numpy(),
            rtol=2e-2,
            atol=2e-2,
            base_path=f"{test_name}_{next(counter)}_det_linen_torch",
        )

    # Test border cases
    if drop_path_rate == 0.0:
        # With drop_path_rate=0.0, deterministic and non-deterministic should be identical
        nnx_out_det = nnx_model(nnx_x, deterministic=True)
        nnx_out_nondet = nnx_model(nnx_x, deterministic=False)

        torch_out_det = torch_model(torch_x, deterministic=True)
        torch_out_nondet = torch_model(torch_x, deterministic=False)

        linen_out_det = linen_model.apply(updated_linen_variables, linen_x, deterministic=True)
        linen_out_nondet = linen_model.apply(updated_linen_variables, linen_x, deterministic=False)

        # Should be identical for drop_path_rate=0.0
        assert_allclose_with_plot(
            np.array(nnx_out_det),
            np.array(nnx_out_nondet),
            rtol=1e-6,
            atol=1e-6,
            base_path=f"{test_name}_{next(counter)}_zero_drop_nnx",
        )

        assert_allclose_with_plot(
            torch_out_det.cpu().detach().numpy(),
            torch_out_nondet.cpu().detach().numpy(),
            rtol=1e-6,
            atol=1e-6,
            base_path=f"{test_name}_{next(counter)}_zero_drop_torch",
        )

        assert_allclose_with_plot(
            np.array(linen_out_det),
            np.array(linen_out_nondet),
            rtol=1e-6,
            atol=1e-6,
            base_path=f"{test_name}_{next(counter)}_zero_drop_linen",
        )

    elif drop_path_rate == 1.0 and not deterministic:
        # With drop_path_rate=1.0 and deterministic=False, output should equal input (identity)
        nnx_out = nnx_model(nnx_x, deterministic=False)
        torch_out = torch_model(torch_x, deterministic=False)
        linen_out = linen_model.apply(updated_linen_variables, linen_x, deterministic=False)

        # Should be identical to input for drop_path_rate=1.0, deterministic=False
        assert_allclose_with_plot(
            np.array(nnx_out),
            x,
            rtol=2e-2,
            atol=2e-2,
            base_path=f"{test_name}_{next(counter)}_full_drop_det_nnx",
        )

        assert_allclose_with_plot(
            torch_out.cpu().detach().numpy(),
            x,
            rtol=2e-2,
            atol=2e-2,
            base_path=f"{test_name}_{next(counter)}_full_drop_det_torch",
        )

        assert_allclose_with_plot(
            np.array(linen_out),
            x,
            rtol=2e-2,
            atol=2e-2,
            base_path=f"{test_name}_{next(counter)}_full_drop_det_linen",
        )


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize(
    "batch_size,seq_len,input_dim,drop_path_rate,deterministic",
    [
        (2, 16, 32, 0.0, True),  # No drop_path, deterministic
        (2, 16, 32, 0.0, False),  # No drop_path, non-deterministic
        (2, 16, 32, 1.0, True),  # Full drop_path, deterministic
        (1, 8, 16, 0.5, True),  # Partial drop_path, deterministic
    ],
    ids=[
        "no-drop-det",
        "no-drop-nondet",
        "full-drop-det",
        "partial-drop-det",
    ],
)
def test_transformer_block_drop_path(batch_size, seq_len, input_dim, drop_path_rate, deterministic, seed, rng, request):
    """Test TransformerBlock drop_path functionality across all frameworks."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create config with drop_path
    config = TransformerBlockConfig(
        input_dim=input_dim,
        num_heads=2,  # Use 2 heads for simplicity
        gated=False,
        skip=True,
        drop_path_rate=drop_path_rate,
        dtype="float32",
        param_dtype="float32",
    )

    # Create models
    nnx_model = NNXTransformerBlock(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchTransformerBlock(config)
    linen_model = LinenTransformerBlock(config)

    # Create input
    x = np.random.randn(batch_size, seq_len, input_dim).astype("float32")
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    linen_variables = linen_model.init(rng_linen, linen_x, deterministic=True)

    # Convert parameters from NNX to PyTorch
    convert_parameters_nnx_to_torch(nnx_model, torch_model)
    # assert_parameters_match(nnx_model, torch_model)

    # Convert parameters from NNX to Linen
    updated_linen_variables = convert_parameters_nnx_to_linen(
        nnx_model, linen_model, linen_variables, exmp_input=linen_x
    )

    # Test deterministic behavior
    if deterministic:
        # All frameworks should produce identical outputs when deterministic=True
        nnx_out = nnx_model(nnx_x, deterministic=True)
        torch_out = torch_model(torch_x, deterministic=True)
        linen_out = linen_model.apply(updated_linen_variables, linen_x, deterministic=True)

        # Compare outputs
        assert_allclose_with_plot(
            np.array(nnx_out),
            torch_out.cpu().detach().numpy(),
            rtol=2e-2,
            atol=2e-2,
            base_path=f"{test_name}_{next(counter)}_det_nnx_torch",
        )

        assert_allclose_with_plot(
            np.array(nnx_out),
            np.array(linen_out),
            rtol=2e-2,
            atol=2e-2,
            base_path=f"{test_name}_{next(counter)}_det_nnx_linen",
        )

        assert_allclose_with_plot(
            np.array(linen_out),
            torch_out.cpu().detach().numpy(),
            rtol=2e-2,
            atol=2e-2,
            base_path=f"{test_name}_{next(counter)}_det_linen_torch",
        )

    # Test border cases for drop_path_rate=0.0
    if drop_path_rate == 0.0:
        # With drop_path_rate=0.0, deterministic and non-deterministic should be identical
        nnx_out_det = nnx_model(nnx_x, deterministic=True)
        nnx_out_nondet = nnx_model(nnx_x, deterministic=False)

        torch_out_det = torch_model(torch_x, deterministic=True)
        torch_out_nondet = torch_model(torch_x, deterministic=False)

        linen_out_det = linen_model.apply(updated_linen_variables, linen_x, deterministic=True)
        linen_out_nondet = linen_model.apply(updated_linen_variables, linen_x, deterministic=False)

        # Should be identical for drop_path_rate=0.0
        assert_allclose_with_plot(
            np.array(nnx_out_det),
            np.array(nnx_out_nondet),
            rtol=1e-6,
            atol=1e-6,
            base_path=f"{test_name}_{next(counter)}_zero_drop_nnx",
        )

        assert_allclose_with_plot(
            torch_out_det.cpu().detach().numpy(),
            torch_out_nondet.cpu().detach().numpy(),
            rtol=1e-6,
            atol=1e-6,
            base_path=f"{test_name}_{next(counter)}_zero_drop_torch",
        )

        assert_allclose_with_plot(
            np.array(linen_out_det),
            np.array(linen_out_nondet),
            rtol=1e-6,
            atol=1e-6,
            base_path=f"{test_name}_{next(counter)}_zero_drop_linen",
        )


@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize(
    "batch_size,seq_len,input_dim,drop_path_rate,deterministic",
    [
        (2, 8, 32, 0.0, True),  # No drop_path, deterministic
        (2, 8, 32, 0.0, False),  # No drop_path, non-deterministic
        (2, 8, 32, 1.0, True),  # Full drop_path, deterministic
        (1, 4, 16, 0.5, True),  # Partial drop_path, deterministic
    ],
    ids=[
        "no-drop-det",
        "no-drop-nondet",
        "full-drop-det",
        "partial-drop-det",
    ],
)
def test_vision_block_drop_path(batch_size, seq_len, input_dim, drop_path_rate, deterministic, seed, rng, request):
    """Test pLSTMVisionBlock1 drop_path functionality across all frameworks."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create config with drop_path
    config = pLSTMVisionBlockConfig1(
        input_dim=input_dim,
        num_heads=2,  # Use 2 heads for simplicity
        drop_path_rate=drop_path_rate,
        dtype="float32",
        param_dtype="float32",
    )

    # Create models
    nnx_model = NNXpLSTMVisionBlock1(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchpLSTMVisionBlock1(config)
    linen_model = LinenpLSTMVisionBlock1(config)

    # Create input (vision blocks expect 4D input: batch, height, width, channels)
    x = np.random.randn(batch_size, seq_len, seq_len, input_dim).astype("float32")
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    linen_variables = linen_model.init(rng_linen, linen_x, deterministic=True)

    # Convert parameters from NNX to PyTorch
    convert_parameters_nnx_to_torch(nnx_model, torch_model)
    assert_parameters_match(nnx_model, torch_model)

    # Convert parameters from NNX to Linen
    updated_linen_variables = convert_parameters_nnx_to_linen(
        nnx_model, linen_model, linen_variables, exmp_input=linen_x
    )

    # Test deterministic behavior
    if deterministic:
        # All frameworks should produce identical outputs when deterministic=True
        nnx_out = nnx_model(nnx_x, deterministic=True)
        torch_out = torch_model(torch_x, deterministic=True)
        linen_out = linen_model.apply(updated_linen_variables, linen_x, deterministic=True)

        # Compare outputs
        assert_allclose_with_plot(
            np.array(nnx_out),
            torch_out.cpu().detach().numpy(),
            rtol=2e-2,
            atol=2e-2,
            base_path=f"{test_name}_{next(counter)}_det_nnx_torch",
        )

        assert_allclose_with_plot(
            np.array(nnx_out),
            np.array(linen_out),
            rtol=2e-2,
            atol=2e-2,
            base_path=f"{test_name}_{next(counter)}_det_nnx_linen",
        )

        assert_allclose_with_plot(
            np.array(linen_out),
            torch_out.cpu().detach().numpy(),
            rtol=2e-2,
            atol=2e-2,
            base_path=f"{test_name}_{next(counter)}_det_linen_torch",
        )

    # Test border cases for drop_path_rate=0.0
    if drop_path_rate == 0.0:
        # With drop_path_rate=0.0, deterministic and non-deterministic should be identical
        nnx_out_det = nnx_model(nnx_x, deterministic=True)
        nnx_out_nondet = nnx_model(nnx_x, deterministic=False)

        torch_out_det = torch_model(torch_x, deterministic=True)
        torch_out_nondet = torch_model(torch_x, deterministic=False)

        linen_out_det = linen_model.apply(updated_linen_variables, linen_x, deterministic=True)
        linen_out_nondet = linen_model.apply(updated_linen_variables, linen_x, deterministic=False)

        # Should be identical for drop_path_rate=0.0
        assert_allclose_with_plot(
            np.array(nnx_out_det),
            np.array(nnx_out_nondet),
            rtol=1e-6,
            atol=1e-6,
            base_path=f"{test_name}_{next(counter)}_zero_drop_nnx",
        )

        assert_allclose_with_plot(
            torch_out_det.cpu().detach().numpy(),
            torch_out_nondet.cpu().detach().numpy(),
            rtol=1e-6,
            atol=1e-6,
            base_path=f"{test_name}_{next(counter)}_zero_drop_torch",
        )

        assert_allclose_with_plot(
            np.array(linen_out_det),
            np.array(linen_out_nondet),
            rtol=1e-6,
            atol=1e-6,
            base_path=f"{test_name}_{next(counter)}_zero_drop_linen",
        )

    elif drop_path_rate == 1.0 and not deterministic:
        # With drop_path_rate=1.0 and deterministic=False, output should equal input (identity)
        nnx_out = nnx_model(nnx_x, deterministic=False)
        torch_out = torch_model(torch_x, deterministic=False)
        linen_out = linen_model.apply(updated_linen_variables, linen_x, deterministic=False)

        # Should be identical to input for drop_path_rate=1.0, deterministic=False
        assert_allclose_with_plot(
            np.array(nnx_out),
            x,
            rtol=2e-2,
            atol=2e-2,
            base_path=f"{test_name}_{next(counter)}_full_drop_det_nnx",
        )

        assert_allclose_with_plot(
            torch_out.cpu().detach().numpy(),
            x,
            rtol=2e-2,
            atol=2e-2,
            base_path=f"{test_name}_{next(counter)}_full_drop_det_torch",
        )

        assert_allclose_with_plot(
            np.array(linen_out),
            x,
            rtol=2e-2,
            atol=2e-2,
            base_path=f"{test_name}_{next(counter)}_full_drop_det_linen",
        )


@pytest.mark.parametrize("framework", ["nnx", "torch", "linen"])
@pytest.mark.parametrize("block_type", ["post_up_projection", "transformer", "vision"])
def test_drop_path_invalid_config(framework, block_type, request):
    """Test that invalid drop_path configurations raise appropriate errors."""

    if block_type == "post_up_projection":
        config_class = PostUpProjectionBlockConfig
        if framework == "nnx":
            model_class = NNXPostUpProjectionBlock
            model_args = {"rngs": nnx.Rngs(0)}
        elif framework == "torch":
            model_class = TorchPostUpProjectionBlock
            model_args = {}
        else:  # linen
            model_class = LinenPostUpProjectionBlock
            model_args = {}
    elif block_type == "transformer":
        config_class = TransformerBlockConfig
        if framework == "nnx":
            model_class = NNXTransformerBlock
            model_args = {"rngs": nnx.Rngs(0)}
        elif framework == "torch":
            model_class = TorchTransformerBlock
            model_args = {}
        else:  # linen
            model_class = LinenTransformerBlock
            model_args = {}
    else:  # vision
        config_class = pLSTMVisionBlockConfig1
        if framework == "nnx":
            model_class = NNXpLSTMVisionBlock1
            model_args = {"rngs": nnx.Rngs(0)}
        elif framework == "torch":
            model_class = TorchpLSTMVisionBlock1
            model_args = {}
        else:  # linen
            model_class = LinenpLSTMVisionBlock1
            model_args = {}

    # Test invalid drop_path_rate (negative)
    with pytest.raises((AssertionError, ValueError)):
        config = config_class(input_dim=32, drop_path_rate=-0.1)
        if framework == "linen":
            linen_model = model_class(config)
            if block_type == "vision":
                linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 8, 8, 32)))
            else:
                linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 32)))
        else:
            model_class(config, **model_args)

    # Test invalid drop_path_rate (> 1.0)
    with pytest.raises((AssertionError, ValueError)):
        config = config_class(input_dim=32, drop_path_rate=1.1)
        if framework == "linen":
            linen_model = model_class(config)
            if block_type == "vision":
                linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 8, 8, 32)))
            else:
                linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 32)))
        else:
            model_class(config, **model_args)
