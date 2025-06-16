import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx

from plstm.config.blocks import PreUpProjectionBlockConfig, PostUpProjectionBlockConfig
from plstm.nnx.blocks import PreUpProjectionBlock as NNXPreUpProjectionBlock
from plstm.nnx.blocks import PostUpProjectionBlock as NNXPostUpProjectionBlock
from plstm.torch.blocks import PreUpProjectionBlock as TorchPreUpProjectionBlock
from plstm.torch.blocks import PostUpProjectionBlock as TorchPostUpProjectionBlock
from plstm.linen.blocks import PreUpProjectionBlock as LinenPreUpProjectionBlock
from plstm.linen.blocks import PostUpProjectionBlock as LinenPostUpProjectionBlock
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
    "batch_size,seq_len,input_dim,gated,interaction_module_name,skip",
    [
        (2, 16, 32, True, "pLSTM1D", True),  # Basic config with pLSTM1D
        (2, 16, 32, False, "pLSTM1D", True),  # Non-gated with pLSTM1D
        (2, 16, 32, True, "pLSTM2D", True),  # Basic config with pLSTM2D
        (2, 16, 32, False, "pLSTM2D", True),  # Non-gated with pLSTM2D
        (2, 16, 32, True, "pLSTM1D", False),  # No skip connection
        (1, 1, 16, True, "pLSTM1D", True),  # Minimal dimensions
        (4, 1024, 32, True, "pLSTM1D", True),  # Large dimensions
        (4, 32, 32, True, "pLSTM2D", True),  # Large dimensions
    ],
    ids=[
        "basic-plstm1d",
        "nongated-plstm1d",
        "basic-plstm2d",
        "nongated-plstm2d",
        "no-skip",
        "minimal-dims",
        "large-dims1d",
        "large-dims2d",
    ],
)
def test_pre_up_projection_block(
    batch_size, seq_len, input_dim, gated, interaction_module_name, skip, seed, rng, request
):
    """Test PreUpProjectionBlock with various configurations and verify NNX,
    Linen, and PyTorch implementations match."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create config
    config = PreUpProjectionBlockConfig(
        input_dim=input_dim,
        gated=gated,
        interaction_module_name=interaction_module_name,
        skip=skip,
        dtype="float32",
        param_dtype="float32",
    )

    # Create models
    nnx_model = NNXPreUpProjectionBlock(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchPreUpProjectionBlock(config)
    linen_model = LinenPreUpProjectionBlock(config)

    # Create input
    if "1D" in interaction_module_name:
        x = np.random.randn(batch_size, seq_len, input_dim).astype("float32")
    elif "2D" in interaction_module_name:
        x = np.random.randn(batch_size, seq_len, seq_len, input_dim).astype("float32")

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
        rtol=2e-2,
        atol=2e-2,
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
        rtol=2e-2,
        atol=2e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=2e-2,
        atol=2e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test gradients
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
        rtol=2e-2,
        atol=2e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(nnx_grad),
        np.array(linen_grad),
        rtol=2e-2,
        atol=2e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(linen_grad),
        torch_grad.cpu().detach().numpy(),
        rtol=2e-2,
        atol=2e-2,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,input_dim,gated,interaction_module_name,skip,use_scale",
    [
        (2, 16, 32, True, "pLSTM1D", True, False),  # Basic config with pLSTM1D
        (2, 16, 32, False, "pLSTM1D", True, False),  # Non-gated with pLSTM1D
        (2, 16, 32, True, "pLSTM2D", True, False),  # Basic config with pLSTM2D
        (2, 16, 32, False, "pLSTM2D", True, True),  # Non-gated with pLSTM2D
        (2, 16, 32, True, "pLSTM1D", False, False),  # No skip connection
        (1, 1, 16, True, "pLSTM1D", True, False),  # Minimal dimensions
        (4, 32, 32, True, "pLSTM2D", True, False),  # Large dimensions
    ],
    ids=[
        "basic-plstm1d",
        "nongated-plstm1d",
        "basic-plstm2d",
        "nongated-plstm2d",
        "no-skip",
        "minimal-dims",
        "large-dims",
    ],
)
def test_post_up_projection_block(
    batch_size, seq_len, input_dim, gated, interaction_module_name, skip, use_scale, seed, rng, request
):
    """Test PostUpProjectionBlock with various configurations and verify NNX,
    Linen, and PyTorch implementations match."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create config
    config = PostUpProjectionBlockConfig(
        input_dim=input_dim,
        gated=gated,
        interaction_module_name=interaction_module_name,
        skip=skip,
        dtype="float32",
        param_dtype="float32",
        use_scale=use_scale,
    )

    # Create models
    nnx_model = NNXPostUpProjectionBlock(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchPostUpProjectionBlock(config)
    linen_model = LinenPostUpProjectionBlock(config)

    # Create input
    if "1D" in interaction_module_name:
        x = np.random.randn(batch_size, seq_len, input_dim).astype("float32")
    elif "2D" in interaction_module_name:
        x = np.random.randn(batch_size, seq_len, seq_len, input_dim).astype("float32")

    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    linen_variables = linen_model.init(rng_linen, linen_x, deterministic=True)

    # Convert parameters from NNX to PyTorch
    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x, deterministic=True)
    torch_out = torch_model(torch_x, deterministic=True)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.cpu().detach().numpy(),
        rtol=2e-2,
        atol=2e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Convert parameters from NNX to Linen
    updated_linen_variables = convert_parameters_nnx_to_linen(
        nnx_model, linen_model, linen_variables, exmp_input=linen_x
    )

    # Test forward pass for Linen
    linen_out = linen_model.apply(updated_linen_variables, linen_x, deterministic=True)

    # Compare NNX and Linen outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        np.array(linen_out),
        rtol=2e-2,
        atol=2e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=2e-2,
        atol=2e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test gradients
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
        rtol=2e-2,
        atol=2e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(nnx_grad),
        np.array(linen_grad),
        rtol=2e-2,
        atol=2e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(linen_grad),
        torch_grad.cpu().detach().numpy(),
        rtol=2e-2,
        atol=2e-2,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.parametrize("block_type", ["pre", "post"])
@pytest.mark.parametrize("framework", ["jax", "torch", "linen"])
def test_invalid_config(block_type, framework, request):
    """Test that invalid configurations raise appropriate errors in all
    frameworks."""
    if block_type == "pre":
        config_class = PreUpProjectionBlockConfig
        if framework == "jax":
            model_class = NNXPreUpProjectionBlock
            model_args = {"rngs": nnx.Rngs(0)}
        elif framework == "torch":
            model_class = TorchPreUpProjectionBlock
            model_args = {}
        else:  # linen
            model_class = LinenPreUpProjectionBlock
            model_args = {}
    else:
        config_class = PostUpProjectionBlockConfig
        if framework == "jax":
            model_class = NNXPostUpProjectionBlock
            model_args = {"rngs": nnx.Rngs(0)}
        elif framework == "torch":
            model_class = TorchPostUpProjectionBlock
            model_args = {}
        else:  # linen
            model_class = LinenPostUpProjectionBlock
            model_args = {}

    # Test invalid input_dim
    with pytest.raises((AssertionError, ValueError)):
        config = config_class(input_dim=-1)
        if framework == "linen":
            linen_model = model_class(config)
            linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 32)))
        else:
            model_class(config, **model_args)

    # Test invalid interaction_module
    with pytest.raises((AssertionError, ValueError)):
        config = config_class(input_dim=32, interaction_module_name="invalid")
        if framework == "linen":
            linen_model = model_class(config)
            linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 32)))
        else:
            model_class(config, **model_args)
