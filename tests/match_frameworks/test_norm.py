import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx

from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools

# Import the new config classes
from plstm.config.norm import (
    MultiHeadLayerNormConfig,
    MultiHeadRMSNormConfig,
    LayerNormConfig,
    RMSNormConfig,
)

# Import the new implementation classes
from plstm.nnx.norm import (
    MultiHeadLayerNorm as NNXMultiHeadLayerNorm,
    MultiHeadRMSNorm as NNXMultiHeadRMSNorm,
    LayerNorm as NNXLayerNorm,
    RMSNorm as NNXRMSNorm,
)

from plstm.torch.norm import (
    MultiHeadLayerNorm as TorchMultiHeadLayerNorm,
    MultiHeadRMSNorm as TorchMultiHeadRMSNorm,
    LayerNorm as TorchLayerNorm,
    RMSNorm as TorchRMSNorm,
)

from plstm.linen.norm import (
    MultiHeadLayerNorm as LinenMultiHeadLayerNorm,
    MultiHeadRMSNorm as LinenMultiHeadRMSNorm,
    LayerNorm as LinenLayerNorm,
    RMSNorm as LinenRMSNorm,
)

from plstm.conversion import convert_parameters_nnx_to_torch
from plstm.conversion import (
    convert_parameters_nnx_to_linen,
)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,input_dim,bias,scale,dtype,param_dtype,axis",
    [
        (2, 16, 4, 32, True, True, "float32", "float32", -2),  # With bias and scale
        (2, 16, 4, 32, False, True, "float32", "float32", -2),  # Without bias
        (2, 16, 4, 32, True, False, "float32", "float32", -2),  # Without scale
        (2, 16, 4, 32, True, True, "float16", "float32", -2),  # Mixed precision
        (2, 16, 4, 32, True, True, "float32", "float32", None),  # Different axis
        (1, 1, 1, 1, True, True, "float32", "float32", -2),  # Minimal dimensions
        (8, 32, 8, 64, True, True, "float32", "float32", -2),  # Large dimensions
    ],
    ids=[
        "basic",
        "no-bias",
        "no-scale",
        "mixed-precision",
        "different-axis",
        "minimal-dims",
        "large-dims",
    ],
)
def test_multihead_layernorm(
    batch_size, seq_len, num_heads, input_dim, bias, scale, dtype, param_dtype, axis, seed, rng, request
):
    """Test MultiHeadLayerNorm with various configurations and verify NNX,
    Linen, and PyTorch implementations match.

    Args:
        batch_size: Number of samples in batch
        seq_len: Length of sequence
        num_heads: Number of attention heads
        input_dim: Input dimension per head
        bias: Whether to use bias
        scale: Whether to use scale
        dtype: Data type for computation
        param_dtype: Data type for parameters
        axis: Axis for head dimension
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
    config = MultiHeadLayerNormConfig(
        num_heads=num_heads,
        input_dim=input_dim,
        bias=bias,
        scale=scale,
        dtype=dtype,
        param_dtype=param_dtype,
        axis=axis,
    )

    # Create models
    nnx_model = NNXMultiHeadLayerNorm(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchMultiHeadLayerNorm(config)
    linen_model = LinenMultiHeadLayerNorm(config)

    # Create input
    if axis is None:
        shape = [batch_size, seq_len, input_dim]
    elif axis == -2:
        shape = [batch_size, seq_len, num_heads, input_dim // num_heads]
    else:
        shape = [batch_size, seq_len, input_dim // num_heads]
        shape = shape[:axis]
    x = np.random.randn(*shape).astype(dtype)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Convert parameters from NNX to PyTorch
    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.cpu().detach().numpy(),
        rtol=5e-2,
        atol=1e-3,
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
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=5e-2,
        atol=1e-3,
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
        assert_allclose_with_plot(
            np.array(nnx_grad),
            torch_grad.cpu().detach().numpy(),
            rtol=5e-2,
            atol=1e-3,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(nnx_grad),
            np.array(linen_grad),
            rtol=5e-2,
            atol=1e-3,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(linen_grad),
            torch_grad.cpu().detach().numpy(),
            rtol=5e-2,
            atol=1e-3,
            base_path=f"{test_name}_{next(counter)}",
        )


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,input_dim,bias,scale,dtype,param_dtype,axis",
    [
        (2, 16, 4, 32, False, True, "float32", "float32", -2),  # With scale, no bias
        (2, 16, 4, 32, True, True, "float32", "float32", -2),  # With scale and bias
        (2, 16, 4, 32, False, False, "float32", "float32", -2),  # Without scale or bias
        (2, 16, 4, 32, True, False, "float32", "float32", -2),  # With bias, no scale
        (2, 16, 4, 32, True, True, "float16", "float32", -2),  # Mixed precision
        (2, 16, 4, 32, True, True, "float32", "float32", None),  # Different axis
        (1, 1, 1, 1, True, True, "float32", "float32", -2),  # Minimal dimensions
        (8, 32, 8, 64, True, True, "float32", "float32", -2),  # Large dimensions
    ],
    ids=[
        "with-scale-no-bias",
        "with-scale-and-bias",
        "no-scale-no-bias",
        "with-bias-no-scale",
        "mixed-precision",
        "different-axis",
        "minimal-dims",
        "large-dims",
    ],
)
def test_multihead_rmsnorm(
    batch_size, seq_len, num_heads, input_dim, bias, scale, dtype, param_dtype, axis, seed, rng, request
):
    """Test MultiHeadRMSNorm with various configurations and verify NNX, Linen,
    and PyTorch implementations match.

    Args:
        batch_size: Number of samples in batch
        seq_len: Length of sequence
        num_heads: Number of attention heads
        input_dim: Input dimension per head
        scale: Whether to use scale
        dtype: Data type for computation
        param_dtype: Data type for parameters
        axis: Axis for head dimension
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
    config = MultiHeadRMSNormConfig(
        num_heads=num_heads,
        input_dim=input_dim,
        bias=bias,
        scale=scale,
        dtype=dtype,
        param_dtype=param_dtype,
        axis=axis,
    )

    # Create models
    nnx_model = NNXMultiHeadRMSNorm(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchMultiHeadRMSNorm(config)
    linen_model = LinenMultiHeadRMSNorm(config)

    # Create input
    if axis is None:
        shape = [batch_size, seq_len, input_dim]
    elif axis == -2:
        shape = [batch_size, seq_len, num_heads, input_dim // num_heads]
    else:
        shape = [batch_size, seq_len, input_dim // num_heads]
        shape = shape[:axis]
    x = np.random.randn(*shape).astype(dtype)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Convert parameters from NNX to PyTorch
    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.cpu().detach().numpy(),
        rtol=5e-2,
        atol=1e-8,
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
        atol=1e-8,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=5e-2,
        atol=1e-8,
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
        assert_allclose_with_plot(
            np.array(nnx_grad),
            torch_grad.cpu().detach().numpy(),
            rtol=5e-2,
            atol=1e-8,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(nnx_grad),
            np.array(linen_grad),
            rtol=5e-2,
            atol=1e-8,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(linen_grad),
            torch_grad.cpu().detach().numpy(),
            rtol=5e-2,
            atol=1e-8,
            base_path=f"{test_name}_{next(counter)}",
        )


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,input_dim,bias,scale,dtype,param_dtype",
    [
        (2, 16, 32, True, True, "float32", "float32"),  # With bias and scale
        (2, 16, 32, False, True, "float32", "float32"),  # Without bias
        (2, 16, 32, True, False, "float32", "float32"),  # Without scale
        (2, 16, 32, True, True, "float16", "float32"),  # Mixed precision
        (1, 1, 1, True, True, "float32", "float32"),  # Minimal dimensions
        (8, 32, 64, True, True, "float32", "float32"),  # Large dimensions
    ],
    ids=[
        "basic",
        "no-bias",
        "no-scale",
        "mixed-precision",
        "minimal-dims",
        "large-dims",
    ],
)
def test_layernorm(batch_size, seq_len, input_dim, bias, scale, dtype, param_dtype, seed, rng, request):
    """Test LayerNorm with various configurations and verify NNX, Linen, and
    PyTorch implementations match.

    Args:
        batch_size: Number of samples in batch
        seq_len: Length of sequence
        input_dim: Input dimension
        bias: Whether to use bias
        scale: Whether to use scale
        dtype: Data type for computation
        param_dtype: Data type for parameters
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
    config = LayerNormConfig(
        input_dim=input_dim,
        bias=bias,
        scale=scale,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    # Create models
    nnx_model = NNXLayerNorm(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchLayerNorm(config)
    linen_model = LinenLayerNorm(config)

    # Create input
    x = np.random.randn(batch_size, seq_len, input_dim).astype(dtype)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Convert parameters from NNX to PyTorch
    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.cpu().detach().numpy(),
        rtol=5e-2,
        atol=1e-3,
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
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=5e-2,
        atol=1e-3,
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
        assert_allclose_with_plot(
            np.array(nnx_grad),
            torch_grad.cpu().detach().numpy(),
            rtol=5e-2,
            atol=1e-3,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(nnx_grad),
            np.array(linen_grad),
            rtol=5e-2,
            atol=1e-3,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(linen_grad),
            torch_grad.cpu().detach().numpy(),
            rtol=5e-2,
            atol=1e-3,
            base_path=f"{test_name}_{next(counter)}",
        )


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,input_dim,bias,scale,dtype,param_dtype",
    [
        (2, 16, 32, False, True, "float32", "float32"),  # With scale, no bias
        (2, 16, 32, True, True, "float32", "float32"),  # With scale and bias
        (2, 16, 32, False, False, "float32", "float32"),  # Without scale or bias
        (2, 16, 32, True, False, "float32", "float32"),  # With bias, no scale
        (2, 16, 32, True, True, "float16", "float32"),  # Mixed precision
        (1, 1, 1, True, True, "float32", "float32"),  # Minimal dimensions
        (8, 32, 64, True, True, "float32", "float32"),  # Large dimensions
    ],
    ids=[
        "with-scale-no-bias",
        "with-scale-and-bias",
        "no-scale-no-bias",
        "with-bias-no-scale",
        "mixed-precision",
        "minimal-dims",
        "large-dims",
    ],
)
def test_rmsnorm(batch_size, seq_len, input_dim, bias, scale, dtype, param_dtype, seed, rng, request):
    """Test RMSNorm with various configurations and verify NNX, Linen, and
    PyTorch implementations match.

    Args:
        batch_size: Number of samples in batch
        seq_len: Length of sequence
        input_dim: Input dimension
        scale: Whether to use scale
        dtype: Data type for computation
        param_dtype: Data type for parameters
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
    config = RMSNormConfig(
        input_dim=input_dim,
        bias=bias,
        scale=scale,
        dtype=dtype,
        param_dtype=param_dtype,
    )

    # Create models
    nnx_model = NNXRMSNorm(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchRMSNorm(config)
    linen_model = LinenRMSNorm(config)

    # Create input
    x = np.random.randn(batch_size, seq_len, input_dim).astype(dtype)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize models
    nnx.bridge.lazy_init(nnx_model, nnx_x)
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Convert parameters from NNX to PyTorch
    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.cpu().detach().numpy(),
        rtol=5e-2,
        atol=1e-3,
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
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=5e-2,
        atol=1e-3,
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
        assert_allclose_with_plot(
            np.array(nnx_grad),
            torch_grad.cpu().detach().numpy(),
            rtol=5e-2,
            atol=1e-3,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(nnx_grad),
            np.array(linen_grad),
            rtol=5e-2,
            atol=1e-3,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(linen_grad),
            torch_grad.cpu().detach().numpy(),
            rtol=5e-2,
            atol=1e-3,
            base_path=f"{test_name}_{next(counter)}",
        )
