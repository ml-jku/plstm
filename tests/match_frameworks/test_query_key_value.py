import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools

from plstm.config.query_key_value import QueryLayerConfig, KeyLayerConfig, ValueLayerConfig
from plstm.nnx.query_key_value import QueryLayer as NNXQueryLayer
from plstm.nnx.query_key_value import KeyLayer as NNXKeyLayer
from plstm.nnx.query_key_value import ValueLayer as NNXValueLayer
from plstm.torch.query_key_value import QueryLayer as TorchQueryLayer
from plstm.torch.query_key_value import KeyLayer as TorchKeyLayer
from plstm.torch.query_key_value import ValueLayer as TorchValueLayer
from plstm.linen.query_key_value import QueryLayer as LinenQueryLayer
from plstm.linen.query_key_value import KeyLayer as LinenKeyLayer
from plstm.linen.query_key_value import ValueLayer as LinenValueLayer
from plstm.conversion import (
    convert_parameters_nnx_to_linen,
)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,sub_heads,input_dim,DK,J,bias",
    [
        (2, 16, 4, 2, 32, 32, 1, True),  # Basic case
        (2, 16, 4, 1, 32, 32, 1, False),  # Without bias
        (1, 8, 1, 1, 16, 16, 1, True),  # Minimal dimensions with ones init
        (2, 16, 4, 2, 32, 16, 2, True),  # Different DK and J values
    ],
    ids=[
        "basic",
        "no-bias",
        "minimal-ones",
        "different-dims",
    ],
)
def test_query_layer(batch_size, seq_len, num_heads, sub_heads, input_dim, DK, J, bias, seed, rng, request):
    """Test QueryLayer with various configurations and verify NNX, Linen, and
    PyTorch implementations match.

    Args:
        batch_size: Number of samples in batch
        seq_len: Length of sequence
        num_heads: Number of attention heads
        sub_heads: Number of sub-heads
        input_dim: Input dimension
        DK: Key dimension
        J: JQ dimension
        bias: Whether to use bias
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
    config = QueryLayerConfig(
        num_heads=num_heads,
        sub_heads=sub_heads,
        input_dim=input_dim,
        DK=DK,
        JQ=J,
        bias=bias,
        dtype="float32",
        param_dtype="float32",
    )

    # Create models
    nnx_model = NNXQueryLayer(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchQueryLayer(config)
    linen_model = LinenQueryLayer(config)

    # Create input
    x = np.random.randn(batch_size, seq_len, num_heads, input_dim).astype(np.float32)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize linen model
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Copy parameters from NNX to PyTorch
    torch_model.weight.data = torch.from_numpy(np.array(nnx_model.weight))
    if bias:
        torch_model.bias.data = torch.from_numpy(np.array(nnx_model.bias))

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.cpu().detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
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
        rtol=2e-3,
        atol=2e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
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
        rtol=2e-3,
        atol=2e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(nnx_grad),
        np.array(linen_grad),
        rtol=2e-3,
        atol=2e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(linen_grad),
        torch_grad.cpu().detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,sub_heads,input_dim,DK,J,bias",
    [
        (2, 16, 4, 2, 32, 32, 1, True),  # Basic case
        (2, 16, 4, 1, 32, 32, 1, False),  # Without bias
        (1, 8, 1, 1, 16, 16, 1, True),  # Minimal dimensions with ones init
        (2, 16, 4, 2, 32, 16, 2, True),  # Different DK and J values
    ],
    ids=[
        "basic",
        "no-bias",
        "minimal-ones",
        "different-dims",
    ],
)
def test_key_layer(batch_size, seq_len, num_heads, sub_heads, input_dim, DK, J, bias, seed, rng, request):
    """Test KeyLayer with various configurations and verify NNX, Linen, and
    PyTorch implementations match."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create config
    config = KeyLayerConfig(
        num_heads=num_heads,
        sub_heads=sub_heads,
        input_dim=input_dim,
        DK=DK,
        JK=J,
        bias=bias,
        dtype="float32",
        param_dtype="float32",
    )

    # Create models
    nnx_model = NNXKeyLayer(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchKeyLayer(config)
    linen_model = LinenKeyLayer(config)

    # Create input
    x = np.random.randn(batch_size, seq_len, num_heads, input_dim).astype(np.float32)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize linen model
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Copy parameters from NNX to PyTorch
    torch_model.weight.data = torch.from_numpy(np.array(nnx_model.weight))
    if bias:
        torch_model.bias.data = torch.from_numpy(np.array(nnx_model.bias))

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.cpu().detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
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
        rtol=2e-3,
        atol=2e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
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
        rtol=2e-3,
        atol=2e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(nnx_grad),
        np.array(linen_grad),
        rtol=2e-3,
        atol=2e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(linen_grad),
        torch_grad.cpu().detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,sub_heads,input_dim,DV,J,bias",
    [
        (2, 16, 4, 2, 32, 32, 1, True),  # Basic case
        (2, 16, 4, 1, 32, 32, 1, False),  # Without bias
        (1, 8, 1, 1, 16, 16, 1, True),  # Minimal dimensions with ones init
        (2, 16, 4, 2, 32, 16, 2, True),  # Different DV and J values
    ],
    ids=[
        "basic",
        "no-bias",
        "minimal-ones",
        "different-dims",
    ],
)
def test_value_layer(batch_size, seq_len, num_heads, sub_heads, input_dim, DV, J, bias, seed, rng, request):
    """Test ValueLayer with various configurations and verify NNX, Linen, and
    PyTorch implementations match."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create config
    config = ValueLayerConfig(
        num_heads=num_heads,
        sub_heads=sub_heads,
        input_dim=input_dim,
        DV=DV,
        JV=J,
        bias=bias,
        dtype="float32",
        param_dtype="float32",
    )

    # Create models
    nnx_model = NNXValueLayer(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchValueLayer(config)
    linen_model = LinenValueLayer(config)

    # Create input
    x = np.random.randn(batch_size, seq_len, num_heads, input_dim).astype(np.float32)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize linen model
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Copy parameters from NNX to PyTorch
    torch_model.weight.data = torch.from_numpy(np.array(nnx_model.weight))
    if bias:
        torch_model.bias.data = torch.from_numpy(np.array(nnx_model.bias))

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.cpu().detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
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
        rtol=2e-3,
        atol=2e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
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
        rtol=2e-3,
        atol=2e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(nnx_grad),
        np.array(linen_grad),
        rtol=2e-3,
        atol=2e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(linen_grad),
        torch_grad.cpu().detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.parametrize("layer_type", ["query", "key", "value"])
def test_invalid_config(layer_type, request):
    """Test that invalid configurations raise appropriate errors.

    Args:
        layer_type: Type of layer ('query', 'key', or 'value')
    """
    if layer_type == "query":
        config_class = QueryLayerConfig
        nnx_class = NNXQueryLayer
        torch_class = TorchQueryLayer
        linen_class = LinenQueryLayer
        D_param = "DK"
        J_param = "JQ"
    elif layer_type == "key":
        config_class = KeyLayerConfig
        nnx_class = NNXKeyLayer
        torch_class = TorchKeyLayer
        linen_class = LinenKeyLayer
        D_param = "DK"
        J_param = "JK"
    else:  # value
        config_class = ValueLayerConfig
        nnx_class = NNXValueLayer
        torch_class = TorchValueLayer
        linen_class = LinenValueLayer
        D_param = "DV"
        J_param = "JV"

    rngs = nnx.Rngs(0)

    # Test invalid num_heads
    with pytest.raises(AssertionError):
        config = config_class(num_heads=-1, input_dim=32, **{D_param: 32, J_param: 1})
        nnx_class(config, rngs=rngs)
        torch_class(config)
        linen_model = linen_class(config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 4, 32)))

    # Test invalid input_dim
    with pytest.raises(AssertionError):
        config = config_class(num_heads=4, input_dim=-1, **{D_param: 32, J_param: 1})
        nnx_class(config, rngs=rngs)
        torch_class(config)
        linen_model = linen_class(config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 4, 32)))

    # Test invalid D dimension
    with pytest.raises(AssertionError):
        config = config_class(num_heads=4, input_dim=32, **{D_param: -1, J_param: 1})
        nnx_class(config, rngs=rngs)
        torch_class(config)
        linen_model = linen_class(config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 4, 32)))
