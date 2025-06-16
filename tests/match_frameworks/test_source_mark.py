import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools

from plstm.config.source_mark_layer import (
    SourceLayerConfig,
    MarkLayerConfig,
    DirectLayerConfig,
)
from plstm.nnx.source_mark_layer import (
    SourceLayer as NNXSourceLayer,
    MarkLayer as NNXMarkLayer,
    DirectLayer as NNXDirectLayer,
)
from plstm.torch.source_mark_layer import (
    SourceLayer as TorchSourceLayer,
    MarkLayer as TorchMarkLayer,
    DirectLayer as TorchDirectLayer,
)
from plstm.linen.source_mark_layer import (
    SourceLayer as LinenSourceLayer,
    MarkLayer as LinenMarkLayer,
    DirectLayer as LinenDirectLayer,
)
from plstm.conversion import (
    convert_parameters_nnx_to_linen,
)
from plstm.config.initialization import DiagonalInitConfig


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,JK,JV,JT,input_dim,weight",
    [
        (2, 16, 4, 2, 2, 4, 32, False),  # Basic case with eye init
        (2, 16, 4, 2, 2, 4, 32, False),  # Basic case with ones init
        (2, 16, 4, 2, 2, 4, 32, True),  # With weights
        (1, 8, 1, 1, 1, 1, 16, False),  # Minimal dimensions
        (8, 32, 8, 4, 4, 8, 64, True),  # Large dimensions
    ],
    ids=[
        "basic-eye",
        "basic-ones",
        "with-weights",
        "minimal-dims",
        "large-dims",
    ],
)
def test_source_layer(
    batch_size,
    seq_len,
    num_heads,
    JK,
    JV,
    JT,
    input_dim,
    weight,
    seed,
    request,
):
    """Test SourceLayer with various configurations and verify NNX, Linen, and
    PyTorch implementations match."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create config
    config = SourceLayerConfig(
        num_heads=num_heads,
        JK=JK,
        JV=JV,
        JT=JT,
        input_dim=input_dim,
        weight=weight,
    )

    # Create models
    nnx_model = NNXSourceLayer(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchSourceLayer(config)
    linen_model = LinenSourceLayer(config)

    # Create input
    x = np.random.randn(batch_size, seq_len, num_heads, input_dim).astype(np.float32)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize linen model
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Copy parameters from NNX to PyTorch
    torch_model.bias.data = torch.from_numpy(np.array(nnx_model.bias))
    if weight:
        torch_model.weight.data = torch.from_numpy(np.array(nnx_model.weight))

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.cpu().detach().numpy(),
        rtol=5e-3,
        atol=5e-3,
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
        rtol=5e-3,
        atol=5e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=5e-3,
        atol=5e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    if weight:
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
            rtol=5e-3,
            atol=5e-3,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(nnx_grad),
            np.array(linen_grad),
            rtol=5e-3,
            atol=5e-3,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(linen_grad),
            torch_grad.cpu().detach().numpy(),
            rtol=5e-3,
            atol=5e-3,
            base_path=f"{test_name}_{next(counter)}",
        )


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,JQ,JO,JT,input_dim,weight",
    [
        (2, 16, 4, 2, 2, 4, 32, False),  # Basic case with eye init
        (2, 16, 4, 2, 2, 4, 32, False),  # Basic case with ones init
        (2, 16, 4, 2, 2, 4, 32, True),  # With weights
        (1, 8, 1, 1, 1, 1, 16, False),  # Minimal dimensions
        (8, 32, 8, 4, 4, 8, 64, True),  # Large dimensions
    ],
    ids=[
        "basic-eye",
        "basic-ones",
        "with-weights",
        "minimal-dims",
        "large-dims",
    ],
)
def test_mark_layer(
    batch_size,
    seq_len,
    num_heads,
    JQ,
    JO,
    JT,
    input_dim,
    weight,
    seed,
    request,
):
    """Test MarkLayer with various configurations and verify NNX, Linen, and
    PyTorch implementations match."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create config
    config = MarkLayerConfig(
        num_heads=num_heads,
        JQ=JQ,
        JO=JO,
        JT=JT,
        input_dim=input_dim,
        weight=weight,
    )

    # Create models
    nnx_model = NNXMarkLayer(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchMarkLayer(config)
    linen_model = LinenMarkLayer(config)

    # Create input
    x = np.random.randn(batch_size, seq_len, num_heads, input_dim).astype(np.float32)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize linen model
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Copy parameters from NNX to PyTorch
    torch_model.bias.data = torch.from_numpy(np.array(nnx_model.bias))
    if weight:
        torch_model.weight.data = torch.from_numpy(np.array(nnx_model.weight))

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.cpu().detach().numpy(),
        rtol=5e-3,
        atol=5e-3,
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
        rtol=5e-3,
        atol=5e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=5e-3,
        atol=5e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    if weight:
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
            rtol=5e-3,
            atol=5e-3,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(nnx_grad),
            np.array(linen_grad),
            rtol=5e-3,
            atol=5e-3,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(linen_grad),
            torch_grad.cpu().detach().numpy(),
            rtol=5e-3,
            atol=5e-3,
            base_path=f"{test_name}_{next(counter)}",
        )


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,JQ,JK,JO,JV,input_dim,weight",
    [
        (2, 16, 4, 2, 2, 2, 2, 32, False),  # Basic case with eye init
        (2, 16, 4, 2, 2, 2, 2, 32, False),  # Basic case with ones init
        (2, 16, 4, 2, 2, 2, 2, 32, True),  # With weights
        (1, 8, 1, 1, 1, 1, 1, 16, False),  # Minimal dimensions
        (8, 32, 8, 4, 4, 4, 4, 64, True),  # Large dimensions
    ],
    ids=[
        "basic-eye",
        "basic-ones",
        "with-weights",
        "minimal-dims",
        "large-dims",
    ],
)
def test_direct_layer(
    batch_size,
    seq_len,
    num_heads,
    JQ,
    JK,
    JO,
    JV,
    input_dim,
    weight,
    seed,
    request,
):
    """Test DirectLayer with various configurations and verify NNX, Linen, and
    PyTorch implementations match."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    # Set random seeds
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create config
    config = DirectLayerConfig(
        num_heads=num_heads,
        JQ=JQ,
        JK=JK,
        JO=JO,
        JV=JV,
        input_dim=input_dim,
        weight=weight,
    )

    # Create models
    nnx_model = NNXDirectLayer(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchDirectLayer(config)
    linen_model = LinenDirectLayer(config)

    # Create input
    x = np.random.randn(batch_size, seq_len, num_heads, input_dim).astype(np.float32)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize linen model
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Copy parameters from NNX to PyTorch
    torch_model.bias.data = torch.from_numpy(np.array(nnx_model.bias))
    if weight:
        torch_model.weight.data = torch.from_numpy(np.array(nnx_model.weight))

    # Test forward pass for NNX and PyTorch
    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.cpu().detach().numpy(),
        rtol=5e-3,
        atol=5e-3,
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
        rtol=5e-3,
        atol=5e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=5e-3,
        atol=5e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    if weight:
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
            rtol=5e-3,
            atol=5e-3,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(nnx_grad),
            np.array(linen_grad),
            rtol=5e-3,
            atol=5e-3,
            base_path=f"{test_name}_{next(counter)}",
        )

        assert_allclose_with_plot(
            np.array(linen_grad),
            torch_grad.cpu().detach().numpy(),
            rtol=5e-3,
            atol=5e-3,
            base_path=f"{test_name}_{next(counter)}",
        )


@pytest.mark.parametrize("layer_type", ["source", "mark", "direct"])
def test_invalid_config(layer_type, request):
    """Test that invalid configurations raise appropriate errors.

    Args:
        layer_type: Type of layer ('source', 'mark', 'direct')
    """
    if layer_type == "source":
        config_class = SourceLayerConfig
        nnx_class = NNXSourceLayer
        torch_class = TorchSourceLayer
        linen_class = LinenSourceLayer
    elif layer_type == "mark":
        config_class = MarkLayerConfig
        nnx_class = NNXMarkLayer
        torch_class = TorchMarkLayer
        linen_class = LinenMarkLayer
    elif layer_type == "direct":
        config_class = DirectLayerConfig
        nnx_class = NNXDirectLayer
        torch_class = TorchDirectLayer
        linen_class = LinenDirectLayer

    rngs = nnx.Rngs(0)

    # Test invalid num_heads
    with pytest.raises(AssertionError):
        config = config_class(num_heads=-1, input_dim=32)
        nnx_class(config, rngs=rngs)
        torch_class(config)
        linen_model = linen_class(config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 4, 32)))

    # Test invalid input_dim
    with pytest.raises(AssertionError):
        config = config_class(num_heads=4, input_dim=-1)
        nnx_class(config, rngs=rngs)
        torch_class(config)
        linen_model = linen_class(config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 4, 32)))

    if layer_type == "direct":
        with pytest.raises(AssertionError):
            config = config_class(
                num_heads=4,
                input_dim=32,
                JO=2,
                JQ=2,
                JK=2,
                JV=3,
                bias_init=DiagonalInitConfig(in_axes=(-4, -3), out_axes=(-2, -1)),
            )
            nnx_class(config, rngs=rngs)
            torch_class(config)
            linen_model = linen_class(config)
            linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 4, 32)))
