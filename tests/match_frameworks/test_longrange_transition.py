import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools


from plstm.config.longrange_transition_layer import (
    LongRangeTransitionLayerConfig,
)
from plstm.nnx.longrange_transition_layer import (
    LongRangeTransitionLayer as NNXLongRangeTransitionLayer,
)
from plstm.torch.longrange_transition_layer import (
    LongRangeTransitionLayer as TorchLongRangeTransitionLayer,
)
from plstm.linen.longrange_transition_layer import (
    LongRangeTransitionLayer as LinenLongRangeTransitionLayer,
)
from plstm.conversion import (
    convert_parameters_nnx_to_linen,
)


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,transition_dim,input_dim,symmetric,normalization_mode,eigenvalue_representation",
    [
        (2, 16, 4, 8, 32, False, "exponential_orthogonalization", "logsigmoid"),  # Basic case
        (2, 16, 4, 8, 32, True, "exponential_orthogonalization", "logsigmoid"),  # Symmetric
        (1, 8, 1, 4, 16, False, "qr", "logsigmoid"),  # Minimal dimensions with QR
        # (8, 32, 8, 16, 64, False, "eigenvalue_restriction", "expexp"),  # Large dims with eigenvalue restriction
        (2, 16, 4, 1, 32, False, "exponential_orthogonalization", "logsigmoid"),  # Single transition dim
    ],
    ids=[
        "basic",
        "symmetric",
        "minimal-qr",
        # "large-eigenval", this one does not allow backprop
        "single-transition",
    ],
)
def test_longrange_transition_layer(
    batch_size,
    seq_len,
    num_heads,
    transition_dim,
    input_dim,
    symmetric,
    normalization_mode,
    eigenvalue_representation,
    seed,
    request,
):
    """Test LongRangeTransitionLayer with various configurations and verify
    NNX, Linen, and PyTorch implementations match."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)
    rng_nnx, rng_linen = jax.random.split(rng)

    # Create config
    config = LongRangeTransitionLayerConfig(
        num_heads=num_heads,
        transition_dim=transition_dim,
        input_dim=input_dim,
        symmetric=symmetric,
        normalization_mode=normalization_mode,
        eigenvalue_representation=eigenvalue_representation,
        param_dtype="float32",
        dtype="float32",
    )

    # Create models
    nnx_model = NNXLongRangeTransitionLayer(config, rngs=nnx.Rngs(rng_nnx))
    torch_model = TorchLongRangeTransitionLayer(config)
    linen_model = LinenLongRangeTransitionLayer(config)

    # Create input
    x = np.random.randn(batch_size, seq_len, num_heads, input_dim).astype(np.float32)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize linen model
    linen_variables = linen_model.init(rng_linen, linen_x)

    # Copy parameters from NNX to PyTorch
    if transition_dim > 1:
        torch_model.inproj_bias.data = torch.from_numpy(np.array(nnx_model.inproj_bias))
        torch_model.inproj_weight.data = torch.from_numpy(np.array(nnx_model.inproj_weight))
        if not symmetric:
            torch_model.outproj_bias.data = torch.from_numpy(np.array(nnx_model.outproj_bias))
            torch_model.outproj_weight.data = torch.from_numpy(np.array(nnx_model.outproj_weight))

    torch_model.eigenvalues_bias.data = torch.from_numpy(np.array(nnx_model.eigenvalues_bias))
    torch_model.eigenvalues_weight.data = torch.from_numpy(np.array(nnx_model.eigenvalues_weight))

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


def test_invalid_config(request):
    """Test that invalid configurations raise appropriate errors."""
    config_class = LongRangeTransitionLayerConfig
    nnx_class = NNXLongRangeTransitionLayer
    torch_class = TorchLongRangeTransitionLayer
    linen_class = LinenLongRangeTransitionLayer
    rngs = nnx.Rngs(0)

    # Test invalid num_heads
    with pytest.raises(AssertionError):
        config = config_class(num_heads=-1, input_dim=32, transition_dim=8)
        nnx_class(config, rngs=rngs)
        torch_class(config)
        linen_model = linen_class(config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 4, 32)))

    # Test invalid input_dim
    with pytest.raises(AssertionError):
        config = config_class(num_heads=4, input_dim=-1, transition_dim=8)
        nnx_class(config, rngs=rngs)
        torch_class(config)
        linen_model = linen_class(config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 4, 32)))

    # Test invalid transition_dim
    with pytest.raises(AssertionError):
        config = config_class(num_heads=4, input_dim=32, transition_dim=-1)
        nnx_class(config, rngs=rngs)
        torch_class(config)
        linen_model = linen_class(config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 4, 32)))

    # Test invalid normalization mode
    with pytest.raises(AssertionError):
        config = config_class(
            num_heads=4,
            input_dim=32,
            transition_dim=8,
            normalization_mode="invalid",
        )
        nnx_class(config, rngs=rngs)
        torch_class(config)
        linen_model = linen_class(config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 10, 4, 32)))
