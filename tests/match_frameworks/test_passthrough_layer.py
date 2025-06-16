import pytest
from plstm.config.passthrough import PassthroughLayerConfig
from plstm.nnx.passthrough import PassthroughLayer as NNXPassthroughLayer
from plstm.torch.passthrough import PassthroughLayer as TorchPassthroughLayer
from plstm.linen.passthrough import PassthroughLayer as LinenPassthroughLayer
import jax
import jax.numpy as jnp
import torch
from flax.nnx import Rngs
import numpy as np
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
from plstm.conversion import convert_parameters_nnx_to_torch
from plstm.conversion import (
    convert_parameters_nnx_to_linen,
)
import itertools


@pytest.mark.parametrize("batch_size,num_heads,input_dim", [(3, 2, 16), (1, 2, 16), (1, 1, 16), (3, 1, 16)])
def test_passthrough_layer(batch_size, num_heads, input_dim, request):
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    config = PassthroughLayerConfig(
        input_dim=input_dim,
        param_dtype="float32",
        dtype="float32",
    )
    rng = jax.random.PRNGKey(0)
    rng_nnx, rng_linen = jax.random.split(rng)

    nnx_model = NNXPassthroughLayer(config, rngs=Rngs(rng_nnx))
    torch_model = TorchPassthroughLayer(config)
    linen_model = LinenPassthroughLayer(config)

    shape = (batch_size, num_heads, input_dim)
    x = np.random.randn(*shape)
    nnx_x = jnp.array(x)
    linen_x = jnp.array(x)
    torch_x = torch.from_numpy(x)

    # Initialize linen model
    linen_variables = linen_model.init(rng_linen, linen_x)

    convert_parameters_nnx_to_torch(nnx_model, torch_model)

    nnx_out = nnx_model(nnx_x)
    torch_out = torch_model(torch_x)

    # Compare NNX and PyTorch outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        torch_out.cpu().detach().numpy(),
        rtol=1e-3,
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
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Compare Linen and PyTorch outputs
    assert_allclose_with_plot(
        np.array(linen_out),
        torch_out.cpu().detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
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
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(nnx_grad),
        np.array(linen_grad),
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )

    assert_allclose_with_plot(
        np.array(linen_grad),
        torch_grad.cpu().detach().numpy(),
        rtol=1e-3,
        atol=1e-3,
        base_path=f"{test_name}_{next(counter)}",
    )


def test_invalid_config(request):
    """Test that invalid configurations raise appropriate errors in all
    frameworks."""
    with pytest.raises(AssertionError):
        config = PassthroughLayerConfig(input_dim=-1)
        linen_model = LinenPassthroughLayer(config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 1, 16)))
