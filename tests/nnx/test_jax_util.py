"""Test JAX utility functions."""

import itertools
import pytest
import jax
import jax.numpy as jnp
from plstm.nnx_dummy import nnx

from plstm.nnx.util import (
    module_named_params,
    rev_cumsum_off,
    cumsum_off,
    rev_cumsum,
)
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot


class CustomModule(nnx.Module):
    """Simple custom module for testing parameter traversal."""

    def __init__(self, *, rngs: nnx.Rngs):
        super().__init__()
        self.weight = nnx.Param(nnx.initializers.normal()(rngs.params(), (4, 4), jnp.float32))
        self.bias = nnx.Param(nnx.initializers.zeros_init()(rngs.params(), (4,), jnp.float32))
        self.scale = nnx.Param(nnx.initializers.ones_init()(rngs.params(), (1,), jnp.float32))


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


def test_module_named_params_custom(rng, request):
    """Test parameter traversal with custom module."""
    # Create custom module
    module = CustomModule(rngs=nnx.Rngs(rng))

    # Get parameters
    params = dict(module_named_params(module))

    # Verify all parameters are found
    assert set(params.keys()) == {"weight", "bias", "scale"}

    # Verify parameter shapes
    assert params["weight"].shape == (4, 4)
    assert params["bias"].shape == (4,)
    assert params["scale"].shape == (1,)


def test_module_named_params_linear(rng, request):
    """Test parameter traversal with nnx.Linear."""
    # Create linear layer
    module = nnx.Linear(
        in_features=32,
        out_features=64,
        use_bias=True,
        rngs=nnx.Rngs(rng),
    )

    # Get parameters
    params = dict(module_named_params(module))

    # Verify all parameters are found
    assert set(params.keys()) == {"kernel", "bias"}

    # Verify parameter shapes
    assert params["kernel"].shape == (32, 64)
    assert params["bias"].shape == (64,)


def test_module_named_params_linear_no_bias(rng, request):
    """Test parameter traversal with nnx.Linear without bias."""
    # Create linear layer without bias
    module = nnx.Linear(
        in_features=32,
        out_features=64,
        use_bias=False,
        rngs=nnx.Rngs(rng),
    )

    # Get parameters
    params = dict(module_named_params(module))

    # Verify only kernel is found
    assert set(params.keys()) == {"kernel"}

    # Verify parameter shape
    assert params["kernel"].shape == (32, 64)


def test_module_named_params_nested(request):
    """Test parameter traversal doesn't recurse into nested modules."""
    rng = jax.random.PRNGKey(0)

    class NestedModule(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs):
            super().__init__()
            self.outer = nnx.Param(nnx.initializers.ones_init()(rngs.params(), (1,), jnp.float32))
            self.linear = nnx.Linear(32, 64, rngs=rngs)

    # Create nested module
    module = NestedModule(rngs=nnx.Rngs(rng))

    # Get parameters
    params = dict(module_named_params(module))

    # Verify only outer parameter is found (no recursion into linear)
    assert set(params.keys()) == {"outer"}
    assert params["outer"].shape == (1,)


@pytest.mark.parametrize(
    "shape",
    [
        (5,),
        (3, 4),
        (2, 3, 4),
        (2, 1, 3, 4),
    ],
)
def test_cumsum_functions_shape(shape, request):
    """Test that cumsum functions preserve input shape."""
    x = jnp.ones(shape)

    # Test rev_cumsum_off
    y1 = rev_cumsum_off(x)
    assert y1.shape == x.shape

    # Test cumsum_off
    y2 = cumsum_off(x)
    assert y2.shape == x.shape

    # Test rev_cumsum
    y3 = rev_cumsum(x)
    assert y3.shape == x.shape


def test_cumsum_functions_values(request):
    """Test that cumsum functions compute correct values."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()

    # Test rev_cumsum_off
    expected_rev_off = jnp.array([14.0, 12.0, 9.0, 5.0, 0.0])
    assert_allclose_with_plot(rev_cumsum_off(x), expected_rev_off, rtol=1e-5, base_path=f"{test_name}_{next(counter)}")

    # Test cumsum_off
    expected_off = jnp.array([0.0, 1.0, 3.0, 6.0, 10.0])
    assert_allclose_with_plot(cumsum_off(x), expected_off, rtol=1e-5, base_path=f"{test_name}_{next(counter)}")

    # Test rev_cumsum
    expected_rev = jnp.array([15.0, 14.0, 12.0, 9.0, 5.0])
    assert_allclose_with_plot(rev_cumsum(x), expected_rev, rtol=1e-5, base_path=f"{test_name}_{next(counter)}")


def test_cumsum_functions_2d(request):
    """Test cumsum functions with 2D input."""
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()

    # Test rev_cumsum_off
    expected_rev_off = jnp.array([[5.0, 3.0, 0.0], [11.0, 6.0, 0.0]])
    assert_allclose_with_plot(rev_cumsum_off(x), expected_rev_off, rtol=1e-5, base_path=f"{test_name}_{next(counter)}")

    # Test cumsum_off
    expected_off = jnp.array([[0.0, 1.0, 3.0], [0.0, 4.0, 9.0]])
    assert_allclose_with_plot(cumsum_off(x), expected_off, rtol=1e-5, base_path=f"{test_name}_{next(counter)}")

    # Test rev_cumsum
    expected_rev = jnp.array([[6.0, 5.0, 3.0], [15.0, 11.0, 6.0]])
    assert_allclose_with_plot(rev_cumsum(x), expected_rev, rtol=1e-5, base_path=f"{test_name}_{next(counter)}")
