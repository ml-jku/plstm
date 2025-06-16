import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
import itertools

from plstm.config.vision_model import pLSTMVisionModelConfig
from plstm.nnx.vision_model import pLSTMVisionModel as NNXVisionModel
from plstm.torch.vision_model import pLSTMVisionModel as TorchVisionModel
from plstm.linen.vision_model import pLSTMVisionModel as LinenVisionModel
from plstm.nnx.util import count_parameters as count_parameters_nnx
from plstm.torch.util import count_parameters as count_parameters_torch
from plstm.config.vision_blocks import pLSTMVisionBlockConfig1
from plstm.config.block_stack import BlockStackConfig

from plstm.conversion import convert_parameters_nnx_to_torch
from plstm.conversion import convert_parameters_nnx_to_linen


@pytest.mark.parametrize("seed", [0, 42, 123])
@pytest.mark.parametrize(
    "batch_size,num_channels,resolution,patch_size,dim,num_blocks,num_heads,pooling,channels_first",
    [
        (2, 3, (32, 32), 8, 96, 4, 8, "corners", True),  # Basic case
        (2, 3, (64, 64), 16, 128, 6, 8, "center", False),  # Larger dimensions
        (1, 3, (16, 16), 4, 64, 2, 8, "corners", True),  # Minimal dimensions
        (4, 3, (48, 48), 8, 192, 8, 8, "center", True),  # More blocks
    ],
)
def test_vision_model(
    batch_size,
    num_channels,
    resolution,
    patch_size,
    dim,
    num_blocks,
    num_heads,
    pooling,
    channels_first,
    seed,
    request,
):
    """Test pLSTMVisionModel layer equivalence between NNX and PyTorch
    implementations."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config and models
    config = pLSTMVisionModelConfig(
        dim=dim,
        block_stack=BlockStackConfig(
            block=pLSTMVisionBlockConfig1(
                input_dim=dim,
                num_heads=num_heads,
                dtype="float32",
                param_dtype="float32",
            ),
            input_dim=dim,
            num_blocks=num_blocks,
            dtype="float32",
            param_dtype="float32",
        ),
        num_channels=num_channels,
        resolution=resolution,
        channels_first=channels_first,
        patch_size=patch_size,
        num_blocks=num_blocks,
        num_heads=num_heads,
        pooling=pooling,
        dtype="float32",
        param_dtype="float32",
        output_shape=[1000],  # Standard ImageNet classes
    )

    nnx_model = NNXVisionModel(config=config, rngs=nnx.Rngs(rng))
    torch_model = TorchVisionModel(config=config)
    linen_model = LinenVisionModel(config=config)

    # Create input
    if channels_first:
        x = np.random.randn(batch_size, num_channels, *resolution).astype(np.float32)
    else:
        x = np.random.randn(batch_size, *resolution, num_channels).astype(np.float32)

    nnx.bridge.lazy_init(nnx_model, x)
    linen_variables = linen_model.init(rng, jnp.array(x))
    assert count_parameters_nnx(nnx_model) == count_parameters_torch(torch_model)
    # Convert parameters from NNX to PyTorch
    convert_parameters_nnx_to_torch(nnx_model, torch_model)
    linen_variables = convert_parameters_nnx_to_linen(
        nnx_model, linen_model, variables=linen_variables, exmp_input=jnp.array(x)
    )

    # Test forward pass
    nnx_out = nnx_model(jnp.array(x))
    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x))
    linen_out = linen_model.apply(linen_variables, jnp.array(x))

    # Compare outputs
    assert_allclose_with_plot(
        np.array(nnx_out), torch_out.numpy(), rtol=5e-3, atol=1e-2, base_path=f"{test_name}_{next(counter)}"
    )

    # Compare outputs
    assert_allclose_with_plot(
        np.array(nnx_out), np.array(linen_out), rtol=5e-3, atol=1e-2, base_path=f"{test_name}_{next(counter)}"
    )


def test_invalid_configs():
    """Test invalid configurations raise appropriate errors."""
    # Test invalid patch size
    with pytest.raises(AssertionError):
        config = pLSTMVisionModelConfig(
            dim=64,
            input_shape=(3, 32, 32),
            resolution=(32, 32),
            patch_size=7,  # Invalid: not divisible into input shape
            num_blocks=4,
            num_heads=8,
            channels_first=True,
        )
        NNXVisionModel(config=config, rngs=nnx.Rngs(0))
        TorchVisionModel(config=config)

    # Test invalid number of heads
    with pytest.raises(AssertionError):
        config = pLSTMVisionModelConfig(
            dim=64,
            input_shape=(3, 32, 32),
            patch_size=8,
            num_blocks=4,
            num_heads=5,  # Invalid: dim not divisible by num_heads
        )
        NNXVisionModel(config=config, rngs=nnx.Rngs(0))
        TorchVisionModel(config=config)

    # Test invalid patch size for Linen
    with pytest.raises(AssertionError):
        config = pLSTMVisionModelConfig(
            dim=64,
            input_shape=(3, 32, 32),
            patch_size=7,  # Invalid: not divisible into input shape
            num_blocks=4,
            num_heads=8,
        )
        linen_model = LinenVisionModel(config=config)
        linen_model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))
