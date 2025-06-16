import jax
import time
import pytest
from plstm.nnx_dummy import nnx

from plstm.config.vision_model import pLSTMVisionModelConfig
from plstm.config.vision_blocks import pLSTMVisionBlockConfig1
from plstm.config.block_stack import BlockStackConfig
from plstm.config.norm import LayerNormConfig
from plstm.config.vision_util import VitPatchEmbedConfig

# Import both implementations
from plstm.linen.vision_model import pLSTMVisionModel as LinenpLSTMVisionModel
from plstm.nnx.vision_model import pLSTMVisionModel as NNXpLSTMVisionModel


from plstm.nnx_dummy import _NNX_IS_DUMMY

has_nnx = not _NNX_IS_DUMMY


def create_test_configs(dim=128, num_heads=4, num_blocks=2, resolution=(224, 224)):
    """Create test configurations for both model implementations."""
    # Common parameters
    patch_size = 16
    seqlens = [resolution[i] // patch_size for i in range(len(resolution))]
    num_patches = seqlens[0] * seqlens[1]

    # Create block config
    block_config = pLSTMVisionBlockConfig1(
        input_dim=dim,
        num_heads=num_heads,
    )

    # Create block stack config
    block_stack_config = BlockStackConfig(
        block=block_config,
        input_dim=dim,
        num_blocks=num_blocks,
    )

    # Create norm config
    norm_config = LayerNormConfig(
        input_dim=dim,
    )

    # Create patch embed configs
    vit_patch_embed_config = VitPatchEmbedConfig(
        patch_size=patch_size,
        num_channels=3,
        resolution=resolution,
        dim=dim,
    )

    # Create model configs
    config = pLSTMVisionModelConfig(
        dim=dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        resolution=resolution,
        patch_size=patch_size,
        seqlens=seqlens,
        num_patches=num_patches,
        block_stack=block_stack_config,
        norm=norm_config,
        patch_embed=vit_patch_embed_config,
    )

    return config


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
def test_vision_model_shapes():
    """Test that both implementations produce the same output shapes."""
    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    # Create configs
    config = create_test_configs()

    # Create input data
    batch_size = 2
    resolution = config.resolution
    num_channels = config.num_channels

    # Create input with channels first (as expected by the models)
    x = jax.random.normal(key, (batch_size, *resolution, num_channels))

    # Initialize the NNX model
    nnx_model = NNXpLSTMVisionModel(config, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)))
    nnx.bridge.lazy_init(nnx_model, x)

    # Initialize the Linen model
    linen_model = LinenpLSTMVisionModel(config)
    variables = linen_model.init(subkey, x)

    # Run inference
    print("Running NNX model...")
    start_time = time.time()
    nnx_output = nnx_model(x)
    nnx_time = time.time() - start_time

    print("Running Linen model...")
    start_time = time.time()
    linen_output = linen_model.apply(variables, x)
    linen_time = time.time() - start_time

    # Print shapes and timing
    print(f"Input shape: {x.shape}")
    print(f"NNX output shape: {nnx_output.shape}")
    print(f"Linen output shape: {linen_output.shape}")
    print(f"NNX time: {nnx_time:.4f}s")
    print(f"Linen time: {linen_time:.4f}s")
    print(f"Time ratio (Linen/NNX): {linen_time / nnx_time:.2f}x")

    # Check that shapes match
    assert (
        nnx_output.shape == linen_output.shape
    ), f"Output shapes don't match: {nnx_output.shape} vs {linen_output.shape}"

    # Return the outputs for further inspection if needed
    return nnx_output, linen_output


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
def test_vision_model_memory():
    """Test memory usage of both implementations."""
    import tracemalloc

    # Set random seed for reproducibility
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    # Create configs
    config = create_test_configs()

    # Create input data
    batch_size = 2
    resolution = config.resolution
    num_channels = config.num_channels

    # Create input with channels first (as expected by the models)
    x = jax.random.normal(key, (batch_size, *resolution, num_channels))

    # Test NNX model memory usage
    tracemalloc.start()
    nnx_model = NNXpLSTMVisionModel(config, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)))
    nnx.bridge.lazy_init(nnx_model, x)

    nnx_model(x)

    nnx_current, nnx_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Test Linen model memory usage
    tracemalloc.start()
    linen_model = LinenpLSTMVisionModel(config)
    variables = linen_model.init(subkey, x)
    linen_model.apply(variables, x)
    linen_current, linen_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Print memory usage
    print(f"NNX peak memory: {nnx_peak / 1024 / 1024:.2f} MB")
    print(f"Linen peak memory: {linen_peak / 1024 / 1024:.2f} MB")
    print(f"Memory ratio (Linen/NNX): {linen_peak / nnx_peak:.2f}x")

    return nnx_peak, linen_peak


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
def test_larger_models():
    """Test with larger models to see if differences become more pronounced."""
    # Test with different model sizes
    sizes = [
        {"dim": 128, "num_heads": 4, "num_blocks": 2, "name": "tiny"},
        {"dim": 256, "num_heads": 4, "num_blocks": 6, "name": "small"},
        {"dim": 512, "num_heads": 8, "num_blocks": 12, "name": "medium"},
    ]

    results = []

    for size in sizes:
        print(f"\nTesting {size['name']} model:")
        dim, num_heads, num_blocks = size["dim"], size["num_heads"], size["num_blocks"]

        # Set random seed for reproducibility
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)

        # Create configs
        config = create_test_configs(dim=dim, num_heads=num_heads, num_blocks=num_blocks)

        # Create input data
        batch_size = 2
        resolution = config.resolution
        num_channels = config.num_channels

        # Create input with channels first (as expected by the models)
        x = jax.random.normal(key, (batch_size, *resolution, num_channels))

        # Initialize the NNX model
        nnx_model = NNXpLSTMVisionModel(config, rngs=nnx.Rngs(params=jax.random.PRNGKey(0)))
        nnx.bridge.lazy_init(nnx_model, x)

        # Initialize the Linen model
        linen_model = LinenpLSTMVisionModel(config)
        variables = linen_model.init(subkey, x)

        # Run inference and measure time
        start_time = time.time()
        nnx_model(x)
        nnx_time = time.time() - start_time

        start_time = time.time()
        linen_model.apply(variables, x)
        linen_time = time.time() - start_time

        # Print results
        print(f"NNX time: {nnx_time:.4f}s")
        print(f"Linen time: {linen_time:.4f}s")
        print(f"Time ratio (Linen/NNX): {linen_time / nnx_time:.2f}x")

        results.append(
            {"size": size["name"], "nnx_time": nnx_time, "linen_time": linen_time, "ratio": linen_time / nnx_time}
        )

    # Print summary
    print("\nSummary:")
    for result in results:
        print(f"{result['size']}: Linen is {result['ratio']:.2f}x slower than NNX")

    return results


if __name__ == "__main__":
    print("Testing vision model shapes...")
    nnx_output, linen_output = test_vision_model_shapes()

    print("\nTesting vision model memory usage...")
    nnx_peak, linen_peak = test_vision_model_memory()

    print("\nTesting with larger models...")
    results = test_larger_models()
