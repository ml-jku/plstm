import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx
import optax

from plstm.config.lm_model import pLSTMLMModelConfig
from plstm.nnx.lm_model import pLSTMLMModel as NNXLMModel
from plstm.linen.lm_model import pLSTMLMModel as LinenLMModel
from plstm.torch.lm_model import pLSTMLMModel as TorchLMModel
from plstm.test.util import request_pytest_filepath
from plstm.test.numerics import assert_allclose_with_plot
from plstm.conversion.test import assert_parameters_match, assert_linen_nnx_parameters_match
import itertools
from plstm.conversion import convert_parameters_nnx_to_torch, convert_parameters_nnx_to_linen


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.mark.parametrize("seed", [0, 42, 123])  # Test with different random seeds
@pytest.mark.parametrize(
    "batch_size,seq_len,vocab_size,embedding_dim,num_blocks,block_type,tie_weights",
    [
        (2, 16, 1000, 32, 2, "pre_up", True),  # Basic config with tied weights
        (2, 16, 1000, 32, 2, "post_up", True),  # Basic config with post_up
        (2, 16, 1000, 32, 2, "post_up", False),  # Without tied weights
        (1, 1, 100, 16, 1, "pre_up", True),  # Minimal dimensions
        (4, 32, 5000, 64, 4, "pre_up", True),  # Large dimensions
    ],
    ids=[
        "basic-pre-tied",
        "basic-post-tied",
        "basic-post-untied",
        "minimal-dims",
        "large-dims",
    ],
)
def test_lm_model(
    batch_size, seq_len, vocab_size, embedding_dim, num_blocks, block_type, tie_weights, seed, rng, request
):
    """Test LMModel with various configurations and verify JAX and PyTorch
    implementations match."""
    test_name, counter = request_pytest_filepath(request, __file__), itertools.count()

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = jax.random.PRNGKey(seed)

    # Create config
    config = pLSTMLMModelConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_blocks=num_blocks,
        block_type=block_type,
        tie_weights=tie_weights,
        param_dtype="float32",
        dtype="float32",
    )

    # Create models
    nnx_model = NNXLMModel(config, rngs=nnx.Rngs(rng))
    linen_model = LinenLMModel(config)
    torch_model = TorchLMModel(config)

    # Create input (token ids should be integers in [0, vocab_size))
    token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32)

    # Initialize JAX model
    nnx.bridge.lazy_init(nnx_model, jnp.array(token_ids))
    linen_variables = linen_model.init(jax.random.PRNGKey(0), jnp.array(token_ids))
    convert_parameters_nnx_to_torch(nnx_model, torch_model)
    linen_variables = convert_parameters_nnx_to_linen(
        nnx_model, linen_model, variables=linen_variables, exmp_input=jnp.array(token_ids)
    )
    assert_parameters_match(nnx_model, torch_model)
    assert_linen_nnx_parameters_match(linen_model, nnx_model, variables=linen_variables)

    # Test forward pass
    nnx_out = nnx.nn.activations.log_softmax(nnx_model(jnp.array(token_ids)), axis=-1)
    torch_out = torch.nn.functional.log_softmax(torch_model(torch.from_numpy(token_ids)), dim=-1)
    linen_out = nnx.nn.activations.log_softmax(linen_model.apply(linen_variables, jnp.array(token_ids)), axis=-1)

    # Compare outputs
    # assert_allclose_with_plot(
    #     np.array(nnx_out),
    #     torch_out.cpu().detach().numpy(),
    #     rtol=3e-2,
    #     atol=3e-2,
    #     base_path=f"{test_name}_{next(counter)}",
    # )

    # Compare outputs
    assert_allclose_with_plot(
        np.array(nnx_out),
        np.array(linen_out),
        rtol=1.5e-1,
        atol=3e-2,
        base_path=f"{test_name}_{next(counter)}",
    )

    # Test gradients
    def nnx_loss(nnx_model, token_ids):
        return optax.softmax_cross_entropy_with_integer_labels(logits=nnx_model(token_ids), labels=token_ids).mean()

    nnx_grad_fn = nnx.grad(nnx_loss, argnums=0)

    nnx_grad = nnx_grad_fn(nnx_model, token_ids)
    torch_model.zero_grad()
    torch_token_ids = torch.from_numpy(token_ids)
    torch_out = torch_model(torch_token_ids)
    torch_loss = torch.nn.functional.cross_entropy(
        torch_out.view(-1, vocab_size), torch.from_numpy(token_ids).view(-1).to(dtype=torch.long)
    )
    torch_loss.backward()
    torch_grad_embedding = torch_model.token_embedding.weight.grad
    nnx_grad_embedding = nnx_grad["token_embedding"]["embedding"].value

    # Compare gradients
    assert_allclose_with_plot(
        np.array(nnx_grad_embedding),
        torch_grad_embedding.cpu().detach().numpy(),
        rtol=1.5e-1,
        atol=1.5e-3,
        base_path=f"{test_name}_{next(counter)}",
    )


@pytest.mark.parametrize("framework", ["jax", "torch"])
def test_invalid_config(framework, request):
    """Test that invalid configurations raise appropriate errors in both
    frameworks."""
    model_class = NNXLMModel if framework == "jax" else TorchLMModel
    model_args = {"rngs": nnx.Rngs(0)} if framework == "jax" else {}

    # Test invalid vocab_size
    with pytest.raises((AssertionError, ValueError)):
        config = pLSTMLMModelConfig(vocab_size=-1)
        model_class(config, **model_args)

    # Test invalid embedding_dim
    with pytest.raises((AssertionError, ValueError)):
        config = pLSTMLMModelConfig(embedding_dim=-1)
        model_class(config, **model_args)

    # Test invalid num_blocks
    with pytest.raises((AssertionError, ValueError)):
        config = pLSTMLMModelConfig(num_blocks=-1)
        model_class(config, **model_args)

    # Test invalid block_type
    with pytest.raises((AssertionError, ValueError)):
        config = pLSTMLMModelConfig(block_type="invalid")
        model_class(config, **model_args)
