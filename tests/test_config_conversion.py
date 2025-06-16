from dataclasses import asdict
from plstm.config.lm_model import pLSTMLMModelConfig
from plstm.config.vision_model import pLSTMVisionModelConfig
from compoconf import parse_config
from plstm.util import pytree_diff
from plstm.nnx.lm_model import pLSTMLMModel  # noqa
from plstm.nnx.vision_model import pLSTMVisionModel  # noqa


def test_plstm_lm_config_conversion():
    """Test conversion of pLSTMLMModelConfig to dict and back."""
    # Create initial config
    cfg_lm = pLSTMLMModelConfig(embedding_dim=1024, vocab_size=50000)

    # Convert to dict
    cfg_dict = asdict(cfg_lm)

    # Convert back to config
    cfg_restored = parse_config(pLSTMLMModelConfig, cfg_dict)

    # Convert both to dict for comparison
    original_dict = asdict(cfg_lm)
    restored_dict = asdict(cfg_restored)

    # Compare using pytree_diff
    diff = pytree_diff(original_dict, restored_dict)
    assert not diff, f"Config conversion failed for pLSTMLMModelConfig:\n{diff}"


def test_plstm_vision_config_conversion():
    """Test conversion of pLSTMVisionModelConfig to dict and back."""
    # Create initial config
    cfg_vision = pLSTMVisionModelConfig()

    # Convert to dict
    cfg_dict = asdict(cfg_vision)

    # Convert back to config
    cfg_restored = parse_config(pLSTMVisionModelConfig, cfg_dict)

    # Convert both to dict for comparison
    original_dict = asdict(cfg_vision)
    restored_dict = asdict(cfg_restored)

    # Compare using pytree_diff
    diff = pytree_diff(original_dict, restored_dict)
    assert not diff, f"Config conversion failed for pLSTMVisionModelConfig:\n{diff}"
