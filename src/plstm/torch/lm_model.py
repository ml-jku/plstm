from torch import nn
import torch
import logging

from ..config.lm_model import pLSTMLMModelConfig
from .interfaces import ResidualModule
from .norm import NormInterface
from .scalar import ScalarFunctionLayer
from .initialization import InitInterface

LOGGER = logging.getLogger(__name__)


class pLSTMLMModel(nn.Module):
    config: pLSTMLMModelConfig

    def __init__(self, config: pLSTMLMModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.token_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim)

        self.block_stack = self.config.block_stack.instantiate(ResidualModule)

        self.post_blocks_norm = self.config.post_blocks_norm.instantiate(NormInterface)
        if not self.config.tie_weights:
            self.lm_head = nn.Linear(
                config.embedding_dim,
                config.vocab_size,
                bias=False,
            )
        self.logit_soft_cap: ScalarFunctionLayer = config.logit_soft_cap.instantiate(ScalarFunctionLayer)

    def reset_parameters(self):
        with torch.no_grad():
            self.config.embed_init.instantiate(InitInterface)(self.token_embedding.weight)
            self.config.embed_init.instantiate(InitInterface)(self.lm_head.weight)
            self.post_blocks_norm.reset_parameters()
            for block in self.blocks:
                block.reset_parameters()

    def get_weight_decay_optim_groups(self):
        wd, no_wd = [], []
        for parname, par in self.named_parameters():
            if "lm_head.weight" in parname and self.config.tie_weights:
                continue
            if "weight" in parname and "norm" not in parname and "embedding" not in parname:
                wd.append(par)
            elif (
                "bias" in parname
                or "scale" in parname
                or "embedding" in parname
                or ("norm" in parname and "weight" in parname)
            ):
                no_wd.append(par)
            else:
                LOGGER.warning("Parameter without defined weight decay {parname}")

        return wd, no_wd

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)
        x = self.block_stack(x)
        x2 = self.post_blocks_norm(x)
        if self.config.tie_weights:
            x3 = x2 @ self.token_embedding.weight.T
        else:
            x3 = self.lm_head(x2)
        x4 = self.logit_soft_cap(x3)
        return x4
