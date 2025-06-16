# from copy import deepcopy

import torch
from torch import nn
from ..config.vision_model import pLSTMVisionModelConfig
from .vision_util import VitPatchEmbed, VitPosEmbed2d, interpolate_sincos
from .norm import NormInterface
from .interfaces import ResidualModule
from .scalar import SoftCapFunctionLayer
from .block_stack import *  # noqa registration
from .vision_blocks import *  # noqa registration
from .initialization import InitInterface
from .dtype import str_dtype_to_torch


class pLSTMVisionModel(torch.nn.Module):
    config: pLSTMVisionModelConfig

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config

        self.patch_embed = VitPatchEmbed(self.config.patch_embed)

        if self.config.pooling == "cls":
            self.cls_token = nn.Parameter(
                torch.zeros(self.config.dim, 1, dtype=str_dtype_to_torch(self.config.param_dtype))
            )

        if self.config.use_pos_embed:
            self.pos_embed = VitPosEmbed2d(self.config.pos_embed)

        if self.config.pooling == "corners":
            head_dim = self.config.dim * 4
        else:
            head_dim = self.config.dim

        self.block_stack = config.block_stack.instantiate(ResidualModule)

        self.norm = config.norm.instantiate(NormInterface)

        if config.pooling == "features":
            assert self.output_shape is None
            self.head = None
            if self.config.pooling is None:
                self.output_shape = (self.patch_embed.num_patches, self.config.dim)
            elif self.pooling == "to_image":
                self.output_shape = (self.config.dim, *self.patch_embed.seqlens)
            else:
                raise NotImplementedError("Bad pooling mode")
        elif config.mode == "classifier":
            self.head = nn.Linear(head_dim, self.config.output_shape[0])
            nn.init.trunc_normal_(self.head.weight, std=2e-5)
            nn.init.zeros_(self.head.bias)

        if self.config.logit_softcap is not None:
            self.logit_softcap = SoftCapFunctionLayer(self.config.logit_softcap)
        else:
            self.logit_softcap = None

        self.reset_parameters()

    def reset_parameters(self):
        self.block_stack.reset_parameters()
        if self.config.pooling == "cls":
            self.config.pos_embed.embed_init.instantiate(InitInterface)(self.cls_token)
        if self.config.mode == "classifier":
            # following MAE https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L257
            nn.init.trunc_normal_(self.head.weight, std=2e-5)
            nn.init.zeros_(self.head.bias)

    def load_state_dict(self, state_dict, strict=True):
        # interpolate pos_embed for different resolution (e.g. for fine-tuning on higher-resolution)
        old_pos_embed = state_dict["pos_embed.embed"]
        if old_pos_embed.shape != self.pos_embed.embed.shape:
            state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {"pos_embed.embed"}

    def forward(self, pixels):
        # embed patches
        patches = self.patch_embed(pixels)
        # add positional embedding
        if self.config.use_pos_embed:
            patches = self.pos_embed(patches)

        if self.config.pooling == "cls":
            patches = patches.reshape(patches.shape[0], -1, patches.shape[-1])
            cls_token = self.cls_token.reshape(1, 1, -1).repeat(*patches.shape[:-1], 1).to(dtype=patches.dtype)
            patches = torch.concat((cls_token, patches), axis=1)

        # apply blocks
        patches = self.block_stack(patches)
        patches = self.norm(patches)

        if self.config.pooling == "to_image":
            seqlen_h, seqlen_w = self.config.seqlens

            x = patches.reshape(patches.shape[0], seqlen_h, seqlen_w, self.config.dim).permute(0, 3, 1, 2)
        elif self.config.pooling == "corners":
            x = torch.concat(
                [
                    patches[:, 0, 0],
                    patches[:, -1, 0],
                    patches[:, 0, -1],
                    patches[:, -1, -1],
                ],
                dim=-1,
            )
        elif self.config.pooling == "mean":
            x = torch.mean(patches, dim=(1, 2))
        elif self.config.pooling == "sum":
            x = torch.sum(patches, dim=(1, 2))
        elif self.config.pooling == "cls":
            x = patches[:, 0]
        elif self.config.pooling == "center":
            seqlen_h, seqlen_w = self.config.seqlens
            x = patches[:, seqlen_h // 2, seqlen_w // 2]

        if self.head is not None:
            x = self.head(x)

        if self.logit_softcap is not None:
            x = self.logit_softcap(x)

        return x
