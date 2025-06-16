from flax import linen as nn
import jax.numpy as jnp
import jax

# from compoconf import Registry
from ..config.vision_model import pLSTMVisionModelConfig
from .vision_util import VitPatchEmbed, VitPosEmbed2d
from .norm import NormInterface
from .initialization import InitInterface
from .block_stack import BlockStack  # noqa register block stacks
from .interfaces import ResidualModule
from .vision_blocks import *  # noqa registration
from .scalar import SoftCapFunctionLayer
from .dtype import str_dtype_to_jax


class pLSTMVisionModel(nn.Module):
    config: pLSTMVisionModelConfig

    def setup(self):
        # self.embed = PatchEmbed(config=self.config.patch_embed)

        self.patch_embed = VitPatchEmbed(
            config=self.config.patch_embed,
        )

        if self.config.pooling == "cls":
            self.cls_token = self.param(
                "cls_token",
                self.config.pos_embed.embed_init.instantiate(InitInterface),
                (self.config.dim, 1),
                dtype=str_dtype_to_jax(self.config.param_dtype),
            )

        if self.config.use_pos_embed:
            self.pos_embed = VitPosEmbed2d(
                config=self.config.pos_embed,
            )

        self.block_stack = self.config.block_stack.instantiate(ResidualModule)
        self.norm = self.config.norm.instantiate(NormInterface)

        if self.config.pooling == "features":
            self.head = None
            if self.config.pooling is None:
                self.output_shape = (self.patch_embed.num_patches, self.config.dim)
            elif self.pooling == "to_image":
                self.output_shape = (self.config.dim, *self.patch_embed.seqlens)
            else:
                raise NotImplementedError("Bad pooling mode")
        elif self.config.mode == "classifier":
            self.head = nn.DenseGeneral(self.config.output_shape[0])

        if self.config.logit_softcap is not None:
            self.logit_softcap = SoftCapFunctionLayer(self.config.logit_softcap)
        else:
            self.logit_softcap = None

    def __call__(self, pixels: jax.Array, deterministic: bool = False) -> jax.Array:
        # embed patches
        patches = self.patch_embed(pixels, deterministic=deterministic)

        if self.config.use_pos_embed:
            patches = self.pos_embed(patches, deterministic=deterministic)

        if self.config.pooling == "cls":
            patches = patches.reshape(patches.shape[0], -1, patches.shape[-1])
            cls_token = jnp.repeat(self.cls_token.reshape(1, 1, -1), patches.shape[0], axis=0).astype(patches.dtype)
            patches = jnp.concatenate((cls_token, patches), axis=1)

        patches = self.block_stack(patches, deterministic=deterministic)
        patches = self.norm(patches)

        if self.config.pooling == "to_image":
            seqlen_h, seqlen_w = self.config.seqlens
            x = jnp.transpose(patches.reshape(patches.shape[0], seqlen_h, seqlen_w, self.config.dim), (0, 3, 1, 2))
        elif self.config.pooling == "corners":
            x = jnp.concatenate(
                [
                    patches[:, 0, 0],
                    patches[:, -1, 0],
                    patches[:, 0, -1],
                    patches[:, -1, -1],
                ],
                axis=-1,
            )
        elif self.config.pooling == "mean":
            x = jnp.mean(patches, axis=(1, 2))
        elif self.config.pooling == "sum":
            x = jnp.sum(patches, axis=(1, 2))
        elif self.config.pooling == "center":
            seqlen_h, seqlen_w = self.config.seqlens
            x = patches[:, seqlen_h // 2, seqlen_w // 2]
        elif self.config.pooling == "cls":
            x = patches[:, 0]

        if hasattr(self, "head") and self.head is not None:
            x = self.head(x)

        if self.logit_softcap is not None:
            x = self.logit_softcap(x)

        return x
