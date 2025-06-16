from plstm.nnx_dummy import nnx
import jax.numpy as jnp
import jax

from ..config.vision_model import pLSTMVisionModelConfig
from .vision_util import VitPatchEmbed, VitPosEmbed2d
from .norm import NormInterface
from .block_stack import BlockStack
from .interfaces import ResidualModule
from .vision_blocks import *  # noqa registration
from .scalar import SoftCapFunctionLayer
from .initialization import InitInterface
from .dtype import str_dtype_to_jax


class pLSTMVisionModel(nnx.Module):
    config: pLSTMVisionModelConfig
    patch_embed: VitPatchEmbed
    pos_embed: VitPosEmbed2d
    block_stack: BlockStack
    norm: NormInterface
    head: nnx.Linear | None
    logit_softcap: SoftCapFunctionLayer
    output_shape: tuple[int, ...] | None

    def __init__(self, config: pLSTMVisionModelConfig, *, rngs: nnx.Rngs):
        nnx.Module.__init__(self)
        self.config = config

        self.patch_embed = VitPatchEmbed(
            config=config.patch_embed,
            rngs=rngs,
        )

        if config.use_pos_embed:
            self.pos_embed = VitPosEmbed2d(
                config=config.pos_embed,
                rngs=rngs,
            )

        if self.config.pooling == "corners":
            head_dim = self.config.dim * 4
        else:
            head_dim = self.config.dim
        if self.config.pooling == "cls":
            self.cls_token = nnx.Param(
                config.pos_embed.embed_init.instantiate(InitInterface)(
                    rngs.params(), (self.config.dim, 1), dtype=str_dtype_to_jax(config.param_dtype)
                ).reshape((1, 1, self.config.dim))
            )

        self.block_stack = config.block_stack.instantiate(ResidualModule, rngs=rngs)
        self.norm = config.norm.instantiate(NormInterface, rngs=rngs)

        if config.pooling == "features":
            self.head = None
            if self.config.pooling is None:
                self.output_shape = (self.patch_embed.num_patches, self.config.dim)
            elif self.pooling == "to_image":
                self.output_shape = (self.config.dim, *self.patch_embed.seqlens)
            else:
                raise NotImplementedError("Bad pooling mode")
        elif config.mode == "classifier":
            self.head = nnx.Linear(
                head_dim,
                self.config.output_shape[0],
                use_bias=True,
                bias_init=config.head_bias_init.instantiate(InitInterface),
                kernel_init=config.head_weight_init.instantiate(InitInterface),
                param_dtype=str_dtype_to_jax(config.param_dtype),
                dtype=str_dtype_to_jax(config.param_dtype),
                rngs=rngs,
            )
        if config.logit_softcap is not None:
            self.logit_softcap = SoftCapFunctionLayer(config.logit_softcap, rngs=rngs)
        else:
            self.logit_softcap = None

    def __call__(self, pixels: jax.Array, deterministic: bool = False) -> jax.Array:
        # embed patches
        patches = self.patch_embed(pixels, deterministic=deterministic)
        # add positional embedding
        if self.config.use_pos_embed:
            patches = self.pos_embed(patches, deterministic=deterministic)

        if self.config.pooling == "cls":
            patches = patches.reshape(patches.shape[0], -1, patches.shape[-1])
            cls_token = jnp.repeat(self.cls_token, patches.shape[0], axis=0).astype(patches.dtype)
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

        if self.head is not None:
            x = self.head(x)

        if self.logit_softcap is not None:
            x = self.logit_softcap(x)

        return x
