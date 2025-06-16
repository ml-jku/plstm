"""Parameter conversion for TransformerBlock between NNX, Linen, and
PyTorch."""

from .base import ConversionRegistry, _convert_to_torch
from .nnx_torch import convert_parameters_nnx_to_torch, convert_parameters_torch_to_nnx
import torch
import jax.numpy as jnp
import numpy as np
from plstm.nnx_dummy import nnx
from ..nnx.transformer_block import TransformerBlock as NNXTransformerBlock
from ..linen.transformer_block import TransformerBlock as LinenTransformerBlock
from ..torch.transformer_block import TransformerBlock as TorchTransformerBlock


# NNX <-> PyTorch conversions


@ConversionRegistry.register(NNXTransformerBlock, TorchTransformerBlock)
def convert_transformer_block_nnx_torch(
    nnx_module: NNXTransformerBlock, torch_module: TorchTransformerBlock, *, reverse: bool = False, **kwargs
):
    """Convert parameters between NNX and PyTorch TransformerBlock modules.

    Args:
        nnx_module: NNX TransformerBlock module
        torch_module: PyTorch TransformerBlock module
        reverse: If True, convert from PyTorch to NNX. If False, convert from NNX to PyTorch.
        **kwargs: Additional arguments (not used)
    """
    if reverse:
        # PyTorch to NNX
        for submod in ["norm", "norm2", "upproj", "downproj", "scale", "scale2"]:
            if hasattr(torch_module, submod):
                convert_parameters_torch_to_nnx(getattr(torch_module, submod), getattr(nnx_module, submod))
        input_dim = torch_module.config.input_dim
        num_heads = torch_module.config.num_heads
        nnx_module.multiheadattention.query.kernel.value = jnp.array(
            torch_module.qkv.weight[:input_dim, :]
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 0)
            .reshape(input_dim, num_heads, input_dim // num_heads)
        )
        nnx_module.multiheadattention.key.kernel.value = jnp.array(
            torch_module.qkv.weight[input_dim : 2 * input_dim, :]
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 0)
            .reshape(input_dim, num_heads, input_dim // num_heads)
        )
        nnx_module.multiheadattention.value.kernel.value = jnp.array(
            torch_module.qkv.weight[2 * input_dim :, :]
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 0)
            .reshape(input_dim, num_heads, input_dim // num_heads)
        )
        nnx_module.multiheadattention.out.kernel.value = (
            torch_module.outproj.weight.detach()
            .cpu()
            .numpy()
            .transpose(1, 0)
            .reshape(num_heads, input_dim // num_heads, input_dim)
        )
        if hasattr(torch_module.outproj, "bias") and torch_module.outproj.bias is not None:
            nnx_module.multiheadattention.out.bias.value = jnp.array(
                torch_module.outproj.bias.detach().cpu().numpy().reshape(num_heads, input_dim // num_heads)
            )
        if hasattr(torch_module.qkv, "bias") and torch_module.outproj.bias is not None:
            nnx_module.multiheadattention.query.bias.value = jnp.array(
                torch_module.qkv.bias[:input_dim].detach().cpu().numpy().reshape(num_heads, input_dim // num_heads)
            )
            nnx_module.multiheadattention.key.bias.value = jnp.array(
                torch_module.qkv.bias[input_dim : 2 * input_dim]
                .detach()
                .cpu()
                .numpy()
                .reshape(num_heads, input_dim // num_heads)
            )
            nnx_module.multiheadattention.value.bias.value = jnp.array(
                torch_module.qkv.bias[2 * input_dim :].detach().cpu().numpy().reshape(num_heads, input_dim // num_heads)
            )

    else:
        input_dim = nnx_module.config.input_dim
        num_heads = nnx_module.config.num_heads

        for submod in ["norm", "norm2", "upproj", "downproj", "scale", "scale2"]:
            if hasattr(torch_module, submod):
                convert_parameters_nnx_to_torch(getattr(nnx_module, submod), getattr(torch_module, submod))
        with torch.no_grad():
            torch_module.qkv.weight.data.copy_(
                _convert_to_torch(
                    np.array(
                        jnp.concatenate(
                            (
                                nnx_module.multiheadattention.query.kernel.reshape(input_dim, input_dim),
                                nnx_module.multiheadattention.key.kernel.reshape(input_dim, input_dim),
                                nnx_module.multiheadattention.value.kernel.reshape(input_dim, input_dim),
                            ),
                            axis=-1,
                        )
                    )
                ).transpose(0, 1)
            )
            torch_module.outproj.weight.data.copy_(
                _convert_to_torch(
                    np.array(nnx_module.multiheadattention.out.kernel.reshape(input_dim, input_dim))
                ).transpose(0, 1)
            )
            if hasattr(torch_module.qkv, "bias") and torch_module.qkv.bias is not None:
                torch_module.qkv.bias.data.copy_(
                    _convert_to_torch(
                        np.array(
                            jnp.concatenate(
                                (
                                    nnx_module.multiheadattention.query.bias.reshape(input_dim),
                                    nnx_module.multiheadattention.key.bias.reshape(input_dim),
                                    nnx_module.multiheadattention.value.bias.reshape(input_dim),
                                ),
                                axis=0,
                            )
                        )
                    )
                )
            if hasattr(torch_module.outproj, "bias") and torch_module.outproj.bias is not None:
                torch_module.outproj.bias.data.copy_(
                    _convert_to_torch(np.array(nnx_module.multiheadattention.out.bias.reshape(-1)))
                )


# Linen <-> PyTorch conversions


@ConversionRegistry.register(LinenTransformerBlock, TorchTransformerBlock)
def convert_transformer_block_linen_torch(
    linen_module: LinenTransformerBlock, torch_module: TorchTransformerBlock, *, reverse: bool = False, **kwargs
):
    """Convert parameters between Linen and PyTorch TransformerBlock modules.

    Args:
        linen_module: Linen TransformerBlock module
        torch_module: PyTorch TransformerBlock module
        reverse: If True, convert from PyTorch to Linen. If False, convert from Linen to PyTorch.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    linen_children = kwargs.get("linen_children")

    if reverse:
        # PyTorch to Linen - return updated variables
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Convert common modules
        for submod in ["norm", "norm2", "upproj", "downproj", "scale", "scale2"]:
            if hasattr(torch_module, submod) and submod in linen_children:
                # Get the corresponding child variables
                child_variables = {"params": params.get(submod, {})}
                for collection in updated_variables:
                    if collection != "params" and submod in updated_variables[collection]:
                        child_variables[collection] = updated_variables[collection][submod]

                # Get the corresponding linen child
                linen_child = linen_children.get(submod)
                if linen_child is None:
                    continue

                # Convert parameters
                from .linen_torch import convert_parameters_torch_to_linen

                linen_sub_children = {
                    "/".join(path.split("/")[1:]): mod
                    for path, mod in linen_children.items()
                    if path.startswith(submod + "/")
                }

                updated_child_variables = convert_parameters_torch_to_linen(
                    getattr(torch_module, submod),
                    linen_child,
                    variables=child_variables,
                    linen_children=linen_sub_children,
                )

                # Update the parent variables with the child variables
                params[submod] = updated_child_variables["params"]
                for collection in updated_child_variables:
                    if collection != "params":
                        if collection not in updated_variables:
                            updated_variables[collection] = {}
                        updated_variables[collection][submod] = updated_child_variables[collection]

        # Handle the multiheadattention module
        input_dim = torch_module.config.input_dim
        num_heads = torch_module.config.num_heads
        head_dim = input_dim // num_heads

        # Extract query, key, value weights from the qkv weight
        qkv_weight = torch_module.qkv.weight.detach().cpu().numpy()
        q_weight = qkv_weight[:input_dim, :].transpose(1, 0).reshape(input_dim, num_heads, head_dim)
        k_weight = qkv_weight[input_dim : 2 * input_dim, :].transpose(1, 0).reshape(input_dim, num_heads, head_dim)
        v_weight = qkv_weight[2 * input_dim :, :].transpose(1, 0).reshape(input_dim, num_heads, head_dim)

        # Update the multiheadattention parameters
        if "multiheadattention" not in params:
            params["multiheadattention"] = {}

        params["multiheadattention"]["query"] = {"kernel": jnp.array(q_weight)}
        params["multiheadattention"]["key"] = {"kernel": jnp.array(k_weight)}
        params["multiheadattention"]["value"] = {"kernel": jnp.array(v_weight)}

        if hasattr(torch_module.qkv, "bias") and torch_module.qkv.bias is not None:
            qkv_bias = torch_module.qkv.bias.detach().cpu().numpy()
            params["multiheadattention"]["query"]["bias"] = jnp.array(qkv_bias[:input_dim]).reshape(
                (num_heads, head_dim)
            )
            params["multiheadattention"]["key"]["bias"] = jnp.array(qkv_bias[input_dim : 2 * input_dim]).reshape(
                (num_heads, head_dim)
            )
            params["multiheadattention"]["value"]["bias"] = jnp.array(qkv_bias[2 * input_dim :]).reshape(
                (num_heads, head_dim)
            )

        # Handle the output projection
        if hasattr(torch_module, "outproj"):
            out_weight = (
                torch_module.outproj.weight.detach()
                .cpu()
                .numpy()
                .transpose(1, 0)
                .reshape(num_heads, input_dim // num_heads, input_dim)
            )
            params["multiheadattention"]["out"] = {"kernel": jnp.array(out_weight)}

            if torch_module.outproj.bias is not None:
                out_bias = torch_module.outproj.bias.detach().cpu().numpy().reshape(input_dim)
                params["multiheadattention"]["out"]["bias"] = jnp.array(out_bias)

        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to PyTorch
        # Convert common modules
        for submod in ["norm", "norm2", "upproj", "downproj", "scale", "scale2"]:
            if submod in linen_children and hasattr(torch_module, submod):
                # Get the corresponding child variables
                child_variables = {"params": variables["params"].get(submod, {})}
                for collection in variables:
                    if collection != "params" and submod in variables[collection]:
                        child_variables[collection] = variables[collection][submod]

                # Get the corresponding linen child
                linen_child = linen_children.get(submod)
                if linen_child is None:
                    continue

                # Convert parameters
                from .linen_torch import convert_parameters_linen_to_torch

                linen_sub_children = {
                    "/".join(path.split("/")[1:]): mod
                    for path, mod in linen_children.items()
                    if path.startswith(submod + "/")
                }
                convert_parameters_linen_to_torch(
                    linen_child,
                    getattr(torch_module, submod),
                    variables=child_variables,
                    linen_children=linen_sub_children,
                )

        # Handle the multiheadattention module
        input_dim = linen_module.config.input_dim
        num_heads = linen_module.config.num_heads

        # Get the multiheadattention parameters
        mha_params = variables["params"].get("multiheadattention", {})

        # Extract query, key, value weights
        q_kernel = mha_params.get("query", {}).get("kernel")
        k_kernel = mha_params.get("key", {}).get("kernel")
        v_kernel = mha_params.get("value", {}).get("kernel")

        if q_kernel is not None and k_kernel is not None and v_kernel is not None:
            # Reshape and concatenate the weights
            q_reshaped = np.array(q_kernel).reshape(input_dim, -1)
            k_reshaped = np.array(k_kernel).reshape(input_dim, -1)
            v_reshaped = np.array(v_kernel).reshape(input_dim, -1)

            qkv_weight = np.concatenate([q_reshaped, k_reshaped, v_reshaped], axis=1)

            # Update the qkv weight
            with torch.no_grad():
                torch_module.qkv.weight.copy_(_convert_to_torch(qkv_weight).transpose(0, 1))

            q_bias = mha_params.get("query", {}).get("bias")
            k_bias = mha_params.get("key", {}).get("bias")
            v_bias = mha_params.get("value", {}).get("bias")
            if q_bias is not None and torch_module.qkv.bias is not None:
                with torch.no_grad():
                    torch_module.qkv.bias.copy_(
                        _convert_to_torch(
                            np.array(
                                np.concatenate(
                                    (q_bias.reshape(input_dim), k_bias.reshape(input_dim), v_bias.reshape(input_dim)),
                                    axis=-1,
                                )
                            )
                        )
                    )

        # Handle the output projection
        if hasattr(torch_module, "outproj"):
            out_kernel = mha_params.get("out", {}).get("kernel")
            if out_kernel is not None:
                with torch.no_grad():
                    torch_module.outproj.weight.copy_(
                        _convert_to_torch(np.array(out_kernel).reshape(input_dim, input_dim)).transpose(0, 1)
                    )

            out_bias = mha_params.get("out", {}).get("bias")
            if out_bias is not None and torch_module.outproj.bias is not None:
                with torch.no_grad():
                    torch_module.outproj.bias.copy_(_convert_to_torch(np.array(out_bias).reshape(input_dim)))


# Linen <-> NNX conversions


@ConversionRegistry.register(LinenTransformerBlock, NNXTransformerBlock)
def convert_transformer_block_linen_nnx(
    linen_module: LinenTransformerBlock, nnx_module: NNXTransformerBlock, *, reverse: bool = False, **kwargs
):
    """Convert parameters between Linen and NNX TransformerBlock modules.

    Args:
        linen_module: Linen TransformerBlock module
        nnx_module: NNX TransformerBlock module
        reverse: If True, convert from NNX to Linen. If False, convert from Linen to NNX.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")
    linen_children = kwargs.get("linen_children")

    if reverse:
        # NNX to Linen - return updated variables
        updated_variables = variables.copy()
        params = updated_variables.get("params", {}).copy()

        # Convert common modules
        for submod in ["norm", "norm2", "upproj", "downproj", "scale", "scale2"]:
            if hasattr(nnx_module, submod) and submod in linen_children:
                # Get the corresponding child variables
                child_variables = {"params": params.get(submod, {})}
                for collection in updated_variables:
                    if collection != "params" and submod in updated_variables[collection]:
                        child_variables[collection] = updated_variables[collection][submod]

                # Get the corresponding linen child
                linen_child = linen_children.get(submod)
                if linen_child is None:
                    continue

                # Convert parameters
                from .linen_nnx import convert_parameters_nnx_to_linen

                linen_sub_children = {
                    "/".join(path.split("/")[1:]): mod
                    for path, mod in linen_children.items()
                    if path.startswith(submod + "/")
                }
                updated_child_variables = convert_parameters_nnx_to_linen(
                    getattr(nnx_module, submod),
                    linen_child,
                    variables=child_variables,
                    linen_children=linen_sub_children,
                )

                # Update the parent variables with the child variables
                params[submod] = updated_child_variables["params"]
                for collection in updated_child_variables:
                    if collection != "params":
                        if collection not in updated_variables:
                            updated_variables[collection] = {}
                        updated_variables[collection][submod] = updated_child_variables[collection]

        # Handle the multiheadattention module
        if "multiheadattention" not in params:
            params["multiheadattention"] = {}

        # Copy query, key, value kernels
        if hasattr(nnx_module, "multiheadattention"):
            mha = nnx_module.multiheadattention

            if hasattr(mha, "query") and hasattr(mha.query, "kernel"):
                params["multiheadattention"]["query"] = {"kernel": jnp.array(mha.query.kernel)}
                if hasattr(mha.query, "bias") and mha.query.bias is not None and mha.query.bias.value is not None:
                    params["multiheadattention"]["query"]["bias"] = jnp.array(mha.query.bias)

            if hasattr(mha, "key") and hasattr(mha.key, "kernel"):
                params["multiheadattention"]["key"] = {"kernel": jnp.array(mha.key.kernel)}
                if hasattr(mha.key, "bias") and mha.key.bias is not None and mha.key.bias.value is not None:
                    params["multiheadattention"]["key"]["bias"] = jnp.array(mha.key.bias)

            if hasattr(mha, "value") and hasattr(mha.value, "kernel"):
                params["multiheadattention"]["value"] = {"kernel": jnp.array(mha.value.kernel)}
                if hasattr(mha.value, "bias") and mha.value.bias is not None and mha.value.bias.value is not None:
                    params["multiheadattention"]["value"]["bias"] = jnp.array(mha.value.bias)

            if hasattr(mha, "out") and hasattr(mha.out, "kernel"):
                params["multiheadattention"]["out"] = {"kernel": jnp.array(mha.out.kernel)}

                if hasattr(mha.out, "bias") and mha.out.bias is not None and mha.out.bias.value is not None:
                    params["multiheadattention"]["out"]["bias"] = jnp.array(mha.out.bias)

        updated_variables["params"] = params
        return updated_variables
    else:
        # Linen to NNX
        # Convert common modules
        for submod in ["norm", "norm2", "upproj", "downproj", "scale", "scale2"]:
            if submod in linen_children and hasattr(nnx_module, submod):
                # Get the corresponding child variables
                child_variables = {"params": variables["params"].get(submod, {})}
                for collection in variables:
                    if collection != "params" and submod in variables[collection]:
                        child_variables[collection] = variables[collection][submod]

                # Get the corresponding linen child
                linen_child = linen_children.get(submod)
                if linen_child is None:
                    continue

                # Convert parameters
                from .linen_nnx import convert_parameters_linen_to_nnx

                linen_sub_children = {
                    "/".join(path.split("/")[1:]): mod
                    for path, mod in linen_children.items()
                    if path.startswith(submod + "/")
                }
                convert_parameters_linen_to_nnx(
                    linen_child,
                    getattr(nnx_module, submod),
                    variables=child_variables,
                    linen_children=linen_sub_children,
                )

        # Handle the multiheadattention module
        mha_params = variables["params"].get("multiheadattention", {})

        # Copy query, key, value kernels
        if hasattr(nnx_module, "multiheadattention"):
            mha = nnx_module.multiheadattention

            q_kernel = mha_params.get("query", {}).get("kernel")
            if q_kernel is not None and hasattr(mha, "query"):
                mha.query.kernel = nnx.Param(jnp.array(q_kernel))
            q_bias = mha_params.get("query", {}).get("bias")
            if q_bias is not None and hasattr(mha, "query"):
                mha.query.bias = nnx.Param(jnp.array(q_bias))

            k_kernel = mha_params.get("key", {}).get("kernel")
            if k_kernel is not None and hasattr(mha, "key"):
                mha.key.kernel = nnx.Param(jnp.array(k_kernel))
            k_bias = mha_params.get("key", {}).get("bias")
            if k_bias is not None and hasattr(mha, "key"):
                mha.key.bias = nnx.Param(jnp.array(k_bias))

            v_kernel = mha_params.get("value", {}).get("kernel")
            if v_kernel is not None and hasattr(mha, "value"):
                mha.value.kernel = nnx.Param(jnp.array(v_kernel))
            v_bias = mha_params.get("value", {}).get("bias")
            if v_bias is not None and hasattr(mha, "value"):
                mha.value.bias = nnx.Param(jnp.array(v_bias))

            out_kernel = mha_params.get("out", {}).get("kernel")
            if out_kernel is not None and hasattr(mha, "out"):
                mha.out.kernel = nnx.Param(jnp.array(out_kernel))

                out_bias = mha_params.get("out", {}).get("bias")
                if out_bias is not None:
                    mha.out.bias = nnx.Param(jnp.array(out_bias))
