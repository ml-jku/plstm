"""Parameter conversion for BlockStack between NNX, Linen, and PyTorch."""

from .base import ConversionRegistry
from ..nnx.block_stack import BlockStack as NNXBlockStack
from ..linen.block_stack import BlockStack as LinenBlockStack
from ..torch.block_stack import BlockStack as TorchBlockStack


# NNX <-> PyTorch conversions


@ConversionRegistry.register(NNXBlockStack, TorchBlockStack)
def convert_block_stack_nnx_torch(
    nnx_module: NNXBlockStack, torch_module: TorchBlockStack, *, reverse: bool = False, **kwargs
):
    """Convert parameters between NNX and PyTorch BlockStack modules.

    Args:
        nnx_module: NNX BlockStack module
        torch_module: PyTorch BlockStack module
        reverse: If True, convert from PyTorch to NNX. If False, convert from NNX to PyTorch.
        **kwargs: Additional arguments (not used)
    """
    if reverse:
        # PyTorch to NNX
        for i, torch_block in enumerate(torch_module.blocks):
            if i < nnx_module.config.num_blocks:
                nnx_block = getattr(nnx_module, f"block{i}")
                from .nnx_torch import convert_parameters_torch_to_nnx

                convert_parameters_torch_to_nnx(torch_block, nnx_block)
    else:
        # NNX to PyTorch
        for i, torch_block in enumerate(torch_module.blocks):
            if i < nnx_module.config.num_blocks:
                nnx_block = getattr(nnx_module, f"block{i}")
                from .nnx_torch import convert_parameters_nnx_to_torch

                convert_parameters_nnx_to_torch(nnx_block, torch_block)


# Linen <-> PyTorch conversions


@ConversionRegistry.register(LinenBlockStack, TorchBlockStack)
def convert_block_stack_linen_torch(
    linen_module: LinenBlockStack, torch_module: TorchBlockStack, *, reverse: bool = False, **kwargs
):
    """Convert parameters between Linen and PyTorch BlockStack modules.

    Args:
        linen_module: Linen BlockStack module
        torch_module: PyTorch BlockStack module
        reverse: If True, convert from PyTorch to Linen. If False, convert from Linen to PyTorch.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")

    linen_children = kwargs.get("linen_children")

    if reverse:
        # PyTorch to Linen - return updated variables
        from .linen_torch import convert_parameters_torch_to_linen

        updated_variables = variables.copy()
        # Handle scan vs explicit blocks
        if linen_module.config.scan_blocks:
            # For scan, we need to handle the single block
            for i in range(torch_module.config.num_blocks):
                updated_variables["block"][i] = convert_parameters_torch_to_linen(
                    torch_module.blocks[i], linen_children["block"], variables
                )
        else:
            # For explicit blocks, convert each block
            for i, torch_block in enumerate(torch_module.blocks):
                linen_sub_children = {
                    "/".join(path.split("/")[1:]): mod
                    for path, mod in linen_children.items()
                    if path.startswith(f"blocks_{i}/")
                }
                updated_variables = convert_parameters_torch_to_linen(
                    torch_block, linen_children[f"blocks_{i}"], updated_variables, linen_children=linen_sub_children
                )
            return updated_variables
    else:
        # Linen to PyTorch
        from .linen_torch import convert_parameters_linen_to_torch

        # Handle scan vs explicit blocks
        if linen_module.config.scan_blocks:
            # For scan, we need to handle the single block
            convert_parameters_linen_to_torch(linen_module.block, torch_module.blocks[0], variables)
        else:
            # For explicit blocks, convert each block
            for i, torch_block in enumerate(torch_module.blocks):
                linen_sub_children = {
                    "/".join(path.split("/")[1:]): mod
                    for path, mod in linen_children.items()
                    if path.startswith(f"blocks_{i}/")
                }

                convert_parameters_linen_to_torch(
                    linen_children[f"blocks_{i}"], torch_block, variables, linen_children=linen_sub_children
                )


# Linen <-> NNX conversions


@ConversionRegistry.register(LinenBlockStack, NNXBlockStack)
def convert_block_stack_linen_nnx(
    linen_module: LinenBlockStack, nnx_module: NNXBlockStack, *, reverse: bool = False, **kwargs
):
    """Convert parameters between Linen and NNX BlockStack modules.

    Args:
        linen_module: Linen BlockStack module
        nnx_module: NNX BlockStack module
        reverse: If True, convert from NNX to Linen. If False, convert from Linen to NNX.
        **kwargs: Additional arguments including variables for Linen
    """
    variables = kwargs.get("variables")
    if variables is None:
        raise ValueError("Variables must be provided for Linen conversion")
    linen_children = kwargs.get("linen_children")
    if reverse:
        # NNX to Linen - return updated variables
        from .linen_nnx import convert_parameters_nnx_to_linen

        updated_variables = variables.copy()
        # Handle scan vs explicit blocks
        if linen_module.config.scan_blocks:
            # For scan, we need to handle the single block
            for i in range(nnx_module.config.num_blocks):
                nnx_block = getattr(nnx_module, f"block{i}")
                # Use the same block for all conversions in scan mode
                child_variables = {"params": updated_variables["params"].get("block", {})}
                for collection in updated_variables:
                    if collection != "params" and "block" in updated_variables[collection]:
                        child_variables[collection] = updated_variables[collection]["block"][i]

                linen_sub_children = {
                    "/".join(path.split("/")[1:]): mod
                    for path, mod in linen_children.items()
                    if path.startswith(f"blocks_{i}/")
                }

                updated_child_variables = convert_parameters_nnx_to_linen(
                    nnx_block,
                    linen_children["block"],
                    child_variables,
                    linen_children=linen_sub_children,
                )

                # Update the parent variables with the child variables
                updated_variables["params"]["block"] = updated_child_variables["params"]
                for collection in updated_child_variables:
                    if collection != "params":
                        if collection not in updated_variables:
                            updated_variables[collection] = {}
                        updated_variables[collection]["block"] = updated_child_variables[collection]

                # Only need to convert one block for scan mode
                break
        else:
            # For explicit blocks, convert each block
            for i in range(min(linen_module.config.num_blocks, nnx_module.config.num_blocks)):
                nnx_block = getattr(nnx_module, f"block{i}")

                # Get the corresponding child variables
                child_variables = {"params": updated_variables["params"].get(f"blocks_{i}", {})}
                for collection in updated_variables:
                    if collection != "params" and f"blocks_{i}" in updated_variables[collection]:
                        child_variables[collection] = updated_variables[collection][f"blocks_{i}"]

                linen_sub_children = {
                    "/".join(path.split("/")[1:]): mod
                    for path, mod in linen_children.items()
                    if path.startswith(f"blocks_{i}/")
                }

                updated_child_variables = convert_parameters_nnx_to_linen(
                    nnx_block, linen_children[f"blocks_{i}"], child_variables, linen_children=linen_sub_children
                )

                # Update the parent variables with the child variables
                updated_variables["params"][f"blocks_{i}"] = updated_child_variables["params"]
                for collection in updated_child_variables:
                    if collection != "params":
                        if collection not in updated_variables:
                            updated_variables[collection] = {}
                        updated_variables[collection][f"blocks_{i}"] = updated_child_variables[collection]

        return updated_variables
    else:
        # Linen to NNX
        from .linen_nnx import convert_parameters_linen_to_nnx

        # Handle scan vs explicit blocks
        if linen_module.config.scan_blocks:
            # For scan, we need to handle the single block
            for i in range(nnx_module.config.num_blocks):
                nnx_block = getattr(nnx_module, f"block{i}")

                # Get the corresponding child variables
                child_variables = {"params": variables["params"].get("block", {})}
                for collection in variables:
                    if collection != "params" and "block" in variables[collection]:
                        child_variables[collection] = variables[collection]["block"]
                linen_sub_children = {
                    "/".join(path.split("/")[1:]): mod
                    for path, mod in linen_children.items()
                    if path.startswith(f"block_{i}/")
                }

                convert_parameters_linen_to_nnx(
                    linen_module.block, nnx_block, child_variables, linen_children=linen_sub_children
                )
        else:
            # For explicit blocks, convert each block
            for i in range(nnx_module.config.num_blocks):
                nnx_block = getattr(nnx_module, f"block{i}")

                # Get the corresponding child variables
                child_variables = {"params": variables["params"].get(f"blocks_{i}", {})}
                for collection in variables:
                    if collection != "params" and f"blocks_{i}" in variables[collection]:
                        child_variables[collection] = variables[collection][f"blocks_{i}"]
                linen_sub_children = {
                    "/".join(path.split("/")[1:]): mod
                    for path, mod in linen_children.items()
                    if path.startswith(f"block_{i}/")
                }

                convert_parameters_linen_to_nnx(
                    linen_children[f"block_{i}"], nnx_block, child_variables, linen_children=linen_sub_children
                )
