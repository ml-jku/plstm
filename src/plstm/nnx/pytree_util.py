import logging
from collections.abc import Callable
from typing import Any

import jax
from flax.core import FrozenDict

from ..util import PyTree

LOGGER = logging.getLogger(__name__)


def pytree_key_path_to_str(path: Any, separator: str = ".") -> str:
    """Converts a path to a string.

    An adjusted version of jax.tree_util.keystr to support different separators and easier to read output.

    Args:
        path: Path.  jax.tree_util.KeyPath
        separator: Separator fo r the keys.

    Returns:
        str: Path as string.
    """
    cleaned_keys = []
    for key in path:
        if isinstance(key, jax.tree_util.DictKey):
            cleaned_keys.append(f"{key.key}")
        elif isinstance(key, jax.tree_util.SequenceKey):
            cleaned_keys.append(f"{key.idx}")
        elif isinstance(key, jax.tree_util.GetAttrKey):
            cleaned_keys.append(key.name)
        else:
            cleaned_keys.append(str(key))
    return separator.join(cleaned_keys)


def flatten_pytree(
    pytree: PyTree, separator: str = ".", is_leaf: Callable[[Any], bool] | None = None
) -> dict[str, Any]:
    """Flattens a PyTree into a dict.

    Supports PyTrees with nested dictionaries, lists, tuples, and more. The keys are created by concatenating the
    path to the leaf with the separator. For sequences, the index is used as key (see examples below).

    Args:
        pytree: PyTree to be flattened.
        separator: Separator for the keys.
        is_leaf: Function that determines if a node is a leaf. If None, uses default PyTree leaf detection.

    Returns:
        dict: Flattened PyTree. In case of duplicate keys, a ValueError is raised.

    >>> flatten_pytree({"a": 1, "b": {"c": 2}})
    {'a': 1, 'b.c': 2}
    >>> flatten_pytree({"a": 1, "b": (2, 3, 4)}, separator="/")
    {'a': 1, 'b/0': 2, 'b/1': 3, 'b/2': 4}
    >>> flatten_pytree(("a", "b", "c"))
    {'0': 'a', '1': 'b', '2': 'c'}
    """
    leaves_with_path = jax.tree_util.tree_leaves_with_path(pytree, is_leaf=is_leaf)
    flat_pytree = {}
    for path, leave in leaves_with_path:
        key = pytree_key_path_to_str(path, separator=separator)
        if key in flat_pytree:
            raise ValueError(f"Duplicate key found: {key}")
        flat_pytree[key] = leave
    return flat_pytree


def flatten_dict(d: dict | FrozenDict, separator: str = ".") -> dict[str, Any]:
    """Flattens a nested dictionary.

    In contrast to flatten_pytree, this function is specifically designed for dictionaries and does not flatten
    sequences by default. It is equivalent to setting the is_leaf function in flatten_pytree to:
    `flatten_pytree(d, is_leaf=lambda x: not isinstance(x, (dict, FrozenDict)))`.

    Args:
        d: Dictionary to be flattened.
        separator: Separator for the keys.

    Returns:
        dict: Flattened dictionary.

    >>> flatten_dict({"a": {"b": 1}, "c": (2, 3, 4)})
    {'a.b': 1, 'c': (2, 3, 4)}
    """
    assert isinstance(
        d, (dict, FrozenDict)
    ), f"Expected a dict or FrozenDict, got {type(d)}. For general PyTrees, use flatten_pytree."
    return flatten_pytree(d, separator=separator, is_leaf=lambda x: not isinstance(x, (dict, FrozenDict)))


def get_shape_dtype_pytree(
    x: PyTree,
) -> PyTree:
    """Converts a PyTree of jax.Array objects to a PyTree of ShapeDtypeStruct
    objects.

    Leaf nodes of the PyTree that are not jax.Array objects are left unchanged.

    Args:
        x: PyTree of jax.Array objects.

    Returns:
        PyTree of ShapeDtypeStruct objects.
    """
    return jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding) if isinstance(x, jax.Array) else x, x
    )


def delete_arrays_in_pytree(x: PyTree) -> None:
    """Deletes and frees all jax.Array objects in a PyTree from the device
    memory.

    Leaf nodes of the PyTree that are not jax.Array objects are left unchanged.

    Args:
        x: PyTree of jax.Array objects.
    """

    def _delete_array(x: Any):
        if isinstance(x, jax.Array):
            LOGGER.debug("Delete array of shape", x.shape)
            x.delete()
        else:
            LOGGER.debug("Not deleting object of type", type(x))
        return x

    jax.tree.map(_delete_array, x)
