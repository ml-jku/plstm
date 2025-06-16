from dataclasses import is_dataclass
from functools import lru_cache
from typing import get_type_hints, Literal, Any
from collections.abc import Mapping, Sequence

try:
    import jax
    import jax.numpy as jnp

    JAX_ARRAY = jax.Array
    JNP_ARRAY = jnp.ndarray
except (ImportError, ModuleNotFoundError):
    JAX_ARRAY = None
    JNP_ARRAY = None

try:
    import torch

    TORCH_TENSOR = torch.Tensor
except (ImportError, ModuleNotFoundError, AttributeError):
    TORCH_TENSOR = None

import numpy as np


def prod(*args):
    res = 1
    for arg in args:
        res *= arg
    return res


def revert_map(idxs):
    idxs = [(idx + len(idxs) if idx < 0 else idx) for idx in idxs]
    idx_rev_map = ((idx_map, idx) for idx, idx_map in enumerate(idxs))
    return idx_rev_map


def positive_index(idx, length):
    if idx < 0:
        return length + idx
    else:
        return idx


@lru_cache(maxsize=1024)
def log2(a):
    """
    >>> log2(1)
    0
    >>> log2(2)
    1
    >>> log2(5)
    2
    """
    n = 0
    while a > 1:
        a = a >> 1
        n += 1
    return n


def ispow2(a):
    """
    >>> ispow2(1)
    True
    >>> ispow2(2)
    True
    >>> ispow2(3)
    False
    >>> ispow2(8)
    True
    """
    return (1 << log2(a)) == a


def least_significant_bit(x):
    """
    >>> least_significant_bit(1)
    0
    >>> least_significant_bit(2)
    1
    >>> least_significant_bit(3)
    0
    >>> least_significant_bit(8)
    3
    >>> least_significant_bit(6)
    1
    """
    if x < 0:
        raise ValueError("Input must be > 0")
    for i in range(0, log2(x) + 1):
        d = x - ((x >> (i + 1)) << (i + 1))
        if d:
            return i


def validate_literal_field(obj, field_name):
    """Validates if the value of a specified field in a dataclass object is
    within the allowed Literal options defined in its type annotations.

    Args:
        obj: The dataclass object to validate.
        field_name: The name of the field to check.

    Returns:
        bool: True if the field value is valid, False otherwise.

    Raises:
        ValueError: If the field is not defined or not annotated with Literal.
        TypeError: If the object is not a dataclass instance.
    """
    if not is_dataclass(obj):
        raise TypeError(f"The provided object {obj} is not a dataclass instance.")

    type_hints = get_type_hints(type(obj))

    if field_name not in type_hints:
        raise ValueError(f"Field '{field_name}' is not defined in the dataclass.")

    field_type = type_hints[field_name]

    # Check if the type is a Literal
    if not hasattr(field_type, "__origin__") or field_type.__origin__ is not Literal:
        raise ValueError(f"Field '{field_name}' is not annotated with a Literal.")

    # Extract the allowed values from the Literal
    allowed_values = field_type.__args__

    # Check if the current value is in the allowed values
    current_value = getattr(obj, field_name)
    return current_value in allowed_values


def assert_check_literals(obj):
    """Validates if the value of all Literal field in a dataclass object are
    within the allowed Literal options defined in their type annotations.

    Args:
        obj: The dataclass object to validate.

    Returns:
        bool: True if the field value is valid, False otherwise.

    Raises:
        ValueError: If the field is not defined or not annotated with Literal.
        TypeError: If the object is not a dataclass instance.
    """
    if not is_dataclass(obj):
        raise TypeError(f"The provided object {obj} is not a dataclass instance.")

    type_hints = get_type_hints(type(obj))

    for field_name in type_hints:
        field_type = type_hints[field_name]
        # Check if the type is a Literal
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Literal:
            allowed_values = field_type.__args__
            current_value = getattr(obj, field_name)
            assert current_value in allowed_values, (
                f"In dataclass {type(obj)}: The field {field_name} has a value {current_value} "
                f"not in {allowed_values} defined by Literal annotation."
            )


class PyTree:
    pass


class RecursionLimit:
    pass


def pytree_diff(tree1: PyTree, tree2: PyTree, max_recursion: int = 30) -> PyTree:
    """Computes the difference between two PyTrees.

    Args:
        tree1: First PyTree.
        tree2: Second PyTree.

    Returns:
        A PyTree of the same structure, with only differing leaves.
        Returns None if no differences are found.

    >>> pytree_diff({"a": 1}, {"a": 2})
    {'a': (1, 2)}
    >>> pytree_diff({"a": 1}, {"a": 1})
    >>> pytree_diff([1, 2, 3], [1, 2])
    {'length_mismatch': (3, 2)}
    >>> pytree_diff(np.array([1, 2, 3]), np.array([1, 2]))
    {'shape_mismatch': ((3,), (2,))}
    """

    def diff_fn(a, b) -> Any:
        """Creates a diff of two elementary objects / leaves.

        Args:
            a: Any (not dict|list)
            b: Any (not dict|list)

        Returns:
            None if a == b else an informative diff object
        """
        # Check if both are arrays and calculate the difference
        if isinstance(a, (np.ndarray, JNP_ARRAY, TORCH_TENSOR)) or isinstance(b, (np.ndarray, JNP_ARRAY, TORCH_TENSOR)):
            if (isinstance(a, (np.ndarray, JNP_ARRAY)) and isinstance(b, (np.ndarray, JNP_ARRAY))) or (
                isinstance(a, TORCH_TENSOR) and isinstance(b, TORCH_TENSOR)
            ):
                if a.shape != b.shape:
                    return {"shape_mismatch": (a.shape, b.shape)}
                try:
                    if a.dtype == bool:
                        diff = a ^ b
                    else:
                        diff = a - b
                except ValueError:
                    return {"array_difference": (a, b)}
                if isinstance(diff, JAX_ARRAY):
                    return diff if not np.allclose(diff, jnp.zeros_like(diff)) else None
                elif isinstance(diff, TORCH_TENSOR):
                    diff = diff.detach().cpu().numpy()
                return diff if not np.allclose(diff, np.zeros_like(diff)) else None
            else:
                return a, b
        # Check for scalar values and report if different
        if a != b:
            return a, b
        # If identical, ignore
        return None

    def recursive_diff(t1, t2, max_recursion=30):
        """Recursive diff function for two PyTrees.

        Args:
            t1: PyTree object 1
            t2: PyTree object 2
            max_recursion: Recursion limiter

        Returns:
            None if the PyTree objects are equal, else an informative (recursive) diff object
        """
        if max_recursion == 0:
            return RecursionLimit
        if isinstance(t1, (np.ndarray, JNP_ARRAY, TORCH_TENSOR)) or isinstance(
            t2, (np.ndarray, JNP_ARRAY, TORCH_TENSOR)
        ):
            return diff_fn(t1, t2)
        # Case 1: Both are mappings (e.g., dictionaries)
        if isinstance(t1, Mapping) and isinstance(t2, Mapping):
            diff = {}
            all_keys = set(t1.keys()).union(set(t2.keys()))
            for key in all_keys:
                val1, val2 = t1.get(key), t2.get(key)
                if key not in t1:
                    diff[key] = {"only_in_tree2": val2}
                elif key not in t2:
                    diff[key] = {"only_in_tree1": val1}
                else:
                    sub_diff = recursive_diff(val1, val2, max_recursion=max_recursion - 1)
                    if sub_diff is not None:
                        diff[key] = sub_diff
            return diff if diff else None

        # Case 2: Both are sequences (e.g., lists, tuples) and of the same type
        if (
            isinstance(t1, Sequence)
            and isinstance(t2, Sequence)
            and isinstance(t2, type(t1))
            and isinstance(t1, type(t2))
            and not isinstance(t1, str)
        ):
            if len(t1) != len(t2):
                return {"length_mismatch": (len(t1), len(t2))}
            diff = [recursive_diff(x, y, max_recursion=max_recursion - 1) for x, y in zip(t1, t2)]
            diff = [d for d in diff if d is not None]
            return diff if diff else None

        # Case 3: Both are comparable types (e.g., scalars, arrays)
        return diff_fn(t1, t2)

    diff_tree = recursive_diff(tree1, tree2, max_recursion=max_recursion - 1)
    return diff_tree if diff_tree else None
