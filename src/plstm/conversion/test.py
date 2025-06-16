"""Tests for parameter conversion between JAX (nnx/linen) and PyTorch."""

import torch
import jax
import numpy as np
from plstm.nnx_dummy import nnx
from flax import linen as nn

from plstm.nnx.util import module_named_params as nnx_module_named_params
from plstm.linen.util import module_named_params as linen_module_named_params


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def get_param_stats(param: torch.Tensor | jax.Array | np.ndarray) -> tuple[float, float, float, float]:
    """Get statistical properties of a parameter tensor.

    Args:
        param: Parameter tensor from PyTorch, JAX, or NumPy

    Returns:
        Tuple of (mean, std, first_element, last_element)
    """
    if isinstance(param, torch.Tensor):
        param = param.detach().cpu().numpy()
    elif isinstance(param, jax.Array):
        param = np.array(param)

    flat_param = param.reshape(-1)
    return (
        float(np.mean(param)),
        float(np.std(param)),
        float(flat_param[0]),  # first element
        float(flat_param[-1]),  # last element
    )


def find_best_match(name: str, available_names: list[str]) -> tuple[str, int]:
    """Find the name with minimum edit distance from available names."""
    distances = [(n, levenshtein_distance(name, n)) for n in available_names]
    return min(distances, key=lambda x: x[1])


def compare_torch_nnx_parameters(
    torch_module: torch.nn.Module,
    nnx_module: nnx.Module,
    max_edit_dist: int = 10,
) -> dict[str, tuple[str, int, float, float, float, float]]:
    """Compare parameters between PyTorch and NNX modules.

    Args:
        torch_module: PyTorch module
        nnx_module: NNX module
        max_edit_dist: Maximum edit distance for parameter name matching

    Returns:
        Dictionary mapping torch parameter names to tuples of:
        (jax_name, distance, mean_diff, std_diff, first_diff, last_diff)
        where *_diff values are absolute differences between torch and jax parameters
    """
    # Get parameters from both modules
    torch_params = {name: param for name, param in torch_module.named_parameters() if param is not None}
    jax_params = {name: param.value for name, param in nnx_module_named_params(nnx_module, recursive=True)}

    # First check parameter counts match
    if len(torch_params) != len(jax_params):
        jax_names = list(jax_params.keys())
        torch_names = list(torch_params.keys())
        # remove matching first
        for torch_name, torch_param in torch_params.items():
            if not jax_names:
                break
            if torch_name in jax_names:
                jax_names.remove(torch_name)  # Remove matched name from candidates
                torch_names.remove(torch_name)
        for torch_name in list(torch_names):
            # Find best matching jax parameter name
            if not jax_names:
                break
            jax_name, distance = find_best_match(torch_name, jax_names)
            if distance > max_edit_dist:
                jax_names.remove(jax_name)  # Remove matched name from candidates
                torch_names.remove(torch_name)

        raise ValueError(
            f"Number of parameters doesn't match: torch={len(torch_params)}, nnx={len(jax_params)}\n"
            f"Non-matching torch params: {sorted(torch_names)}\n"
            f"Non-matching nnx params: {sorted(jax_names)}"
        )

    # Match parameters and compute differences
    results = {}
    jax_names = list(jax_params.keys())
    torch_names = list(torch_params.keys())
    left_torch_names = []

    for torch_name in torch_names:
        # Find best matching jax parameter name
        if torch_name in jax_names:
            jax_names.remove(torch_name)  # Remove matched name from candidates

            # Get statistical properties
            torch_stats = get_param_stats(torch_params[torch_name])
            jax_stats = get_param_stats(jax_params[torch_name])

            # Compute absolute differences
            diffs = tuple(abs(t - j) for t, j in zip(torch_stats, jax_stats))

            results[torch_name] = (torch_name, 0, *diffs)
        else:
            left_torch_names.append(torch_name)

    for torch_name in left_torch_names:
        # Find best matching jax parameter name
        jax_name, distance = find_best_match(torch_name, jax_names)
        if distance < max_edit_dist:
            jax_names.remove(jax_name)  # Remove matched name from candidates

            # Get statistical properties
            torch_stats = get_param_stats(torch_params[torch_name])
            jax_stats = get_param_stats(jax_params[jax_name])

            # Compute absolute differences
            diffs = tuple(abs(t - j) for t, j in zip(torch_stats, jax_stats))

            results[torch_name] = (jax_name, distance, *diffs)

    return results


def compare_torch_linen_parameters(
    torch_module: torch.nn.Module,
    linen_module: nn.Module,
    variables: dict,
    max_edit_dist: int = 10,
) -> dict[str, tuple[str, int, float, float, float, float]]:
    """Compare parameters between PyTorch and Linen modules.

    Args:
        torch_module: PyTorch module
        linen_module: Linen module
        variables: Linen variables dict
        max_edit_dist: Maximum edit distance for parameter name matching

    Returns:
        Dictionary mapping torch parameter names to tuples of:
        (linen_name, distance, mean_diff, std_diff, first_diff, last_diff)
        where *_diff values are absolute differences between torch and linen parameters
    """
    # Get parameters from both modules
    torch_params = {name: param for name, param in torch_module.named_parameters() if param is not None}
    linen_params = {name: param for name, param in linen_module_named_params(linen_module, variables, recursive=True)}

    # First check parameter counts match
    if len(torch_params) != len(linen_params):
        linen_names = list(linen_params.keys())
        torch_names = list(torch_params.keys())
        # remove matching first
        for torch_name, torch_param in torch_params.items():
            if not linen_names:
                break
            if torch_name in linen_names:
                linen_names.remove(torch_name)  # Remove matched name from candidates
                torch_names.remove(torch_name)
        for torch_name in list(torch_names):
            # Find best matching linen parameter name
            if not linen_names:
                break
            linen_name, distance = find_best_match(torch_name, linen_names)
            if distance > max_edit_dist:
                linen_names.remove(linen_name)  # Remove matched name from candidates
                torch_names.remove(torch_name)

        raise ValueError(
            f"Number of parameters doesn't match: torch={len(torch_params)}, linen={len(linen_params)}\n"
            f"Non-matching torch params: {sorted(torch_names)}\n"
            f"Non-matching linen params: {sorted(linen_names)}"
        )

    # Match parameters and compute differences
    results = {}
    linen_names = list(linen_params.keys())
    torch_names = list(torch_params.keys())
    left_torch_names = []

    for torch_name in torch_names:
        # Find best matching linen parameter name
        if torch_name in linen_names:
            linen_names.remove(torch_name)  # Remove matched name from candidates

            # Get statistical properties
            torch_stats = get_param_stats(torch_params[torch_name])
            linen_stats = get_param_stats(linen_params[torch_name])

            # Compute absolute differences
            diffs = tuple(abs(t - j) for t, j in zip(torch_stats, linen_stats))

            results[torch_name] = (torch_name, 0, *diffs)
        else:
            left_torch_names.append(torch_name)

    for torch_name in left_torch_names:
        # Find best matching linen parameter name
        linen_name, distance = find_best_match(torch_name, linen_names)
        if distance < max_edit_dist:
            linen_names.remove(linen_name)  # Remove matched name from candidates

            # Get statistical properties
            torch_stats = get_param_stats(torch_params[torch_name])
            linen_stats = get_param_stats(linen_params[linen_name])

            # Compute absolute differences
            diffs = tuple(abs(t - j) for t, j in zip(torch_stats, linen_stats))

            results[torch_name] = (linen_name, distance, *diffs)

    return results


def compare_linen_nnx_parameters(
    linen_module: nn.Module,
    nnx_module: nnx.Module,
    variables: dict,
    max_edit_dist: int = 10,
) -> dict[str, tuple[str, int, float, float, float, float]]:
    """Compare parameters between Linen and NNX modules.

    Args:
        linen_module: Linen module
        nnx_module: NNX module
        variables: Linen variables dict
        max_edit_dist: Maximum edit distance for parameter name matching

    Returns:
        Dictionary mapping linen parameter names to tuples of:
        (nnx_name, distance, mean_diff, std_diff, first_diff, last_diff)
        where *_diff values are absolute differences between linen and nnx parameters
    """
    # Get parameters from both modules
    linen_params = {name: param for name, param in linen_module_named_params(linen_module, variables, recursive=True)}
    nnx_params = {name: param.value for name, param in nnx_module_named_params(nnx_module, recursive=True)}

    # First check parameter counts match
    if len(linen_params) != len(nnx_params):
        linen_names = list(linen_params.keys())
        nnx_names = list(nnx_params.keys())
        # remove matching first
        for linen_name, linen_param in linen_params.items():
            if not nnx_names:
                break
            if linen_name in nnx_names:
                nnx_names.remove(linen_name)  # Remove matched name from candidates
                linen_names.remove(linen_name)
        for linen_name in list(linen_names):
            # Find best matching nnx parameter name
            if not nnx_names:
                break
            nnx_name, distance = find_best_match(linen_name, nnx_names)
            if distance > max_edit_dist:
                nnx_names.remove(nnx_name)  # Remove matched name from candidates
                linen_names.remove(linen_name)

        raise ValueError(
            f"Number of parameters doesn't match: linen={len(linen_params)}, nnx={len(nnx_params)}\n"
            f"Non-matching linen params: {sorted(linen_names)}\n"
            f"Non-matching nnx params: {sorted(nnx_names)}"
        )

    # Match parameters and compute differences
    results = {}
    nnx_names = list(nnx_params.keys())
    linen_names = list(linen_params.keys())
    left_linen_names = []

    for linen_name in linen_names:
        # Find best matching nnx parameter name
        if linen_name in nnx_names:
            nnx_names.remove(linen_name)  # Remove matched name from candidates

            # Get statistical properties
            linen_stats = get_param_stats(linen_params[linen_name])
            nnx_stats = get_param_stats(nnx_params[linen_name])

            # Compute absolute differences
            diffs = tuple(abs(lstat - nstat) for lstat, nstat in zip(linen_stats, nnx_stats))

            results[linen_name] = (linen_name, 0, *diffs)
        else:
            left_linen_names.append(linen_name)

    for linen_name in left_linen_names:
        # Find best matching nnx parameter name
        nnx_name, distance = find_best_match(linen_name, nnx_names)
        if distance < max_edit_dist:
            nnx_names.remove(nnx_name)  # Remove matched name from candidates

            # Get statistical properties
            linen_stats = get_param_stats(linen_params[linen_name])
            nnx_stats = get_param_stats(nnx_params[nnx_name])

            # Compute absolute differences
            diffs = tuple(abs(lstat - nstat) for lstat, nstat in zip(linen_stats, nnx_stats))

            results[linen_name] = (nnx_name, distance, *diffs)

    return results


def assert_torch_nnx_parameters_match(torch_module: torch.nn.Module, nnx_module: nnx.Module, rtol: float = 1e-5):
    """Assert that parameters between PyTorch and NNX modules match within
    tolerance.

    Args:
        torch_module: PyTorch module
        nnx_module: NNX module
        rtol: Relative tolerance for parameter value comparisons

    Raises:
        AssertionError: If parameters don't match within tolerance
    """
    results = compare_torch_nnx_parameters(torch_module=torch_module, nnx_module=nnx_module)

    # Check for any significant differences
    errors = []
    for torch_name, (nnx_name, distance, mean_diff, std_diff, first_diff, last_diff) in results.items():
        if any(diff > rtol for diff in (mean_diff, std_diff, first_diff, last_diff)):
            errors.append(
                f"\nParameter mismatch:\n"
                f"  PyTorch name: {torch_name}\n"
                f"  NNX name: {nnx_name} (edit distance: {distance})\n"
                f"  Differences:\n"
                f"    Mean: {mean_diff:.2e}\n"
                f"    Std: {std_diff:.2e}\n"
                f"    First element: {first_diff:.2e}\n"
                f"    Last element: {last_diff:.2e}"
            )

    if errors:
        raise AssertionError("Parameter mismatches found:" + "".join(errors))


def assert_torch_linen_parameters_match(
    torch_module: torch.nn.Module, linen_module: nn.Module, variables: dict, rtol: float = 1e-5
):
    """Assert that parameters between PyTorch and Linen modules match within
    tolerance.

    Args:
        torch_module: PyTorch module
        linen_module: Linen module
        variables: Linen variables dict
        rtol: Relative tolerance for parameter value comparisons

    Raises:
        AssertionError: If parameters don't match within tolerance
    """
    results = compare_torch_linen_parameters(torch_module=torch_module, linen_module=linen_module, variables=variables)

    # Check for any significant differences
    errors = []
    for torch_name, (linen_name, distance, mean_diff, std_diff, first_diff, last_diff) in results.items():
        if any(diff > rtol for diff in (mean_diff, std_diff, first_diff, last_diff)):
            errors.append(
                f"\nParameter mismatch:\n"
                f"  PyTorch name: {torch_name}\n"
                f"  Linen name: {linen_name} (edit distance: {distance})\n"
                f"  Differences:\n"
                f"    Mean: {mean_diff:.2e}\n"
                f"    Std: {std_diff:.2e}\n"
                f"    First element: {first_diff:.2e}\n"
                f"    Last element: {last_diff:.2e}"
            )

    if errors:
        raise AssertionError("Parameter mismatches found:" + "".join(errors))


def assert_linen_nnx_parameters_match(
    linen_module: nn.Module, nnx_module: nnx.Module, variables: dict, rtol: float = 1e-5
):
    """Assert that parameters between Linen and NNX modules match within
    tolerance.

    Args:
        linen_module: Linen module
        nnx_module: NNX module
        variables: Linen variables dict
        rtol: Relative tolerance for parameter value comparisons

    Raises:
        AssertionError: If parameters don't match within tolerance
    """
    results = compare_linen_nnx_parameters(linen_module=linen_module, nnx_module=nnx_module, variables=variables)

    # Check for any significant differences
    errors = []
    for linen_name, (nnx_name, distance, mean_diff, std_diff, first_diff, last_diff) in results.items():
        if any(diff > rtol for diff in (mean_diff, std_diff, first_diff, last_diff)):
            errors.append(
                f"\nParameter mismatch:\n"
                f"  Linen name: {linen_name}\n"
                f"  NNX name: {nnx_name} (edit distance: {distance})\n"
                f"  Differences:\n"
                f"    Mean: {mean_diff:.2e}\n"
                f"    Std: {std_diff:.2e}\n"
                f"    First element: {first_diff:.2e}\n"
                f"    Last element: {last_diff:.2e}"
            )

    if errors:
        raise AssertionError("Parameter mismatches found:" + "".join(errors))


# Backward compatibility
def compare_module_parameters(
    nnx_module: nnx.Module,
    torch_module: torch.nn.Module,
    max_edit_dist: int = 10,
) -> dict[str, tuple[str, int, float, float, float, float]]:
    """Compare parameters between torch and jax modules (backward
    compatibility).

    Args:
        nnx_module: NNX module
        torch_module: PyTorch module
        max_edit_dist: Maximum edit distance for parameter name matching

    Returns:
        Dictionary mapping torch parameter names to tuples of:
        (jax_name, distance, mean_diff, std_diff, first_diff, last_diff)
        where *_diff values are absolute differences between torch and jax parameters
    """
    return compare_torch_nnx_parameters(torch_module=torch_module, nnx_module=nnx_module, max_edit_dist=max_edit_dist)


def assert_parameters_match(nnx_module: nnx.Module, torch_module: torch.nn.Module, rtol: float = 1e-5):
    """Assert that parameters between torch and jax modules match within
    tolerance (backward compatibility).

    Args:
        nnx_module: NNX module
        torch_module: PyTorch module
        rtol: Relative tolerance for parameter value comparisons

    Raises:
        AssertionError: If parameters don't match within tolerance
    """
    assert_torch_nnx_parameters_match(torch_module=torch_module, nnx_module=nnx_module, rtol=rtol)
