"""Test utilities for JAX."""

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_array_diff(x: np.ndarray, y: np.ndarray, title: str = "", **kwargs):
    """Plot absolute difference between two arrays."""
    shape = x.shape
    fig, ax = plt.subplots()
    ntot = x.size
    nrun = 1
    for s in x.shape:
        if nrun * s > ntot**0.7 and nrun > 1:
            break
        nrun *= s

    x_reshaped = x.reshape(nrun, -1)
    y_reshaped = y.reshape(nrun, -1)
    im = ax.imshow(np.abs(x_reshaped - y_reshaped), vmin=0, **kwargs)
    ax.set_title(f"{title}\nSHAPE: {shape}")
    fig.colorbar(im, ax=ax)
    return fig


def plot_array_diff_rel(x: np.ndarray, y: np.ndarray, title: str = "", **kwargs):
    """Plot relative difference between two arrays."""
    shape = x.shape
    fig, ax = plt.subplots()
    ntot = x.size
    nrun = 1
    for s in x.shape:
        if nrun * s > ntot**0.7 and nrun > 1:
            break
        nrun *= s

    x_reshaped = x.reshape(nrun, -1)
    y_reshaped = y.reshape(nrun, -1)
    rel_diff = np.abs(x_reshaped - y_reshaped) / (np.abs(x_reshaped) + 1e-5)
    im = ax.imshow(rel_diff, vmax=1.0, **kwargs)
    ax.set_title(f"{title}\nSHAPE: {shape}")
    fig.colorbar(im, ax=ax)
    return fig


def assert_allclose_with_plot(
    actual: np.ndarray,
    desired: np.ndarray,
    rtol: float = 1e-7,
    atol: float = 0,
    err_msg: str = "",
    base_path: str = "error",
    **kwargs,
):
    """Assert two arrays are close with error visualization on failure.

    On failure, saves plots of absolute and relative differences to
    plstm/tests_outputs/[test_name]_[abs/rel]_diff.png

    Args:
        actual: Array to test
        desired: Expected array
        rtol: Relative tolerance
        atol: Absolute tolerance
        err_msg: Error message
        test_file: Path to test file (__file__)
        test_name: Name of test function (from extract_test_name(inspect.currentframe()))
        base_path: Path either absolute or relative to "tests/" for the outputs to be saved
        **kwargs: Additional arguments passed to plotting functions
    """
    try:
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, err_msg=err_msg)
    except AssertionError as e:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.split(base_path)[0], exist_ok=True)

        if actual.shape != desired.shape:
            raise e

        # Save absolute difference plot
        fig_abs = plot_array_diff(actual, desired, title="Absolute Difference", **kwargs)
        abs_path = f"{base_path}_abs_diff.png"
        fig_abs.savefig(abs_path)
        plt.close(fig_abs)

        # Save relative difference plot
        fig_rel = plot_array_diff_rel(actual, desired, title="Relative Difference", **kwargs)
        rel_path = f"{base_path}_rel_diff.png"
        fig_rel.savefig(rel_path)
        plt.close(fig_rel)

        # Re-raise the assertion error with the original message and plot paths
        raise AssertionError(f"{str(e)}\nDifference plots saved to:\n  {abs_path}\n  {rel_path}") from None
