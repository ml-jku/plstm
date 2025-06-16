from collections.abc import Callable
import sys
import torch


def plot_diff(x: torch.Tensor, y: torch.Tensor, **kwargs):
    import matplotlib.pyplot as plt

    shape = x.shape
    fig, ax = plt.subplots()
    ntot = x.numel()
    nrun = 1
    for s in x.shape:
        if nrun * s > ntot**0.7 and nrun > 1:
            break
        nrun *= s

    x = x.reshape(nrun, -1)
    y = y.reshape(nrun, -1)
    im = ax.imshow((x - y).abs().detach(), **kwargs)
    ax.set_title(f"SHAPE: {shape}")
    fig.colorbar(im, ax=ax)


def plot_diff_rel(x: torch.Tensor, y: torch.Tensor, **kwargs):
    import matplotlib.pyplot as plt

    shape = x.shape
    fig, ax = plt.subplots()
    ntot = x.numel()
    nrun = 1
    for s in x.shape:
        if nrun * s > ntot**0.5:
            break
        nrun *= s

    x = x.reshape(nrun, -1)
    y = y.reshape(nrun, -1)
    im = ax.imshow(
        ((x - y).abs() / (x.abs() + 1e-5)).detach(),
        vmax=1.0,
        **kwargs,
    )
    ax.set_title(f"SHAPE: {shape}")
    fig.colorbar(im, ax=ax)


def output_shape(x: torch.Tensor | tuple[torch.Tensor]) -> str:
    if isinstance(x, tuple) or isinstance(x, list):
        return "[" + ", ".join(str(xi.shape) for xi in x) + "]"
    else:
        str(x.shape)


def check_forward(
    func1,
    func2,
    inputs: torch.Tensor | tuple[torch.Tensor],
    verbose=True,
    show_plot_diff: bool = True,
    plot_diff_kwargs: dict = {},
    tensor_compare: Callable[[torch.Tensor, torch.Tensor], bool] = torch.allclose,
    **tensor_compare_args,
):
    if isinstance(inputs, torch.Tensor):
        res1 = func1(inputs)
    else:
        res1 = func1(*inputs)
    if isinstance(inputs, torch.Tensor):
        res2 = func2(inputs)
    else:
        res2 = func2(*inputs)

    if isinstance(res1, tuple) or isinstance(res2, list):
        if len(res1) != len(res2) or not isinstance(res2, tuple) or not isinstance(res2, list):
            if verbose:
                print(
                    f"Invalid output vars: {output_shape(res1)} != {output_shape(res2)}",
                    file=sys.stderr,
                )
            return False
        same = True
        for i, _ in enumerate(res1):
            if res1[i].shape != res2[i].shape:
                if verbose:
                    print(
                        f"Shape mismatch {i}: {res1[i].shape} != {res2[i].shape}",
                        file=sys.stderr,
                    )
                same = False
            if not tensor_compare(res1[i], res2[i], **tensor_compare_args):
                if show_plot_diff:
                    plot_diff(res1[i], res2[i], **plot_diff_kwargs)
                if verbose:
                    print(f"Value mismatch {i}: ", file=sys.stderr)
                same = False
        return same
    else:
        if res1.shape != res2.shape:
            if verbose:
                print(
                    f"Invalid output shape: {output_shape(res1)} != {output_shape(res2)}",
                    file=sys.stderr,
                )
            return False
        if not tensor_compare(res1, res2, **tensor_compare_args):
            if show_plot_diff:
                plot_diff(res1, res2, **plot_diff_kwargs)
                # plot_diff_rel(res1, res2, **plot_diff_kwargs)
            if verbose:
                print("Value mismatch.", file=sys.stderr)
            return False
        return True


def check_backward(
    func1,
    func2,
    inputs: torch.Tensor | tuple[torch.Tensor],
    verbose=True,
    show_plot_diff=True,
    tensor_compare: Callable[[torch.Tensor, torch.Tensor], bool] = torch.allclose,
    plot_diff_kwargs: dict = {},
    rand_factor=1.0,
    **tensor_compare_args,
):
    if isinstance(inputs, torch.Tensor):
        inputs1 = inputs.clone().detach()
        inputs2 = inputs.clone().detach()
        inputs2.requires_grad_(True)
        inputs1.requires_grad_(True)
        res1 = func1(inputs1)
        res2 = func2(inputs2)
    else:
        inputs1 = [inp.clone().detach() if inp is not None else None for inp in inputs]
        inputs2 = [inp.clone().detach() if inp is not None else None for inp in inputs]
        for inp in inputs2:
            if inp is not None:
                inp.requires_grad_(True)
        for inp in inputs1:
            if inp is not None:
                inp.requires_grad_(True)
        res1 = func1(*inputs1)
        res2 = func2(*inputs2)

    # print(res1.grad_fn, inputs[0].requires_grad)
    if isinstance(res1, torch.Tensor):
        masks = torch.randn_like(res1)
        masks2 = masks.clone().detach()
        (masks2 * res2).sum().backward()
        (masks * res1).sum().backward()
    else:
        res1, res2 = list(zip(*[(r1, r2) for r1, r2 in zip(res1, res2) if r1 is not None and r2 is not None]))
        masks = [1.0 + rand_factor * torch.randn_like(y1) for y1 in res1]
        masks2 = [y1.clone().detach() for y1 in masks]
        sum((m1 * y1).mean() for m1, y1 in zip(masks, res1)).backward()
        sum((m2 * y2).mean() for m2, y2 in zip(masks2, res2)).backward()

    same = True
    if isinstance(inputs1, torch.Tensor):
        same = True
        if inputs1.grad is None:
            if verbose:
                print("No grad for func1", file=sys.stderr)
            same = False
        if inputs2.grad is None:
            if verbose:
                print("No grad for func2", file=sys.stderr)
            same = False
        if not tensor_compare(inputs1.grad, inputs2.grad, **tensor_compare_args):
            if verbose:
                print("Value mismatch", file=sys.stderr)
            if show_plot_diff:
                plot_diff(inputs1.grad, inputs2.grad, **plot_diff_kwargs)
            same = False
    else:
        for i, _ in enumerate(inputs1):
            if inputs1[i] is None:
                continue
            if inputs1[i].grad is None and inputs2[i].grad is None:
                continue
            if inputs1[i].grad is None:
                if verbose:
                    print(f"No grad for func1 {i}", file=sys.stderr)
                same = False
            if inputs2[i].grad is None:
                if verbose:
                    print(f"No grad for func2 {i}", file=sys.stderr)
                same = False
            if not tensor_compare(inputs1[i].grad, inputs2[i].grad, **tensor_compare_args):
                if show_plot_diff:
                    plot_diff(inputs1[i].grad, inputs2[i].grad, **plot_diff_kwargs)
                    # plot_diff_rel(
                    #     inputs1[i].grad,
                    #     inputs2[i].grad,
                    #     **plot_diff_kwargs,
                    # )
                if verbose:
                    print(f"Value mismatch {i}.", file=sys.stderr)
                    # print(f"Value mismatch {inputs1[i].grad-inputs2[i].grad}.")
                same = False

    return same
