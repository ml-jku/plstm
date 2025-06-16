import torch


def rev_cumsum_off(x):
    y = torch.zeros_like(x)
    y[..., :-1] = x[..., 1:]
    return y.flip(dims=(-1,)).cumsum(dim=-1).flip(dims=(-1,))


def cumsum_off(x):
    y = torch.zeros_like(x)
    y[..., :-1] = x[..., 1:]
    return y.cumsum(dim=-1)


def rev_cumsum(x):
    return x.flip(dims=(-1,)).cumsum(-1).flip(dims=(-1,))


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def plus(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y


def count_parameters(mod: torch.nn.Module) -> int:
    num_pars = 0
    for _, par in mod.named_parameters(recurse=True):
        num_pars += par.numel()
    return num_pars
