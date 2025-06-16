import torch
from ..config.dtype import DTYPES


def str_dtype_to_torch(dtype: DTYPES) -> torch.dtype:
    return getattr(torch, dtype)
