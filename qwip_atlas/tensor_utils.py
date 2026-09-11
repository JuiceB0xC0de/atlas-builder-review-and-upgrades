"""
tensor_utils.py
---------------
Shared tensor helpers used by the census extractor, Sub-Zero probe, and
any other module that needs to pool over variable-length sequences.
"""

from __future__ import annotations

from typing import Any


def slice_and_mean(tensor: Any, seq_lens: list[int]) -> tuple[Any, Any]:
    """Vectorized last-token and variable-length mean over a batch.

    Input tensor has shape [B, max_seq_len, ...] after slicing to the longest
    real sequence. Padding is assumed left-aligned, so real tokens are at the
    end. Returns (last_token, mean_tokens) both shaped [B, ...].
    """
    import numpy as np

    B = len(seq_lens)
    max_seq_len = tensor.shape[1]
    last = tensor[:, -1, ...]

    # Build a per-example length mask.
    mask = np.zeros((B, max_seq_len), dtype=bool)
    for i, length in enumerate(seq_lens):
        mask[i, -length:] = True

    # Expand mask to broadcast against arbitrary trailing dims.
    expand_axes = tuple(range(2, tensor.ndim))
    if expand_axes:
        mask = np.expand_dims(mask, axis=expand_axes)
    masked = tensor * mask
    summed = masked.sum(axis=1)
    mean = summed / np.array(seq_lens).reshape((B,) + (1,) * (tensor.ndim - 2))
    return last, mean


def mean_real_tokens_torch(x, attention_mask):
    """Mean-pool a torch tensor over the real tokens in each example.

    Args:
        x: [B, seq_len, ...] torch.Tensor.
        attention_mask: [B, seq_len] torch.Tensor (1 = real token).

    Returns:
        [B, ...] torch.Tensor with the mean over real positions.
    """
    import torch

    # Expand attention mask to match x's trailing dims.
    shape = list(attention_mask.shape) + [1] * (x.ndim - attention_mask.ndim)
    mask = attention_mask.reshape(shape).to(x.dtype)
    masked = x * mask
    summed = masked.sum(dim=1)
    lengths = attention_mask.sum(dim=1).clamp(min=1).to(x.dtype)
    lengths = lengths.reshape((x.shape[0],) + (1,) * (x.ndim - 2))
    return summed / lengths


def last_real_token_torch(x, attention_mask):
    """Return the last real token for each example in a batch.

    Args:
        x: [B, seq_len, ...] torch.Tensor.
        attention_mask: [B, seq_len] torch.Tensor (1 = real token).

    Returns:
        [B, ...] torch.Tensor with the last real token slice.
    """
    import torch

    last_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0)
    bs = x.shape[0]
    return x[torch.arange(bs, device=x.device), last_idx.to(x.device)]
