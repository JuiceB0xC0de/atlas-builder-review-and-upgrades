import torch

from qwip_atlas.tensor_utils import mean_real_tokens_torch


def test_mean_real_tokens_torch_pools_per_head_tensor():
    x = torch.arange(2 * 4 * 3 * 2, dtype=torch.float32).reshape(2, 4, 3, 2)
    attention_mask = torch.tensor(
        [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=torch.float32,
    )

    pooled = mean_real_tokens_torch(x, attention_mask)

    expected = torch.stack(
        [
            x[0, 1:].mean(dim=0),
            x[1, 2:].mean(dim=0),
        ],
        dim=0,
    )
    assert pooled.shape == (2, 3, 2)
    torch.testing.assert_close(pooled, expected)
