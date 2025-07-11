import torch

from src.models.utils.embedding_utils import mean_pool_embedding


def test_mean_pool_embedding():
    # Simulate a sequence of 5 tokens, each with 4-dim embedding
    embedding = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],  # valid
            [2.0, 2.0, 2.0, 2.0],  # valid
            [3.0, 3.0, 3.0, 3.0],  # valid
            [0.0, 0.0, 0.0, 0.0],  # padding
            [0.0, 0.0, 0.0, 0.0],  # padding
        ]
    )
    mask = torch.tensor([1, 1, 1, 0, 0])  # 3 valid tokens

    pooled = mean_pool_embedding(embedding, mask)

    expected = torch.tensor([2.0, 2.0, 2.0, 2.0])  # mean of [1,2,3]
    assert torch.allclose(pooled, expected), "Mean pooling output is incorrect"

    print("mean_pool_embedding passed basic test.")
