import torch


def mean_pool_embedding(embedding: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean-pooled embedding of a sequence based on the attention mask.

    Parameters
    ----------
    embedding : torch.Tensor
        [seq_len, hidden_dim] tensor of token embeddings.

    mask : torch.Tensor
        [seq_len] tensor with 1s for valid tokens and 0s for padding.

    Returns
    -------
    torch.Tensor
        [hidden_dim] mean-pooled embedding.
    """
    return embedding[:mask.sum()].mean(dim=0)