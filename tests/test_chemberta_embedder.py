import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.chemberta_embedding import ChembertaModel

@pytest.mark.slow
def test_single_embedding():
    sequences = [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" # caffeine
    ]

    embedder = ChembertaModel(device="cpu")
    embeddings = embedder.embed(sequences)

    assert isinstance(embeddings, list), "Output should be a list"
    assert isinstance(embeddings[0], torch.Tensor), "Each item should be a torch.Tensor"
    assert embeddings[0].shape == torch.Size([768]), "Each embedding should be 768-dimensional"

    print("Test passed. Embedding shape:", embeddings[0].shape)

@pytest.mark.slow
def test_multiple_embeddings():
    sequences = [
        "C==C==O",
        "C1=CC=CC=C1",
        "CCO"
    ]

    embedder = ChembertaModel(device="cpu")
    embeddings = embedder.embed(sequences)

    assert isinstance(embeddings, list), "Output should be a list"
    assert isinstance(embeddings[0], torch.Tensor), "Each item should be a torch.Tensor"
    assert len(embeddings) == 3, "For each sequence an embedding should be returned"
    assert embeddings[0].shape == torch.Size([768]), "Each embedding should be 768-dimensional"

    print("Test passed. Embedding shape:", embeddings[0].shape)