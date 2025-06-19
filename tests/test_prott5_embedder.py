import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import torch

from models.protT5_embedding import ProtT5Embedder


def test_embedder():
    # Example protein sequence from UniProt (HIV protease)
    sequences = [
        "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
    ]

    embedder = ProtT5Embedder(device="cpu")  # Use "cuda" if available
    embeddings = embedder.embed(sequences)

    assert isinstance(embeddings, list), "Output should be a list"
    assert isinstance(embeddings[0], torch.Tensor), "Each item should be a torch.Tensor"
    assert embeddings[0].shape == (1024,), "Each embedding should be 1024-dimensional"

    print("Test passed. Embedding shape:", embeddings[0].shape)

if __name__ == "__main__":
    test_embedder()