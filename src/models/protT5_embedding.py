import logging
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modules.embedding_utils import mean_pool_embedding
from src.modules.loaders import load_prott5_model

logger = logging.getLogger(__name__)


class ProtT5Embedder:
    """
    Generate protein embeddings using the ProtT5-XL-UniRef50 model.

    Parameters
    ----------
    device : str, optional
        Torch device to use ('cuda' or 'cpu'). Default is 'cpu'.

    Attributes
    ----------
    tokenizer : T5Tokenizer
        Tokenizer for the ProtT5 model.
    model : T5EncoderModel
        Encoder model for generating embeddings.
    device : torch.device
        Computation device.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        self.tokenizer, self.model = load_prott5_model(self.device)
        self._cache = {}

    def embed(self, sequences: list[str]) -> list[torch.Tensor]:
        """
        Embed a list of protein sequences.

        Parameters
        ----------
        sequences : list of str
            Raw amino acid sequences.

        Returns
        -------
        list of torch.Tensor
            Mean-pooled sequence embeddings, one per input.
        """
        formatted_seqs = [" ".join(list(seq.strip())) for seq in sequences]
        inputs = self.tokenizer(formatted_seqs, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

        pooled = []
        for seq, emb, mask in zip(sequences, embeddings, attention_mask, strict=False):
            if seq in self._cache:
                pooled.append(self._cache[seq])
            else:
                mean_emb = mean_pool_embedding(emb, mask)
                self._cache[seq] = mean_emb
                pooled.append(mean_emb)

        return pooled