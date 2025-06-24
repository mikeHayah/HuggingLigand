import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import logging
import pandas as pd

import torch

from modules.embedding_utils import mean_pool_embedding
from modules.loaders import load_prott5_model

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

    def embed(self) -> list[torch.Tensor]:
        """
        Reads protein sequences from a relative CSV file and returns their mean-pooled ProtT5 embeddings.

        Returns
        -------
        list of torch.Tensor
            Mean-pooled sequence embeddings for each protein sequence in the input file.
        """
        csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "embeddings", "proteins_embeddings.csv")
        csv_path = os.path.abspath(csv_path)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        if 'BindingDB Target Chain Sequence' not in df.columns:
            raise ValueError("CSV must contain a 'BindingDB Target Chain Sequence' column.")

        df = pd.read_csv(csv_path)
        sequences = df['BindingDB Target Chain Sequence'].dropna().tolist()

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