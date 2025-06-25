import sys
import os
import logging
import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


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

    def embed(self, sequences: list[str], batch_size: int = 4) -> pd.DataFrame:
        """
        Generates mean-pooled ProtT5 embeddings for the given protein sequences in batches.

        Parameters
        ----------
        sequences : list of str
            A list of raw amino acid sequences.
        batch_size : int
            Number of sequences to process at a time to reduce memory usage.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each row is the embedding vector for a protein sequence.
        """
        sequences = sequences.iloc[:, 0].tolist()  # Assume input is always a DataFrame

        sequences = sequences[:1000]  # LIMIT FOR DEBUGGING

        all_embeddings = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            formatted_seqs = [" ".join(list(seq.strip())) for seq in batch]
            inputs = self.tokenizer(formatted_seqs, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state

            for seq, emb, mask in zip(batch, embeddings, attention_mask):
                if seq in self._cache:
                    all_embeddings.append(self._cache[seq])
                else:
                    mean_emb = mean_pool_embedding(emb, mask).cpu()
                    mean_emb_list = mean_emb.tolist()
                    self._cache[seq] = mean_emb_list
                    all_embeddings.append(mean_emb_list)

            torch.cuda.empty_cache()

        return pd.DataFrame(all_embeddings)
