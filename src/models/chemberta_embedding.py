import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modules.embedding_utils import mean_pool_embedding
from src.modules.loader import load_chemberta_model


class ChembertaModel:
    """
    A class to handle the ChemBERTa model for generating embeddings from SMILES strings.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize the ChembertaModel with a specified model name and device.

        Args:
            model_name (str): The name of the pre-trained ChemBERTa model.
            device (str): The device to run the model on, e.g., "cpu" or "cuda".
        """
        self.device = torch.device(device)
        self.tokenizer, self.model = load_chemberta_model(self.device)
        self._cache = {}

    def embed(self, smiles_list: list[str]) -> list[torch.Tensor]:
        """
        Get embeddings for a list of SMILES strings.

        Args:
            smiles_list (list of str): List of SMILES strings.

        Returns:
            list of torch.Tensor: Mean-pooled sequence embeddings, one per input.
        """
        inputs = self.tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
            )

        last_hidden_state = outputs.hidden_states[-1]

        pooled = []
        for seq, emb, mask in zip(smiles_list, last_hidden_state, attention_mask, strict=False):
            if seq in self._cache:
                pooled.append(self._cache[seq])
            else:
                mean_emb = mean_pool_embedding(emb, mask)
                self._cache[seq] = mean_emb
                pooled.append(mean_emb)

        return pooled
