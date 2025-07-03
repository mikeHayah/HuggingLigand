import logging
import torch
from tqdm import tqdm

from src.modules.embedding_utils import mean_pool_embedding
from src.modules.loaders import load_prott5_model

logger = logging.getLogger(__name__)


class ProtT5Embedder:
    """
    A class to generate protein embeddings using the ProtT5 model.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize the ProtT5Embedder.

        Args:
            device (str): The device to run the model on ('cpu' or 'cuda').
        """
        self.device = torch.device(device)
        self.tokenizer, self.model = load_prott5_model(self.device)
        self.model.eval()

        if self.device.type == "cpu":
            torch.set_num_threads(torch.get_num_threads())
            torch.jit.optimized_execution(True)

        self._cache = {}

    def embed(self, sequences: list[str], batch_size: int = 4, show_progress: bool = False, full_data: bool = False) -> list[torch.Tensor]:
        """
        Generates mean-pooled ProtT5 embeddings for the given protein sequences in batches.

        Parameters
        ----------
        sequences : list of str
            A list of raw amino acid sequences.
        batch_size : int
            Number of sequences to process at a time to reduce memory usage.
        show_progress : bool
            Whether to display a progress bar during embedding.
        full_data : bool
            Whether to use the full dataset or limit to 1000 sequences.

        Returns
        -------
        list[torch.Tensor]
            A list of tensors where each tensor is the embedding vector for a protein sequence.
        """
        
        if not full_data:
            sequences = sequences[:100]

        if show_progress and len(sequences) > batch_size:
            batch_range = tqdm(range(0, len(sequences), batch_size), desc="Processing batches", unit="batch", leave=False)
        else:
            batch_range = range(0, len(sequences), batch_size)

        all_embeddings = []

        for i in batch_range:
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
                    self._cache[seq] = mean_emb
                    all_embeddings.append(mean_emb)

            torch.cuda.empty_cache()

        return all_embeddings