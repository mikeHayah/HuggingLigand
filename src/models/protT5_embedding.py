import logging
import torch
from tqdm import tqdm

from src.modules.embedding_utils import mean_pool_embedding
from src.modules.loaders import load_prott5_model


class ProtT5Embedder:
    """
    A class to generate protein embeddings using the ProtT5 model.
    
    Parameters
    ----------
    device : str, optional
        The device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.
    model_name : str, optional
        The name of the pre-trained ProtT5 model to load.
        Default is configured in config.ini under [models].protein_model.
    """

    def __init__(self, device: str = "cpu", model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc"):
        """
        Initialize the ProtT5Embedder.

        Parameters
        ----------
        device : str
            The device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.
        model_name : str
            The name of the pre-trained ProtT5 model to load.
            Default is configured in config.ini under [models].protein_model.
        """
        self.device = torch.device(device)
        self.model_name = model_name
        self.tokenizer, self.model = load_prott5_model(self.device, self.model_name)
        self.model.eval()

        if self.device.type == "cpu":
            torch.set_num_threads(torch.get_num_threads())
            torch.jit.optimized_execution(True)

        self._cache = {}

    def embed(self, sequences: list[str], batch_size: int = 4, show_progress: bool = False) -> list[torch.Tensor]:
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

        Returns
        -------
        list[torch.Tensor]
            A list of tensors where each tensor is the embedding vector for a protein sequence.
        """

        batch_range = range(0, len(sequences), batch_size)

        if show_progress and len(sequences) > batch_size:
            batch_range = tqdm(batch_range, desc="Processing batches", unit="batch", leave=False)
            
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

            # Clear intermediate variables to free memory
            del inputs, input_ids, attention_mask, outputs, embeddings
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # Clear cache periodically to prevent unlimited memory growth
            if len(self._cache) > 10 ** 6:  # Adjust this threshold as needed
                logging.info("Clearing embedding cache to free memory...")
                self._cache.clear()

        return all_embeddings