
import torch
from tqdm import tqdm

from src.modules.embedding_utils import mean_pool_embedding
from src.modules.loaders import load_chemberta_model


class ChembertaEmbedder:
    """
    Generate ligand embeddings using the ChemBERTa model.

    Parameters
    ----------
    device : str, optional
        Torch device to use ('cuda' or 'cpu'). Default is 'cpu'.

    Attributes
    ----------
    device : torch.device
        Computation device.
    tokenizer : AutoTokenizer
        Tokenizer for the ChemBERTa model.
    model : AutoModel
        Model for generating embeddings.
    _cache : dict
        Cache for storing computed embeddings.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize the ChembertaEmbedder with a specified device.

        Parameters
        ----------
        device : str
            The device to run the model on, e.g., "cpu" or "cuda".
        """
        self.device = torch.device(device)
        self.tokenizer, self.model = load_chemberta_model(self.device)
        
        if device == "cpu":
            torch.set_num_threads(torch.get_num_threads())  # Use all available CPU cores
            self.model.eval()
            torch.jit.optimized_execution(True)
        
        self._cache = {}

    def embed(self, smiles_list: list[str], batch_size: int = 32) -> list[torch.Tensor]:
        """
        Get embeddings for a list of SMILES strings.

        Parameters
        ----------
        smiles_list : list of str
            List of SMILES strings.
        batch_size : int
            Number of sequences to process at once. Default is 32.
        show_progress : bool
            Whether to show progress bar for batches.

        Returns
        -------
        list of torch.Tensor
            Mean-pooled sequence embeddings, one per input.
        """
        all_embeddings = []
        
        # Create progress bar for batches if requested
        batch_range = range(0, len(smiles_list), batch_size)
        if show_progress and len(smiles_list) > batch_size:
            batch_range = tqdm(batch_range, desc="Processing batches", unit="batch", leave=False)
        
        # Process in batches to avoid memory issues
        for i in batch_range:
            batch = smiles_list[i:i + batch_size]
            batch_embeddings = [None] * len(batch)  # Pre-allocate with correct size
            
            # Check cache first for this batch
            uncached_smiles = []
            uncached_indices = []
            
            for j, seq in enumerate(batch):
                if seq in self._cache:
                    batch_embeddings[j] = self._cache[seq]  # Place at correct position
                else:
                    uncached_smiles.append(seq)
                    uncached_indices.append(j)
            
            # Process uncached sequences
            if uncached_smiles:
                # Use faster tokenization with optimizations
                inputs = self.tokenizer(
                    uncached_smiles, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    add_special_tokens=True,
                    return_attention_mask=True
                )
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)

                with torch.no_grad():
                    # Use half precision for speed (if supported)
                    with torch.autocast(device_type='cpu', dtype=torch.float16, enabled=True):
                        outputs = self.model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            output_hidden_states=True
                        )

                last_hidden_state = outputs.hidden_states[-1]

                # Process uncached embeddings and place at correct positions
                for idx, (seq, emb, mask) in enumerate(zip(uncached_smiles, last_hidden_state, attention_mask, strict=False)):
                    mean_emb = mean_pool_embedding(emb, mask)
                    self._cache[seq] = mean_emb
                    batch_embeddings[uncached_indices[idx]] = mean_emb  # Place at correct position
                
                # Clear intermediate variables to free memory
                del inputs, input_ids, attention_mask, outputs, last_hidden_state
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            
            all_embeddings.extend(batch_embeddings)
            
            # Clear cache periodically to prevent unlimited memory growth
            if len(self._cache) > 10 ** 6:  # Adjust this threshold as needed
                if not show_progress:  # Only print if not using progress bar
                    print("Clearing embedding cache to free memory...")
                self._cache.clear()

        return all_embeddings