import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.modules.embedding_utils import mean_pool_embedding
from src.modules.loaders import load_chemberta_model


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
        
        # CPU optimizations
        if device == "cpu":
            # Use optimized BLAS operations
            torch.set_num_threads(torch.get_num_threads())  # Use all available CPU cores
            # Set model to eval mode and optimize for inference
            self.model.eval()
            # Enable JIT compilation for CPU
            torch.jit.optimized_execution(True)
        
        self._cache = {}

    def embed(self, smiles_list: list[str], batch_size: int = 32, show_progress: bool = False) -> list[torch.Tensor]:
        """
        Get embeddings for a list of SMILES strings.

        Args:
            smiles_list (list of str): List of SMILES strings.
            batch_size (int): Number of sequences to process at once. Default is 32.
            show_progress (bool): Whether to show progress bar for batches.

        Returns:
            list of torch.Tensor: Mean-pooled sequence embeddings, one per input.
        """
        all_embeddings = []
        
        # Create progress bar for batches if requested
        batch_range = range(0, len(smiles_list), batch_size)
        if show_progress and len(smiles_list) > batch_size:
            batch_range = tqdm(batch_range, desc="Processing batches", unit="batch", leave=False)
        
        # Process in batches to avoid memory issues
        for i in batch_range:
            batch = smiles_list[i:i + batch_size]
            batch_embeddings = []
            
            # Check cache first for this batch
            uncached_smiles = []
            uncached_indices = []
            
            for j, seq in enumerate(batch):
                if seq in self._cache:
                    batch_embeddings.append(self._cache[seq])
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
                    max_length=512,  # Limit max length for speed
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

                # Process uncached embeddings
                uncached_embeddings = []
                for seq, emb, mask in zip(uncached_smiles, last_hidden_state, attention_mask, strict=False):
                    mean_emb = mean_pool_embedding(emb, mask)
                    self._cache[seq] = mean_emb
                    uncached_embeddings.append(mean_emb)
                
                # Insert uncached embeddings back into their correct positions
                uncached_iter = iter(uncached_embeddings)
                for j in uncached_indices:
                    batch_embeddings.insert(j, next(uncached_iter))
                
                # Clear intermediate variables to free memory
                del inputs, input_ids, attention_mask, outputs, last_hidden_state
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            
            all_embeddings.extend(batch_embeddings)
            
            # Clear cache periodically to prevent unlimited memory growth
            if len(self._cache) > 10000:  # Adjust this threshold as needed
                if not show_progress:  # Only print if not using progress bar
                    print("Clearing embedding cache to free memory...")
                self._cache.clear()

        return all_embeddings