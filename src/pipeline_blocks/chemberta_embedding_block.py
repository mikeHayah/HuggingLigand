import torch
import os
import pandas as pd
import time
from tqdm import tqdm
import glob

from src.models.chemberta_embedding import ChembertaEmbedder

class ChembertaEmbeddingBlock:
    """
    A pipeline block to generate embeddings for ligands using the ChemBERTa model.

    This block takes a list of SMILES strings, processes them in batches to generate
    embeddings, and returns a pandas DataFrame containing the original SMILES
    and their corresponding embeddings. It includes features for handling large
    datasets, such as batch processing, progress tracking, and saving
    intermediate results to manage memory usage.

    Attributes
    ----------
    ligands : list of str
        The input list of SMILES strings to be processed.
    ligands_embd : pd.DataFrame
        A DataFrame containing the SMILES strings and their embeddings after
        the `run` method has been executed.
    """

    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
        """
        Initializes the ChembertaEmbeddingBlock.
        
        Parameters
        ----------
        model_name : str
            The name of the pre-trained ChemBERTa model to use.
            Default is configured in config.ini under [models].ligand_model.
        """
        self.ligands = None
        self.ligands_embd = None
        self.model_name = model_name


    def set_input(self, ligands: list[str]):
        """
        Sets the input data for the embedding block.

        Parameters
        ----------
        ligands : list of str
            A list of SMILES strings representing the ligands.
        """
        self.ligands = ligands


    def get_output(self) -> pd.DataFrame:
        """
        Retrieves the output of the embedding block.

        Returns
        -------
        pd.DataFrame
            A DataFrame with 'smiles' and 'embedding' columns. Returns None if
            the `run` method has not been completed.
        """
        return self.ligands_embd
    
    
    def run(self, batch_size: int = 32):
        """
        Executes the embedding generation process for the input ligands.

        This method processes the ligands in batches to efficiently
        generate embeddings. It displays progress during the process.

        Parameters
        ----------
        batch_size : int, optional
            The number of SMILES strings to process in a single batch.
            Default is 32.
        
        Raises
        ------
        ValueError
            If the input ligands have not been set via `set_input` before
            calling this method.
        """
        if self.ligands is None:
            raise ValueError("Ligands input not set. Use set_input() before calling run().")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Processing {len(self.ligands)} ligands on {device}")
        
        embedder = ChembertaEmbedder(device=device, model_name=self.model_name)
        
        start_time = time.time()

        # Get embeddings for all ligands, embedder handles batching and progress bar
        embeddings = embedder.embed(self.ligands, batch_size=batch_size)
        
        # Convert to numpy arrays
        embedding_arrays = [emb.cpu().numpy() for emb in embeddings]

        # Create final DataFrame
        self.ligands_embd = pd.DataFrame({
            'smiles': self.ligands,
            'embedding': embedding_arrays
        })
        
        total_time = time.time() - start_time
        print(f"\nCompleted processing {len(self.ligands)} ligands in {total_time:.1f} seconds ({total_time/60:.1f} minutes).")
