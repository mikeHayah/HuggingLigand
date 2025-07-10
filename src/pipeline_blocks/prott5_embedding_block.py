import pandas as pd
import torch
import time
import logging

from src.models.protT5_embedding import ProtT5Embedder

class Prott5EmbeddingBlock:
    """
    A pipeline block to generate embeddings for proteins using the ProtT5 model.
    
    This block takes a list of protein sequences, processes them to generate
    embeddings, and returns a pandas DataFrame containing the original sequences
    and their corresponding embeddings.

    Attributes
    ----------
    proteins : list of str
        The input list of protein sequences to be processed.
    proteins_embd : pd.DataFrame
        A DataFrame containing the protein sequences and their embeddings after
        the `run` method has been executed.
    model_name : str
        The name of the pre-trained ProtT5 model to use.
    """

    def __init__(self, model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc"):
        """
        Initialize the Prott5EmbeddingBlock.
        
        Parameters
        ----------
        model_name : str
            The name of the pre-trained ProtT5 model to use.
            Default is configured in config.ini under [models].protein_model.
        """
        self.proteins = None
        self.proteins_embd = None
        self.model_name = model_name


    def set_input(self, proteins):
        """
        Set the input proteins.
        """
        self.proteins = proteins


    def get_output(self):
        """
        Get the processed proteins.
        """
        return self.proteins_embd
    
    
    def run(self):
        """
        Run the embedding step for the provided proteins.
        """
        if self.proteins is None:
            raise ValueError("Proteins input not set. Use set_input() before calling run().")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info(f"Processing {len(self.proteins)} proteins on {device}")
        
        embedder = ProtT5Embedder(device=device, model_name=self.model_name)

        start_time = time.time()

        embeddings = embedder.embed(self.proteins, show_progress=True)
        self.proteins_embd = pd.DataFrame({'smiles': self.proteins, 'embedding': embeddings})

        total_time = time.time() - start_time
        logging.info(f"\nCompleted processing {len(self.proteins)} proteins in {total_time:.1f} seconds ({total_time/60:.1f} minutes).")