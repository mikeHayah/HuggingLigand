import pandas as pd
import torch

from src.models.protT5_embedding import ProtT5Embedder

class Prott5EmbeddingBlock:
    """
    Preprocessing block for binding affinity datasets.
    Downloads raw data, reformats it, and prepares it for embedding.
    """

    def __init__(self, model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc"):
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
        embedder = ProtT5Embedder(device=device, model_name=self.model_name)
        embeddings = embedder.embed(self.proteins, show_progress=True)
        self.proteins_embd = pd.DataFrame({'smiles': self.proteins, 'embedding': embeddings})