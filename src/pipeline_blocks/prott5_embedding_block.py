import torch
import sys
import os

# Add the src directory to PYTHONPATH at runtime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.protT5_embedding import ProtT5Embedder

class Prott5EmbeddingBlock:
    """
    Preprocessing block for binding affinity datasets.
    Downloads raw data, reformats it, and prepares it for embedding.
    """

    def __init__(self):
        self.proteins = None
        self.proteins_embd = None


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
        embedder = ProtT5Embedder(device=device)
        self.proteins_embd = embedder.embed(self.proteins)
        