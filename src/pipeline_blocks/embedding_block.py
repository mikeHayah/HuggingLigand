

import sys
import os

# Add the src directory to PYTHONPATH at runtime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.protT5_embedding import ProtT5Embedder

class EmbeddingBlock:
    """
    Preprocessing block for binding affinity datasets.
    Downloads raw data, reformats it, and prepares it for embedding.
    """

    def __init__(self):
        self.ligands = None
        self.proteins = None
        self.ligands_embd = None
        self.proteins_embd = None


    def set_input(self, ligands, proteins):
        """
        Set the input ligands and proteins.
        """
        self.ligands = ligands
        self.proteins = proteins

    def get_output(self):
        """
        Get the processed ligands and proteins.
        """
        return self.ligands_embd, self.proteins_embd
    
    def run(self):
        """
        Run the embedding step for the provided proteins.
        """
        if self.proteins is None:
            raise ValueError("Proteins input not set. Use set_input() before calling run().")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = ProtT5Embedder(device=device)
        self.proteins_embd = embedder.embed(self.proteins)
        