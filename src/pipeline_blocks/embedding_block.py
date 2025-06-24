
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
        Run the preprocessing steps.
        This method should be implemented to perform the actual data processing.
        """
        # Placeholder for actual implementation
        # For example, create embedings, etc.
        pass