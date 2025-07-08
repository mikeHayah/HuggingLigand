import pandas as pd

from src.modules.downloader import DataDownloader
from src.modules.reformatter import Reformatter
from src.modules.bindingdata import BindingData

class PreEmbeddingBlock:
    """
    Preprocessing block for binding affinity datasets.
    Downloads raw data, reformats it, and prepares it for embedding.
    """

    def __init__(self, download_url, remove_duplicates=False):
        """
        Parameters
        ----------
        download_url : str
            URL to download the raw binding affinity dataset.
        output_path : str
            Path to save the cleaned dataset.
        remove_duplicates : bool
            If True, removes duplicate ligands and proteins.
        """
        self.download_url = download_url
        self.raw_data_path = "data/raw"
        self.output_path = "data/processed"
        self.ligand = None
        self.protein = None 
        self.remove_duplicates = remove_duplicates
        

    def run(self):
        """
        Execute the preprocessing pipeline: download, reformat, and save the dataset.
        """
        # Download raw dataset
        downloader = DataDownloader(self.download_url, self.raw_data_path)
        downloader.download()
        print(f"Downloaded file: {downloader.filename}")

        # Reformat the dataset
        reformated_csv_path=downloader.filename.replace('.zip', '_cleaned.csv')
        reformatter = Reformatter(
            input_path=downloader.filename,
            reformated_path=reformated_csv_path,
            required_columns=["Ligand SMILES", "BindingDB Target Chain Sequence", "Ki (nM)", "IC50 (nM)", "Kd (nM)"]
            #required_columns=["BindingDB Ligand Name", "Ligand SMILES", "Target Name", "BindingDB Target Chain Sequence", "Ki (nM)", "IC50 (nM)", "Kd (nM)"]
        )
        reformatter.reformat()
        print(f"Reformatted dataset saved to: {reformatter.reformated_path}")

        # Load cleaned data
        my_binding_data = BindingData(pd.read_csv(reformatter.reformated_path))

        # Example pipeline usage
        #lig, pro = my_binding_data.decouple()
        self.ligand, self.protein = my_binding_data.pipeline(['decouple'])

        if self.remove_duplicates:
            self._remove_duplicates()
    
    def _remove_duplicates(self):
        """
        Remove duplicate ligands and proteins from the datasets.
        """
        import logging
        
        # Store original counts
        original_ligand_count = len(self.ligand) if self.ligand is not None else 0
        original_protein_count = len(self.protein) if self.protein is not None else 0
        
        if self.ligand is not None and not self.ligand.empty:
            # Remove duplicate ligands based on SMILES
            ligand_column = 'Ligand SMILES'
            if ligand_column in self.ligand.columns:
                # Keep first occurrence of each unique SMILES
                self.ligand = self.ligand.drop_duplicates(
                    subset=[ligand_column], 
                    keep='first'
                ).reset_index(drop=True)
                
                ligand_duplicates_removed = original_ligand_count - len(self.ligand)
                logging.info(f"Removed {ligand_duplicates_removed} duplicate ligands. "
                           f"Remaining: {len(self.ligand)}")
        
        if self.protein is not None and not self.protein.empty:
            # Remove duplicate proteins based on sequence
            protein_column = 'BindingDB Target Chain Sequence'
            if protein_column in self.protein.columns:
                # Keep first occurrence of each unique sequence
                self.protein = self.protein.drop_duplicates(
                    subset=[protein_column], 
                    keep='first'
                ).reset_index(drop=True)
                
                protein_duplicates_removed = original_protein_count - len(self.protein)
                logging.info(f"Removed {protein_duplicates_removed} duplicate proteins. "
                           f"Remaining: {len(self.protein)}")

    def get_output(self):
       
        return self.ligand, self.protein

