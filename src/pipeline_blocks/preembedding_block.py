from modules.downloader import DataDownloader
from modules.reformatter import Reformatter
from modules.bindingdata import BindingData
import pandas as pd

class PreEmbeddingBlock:
    """
    Preprocessing block for binding affinity datasets.
    Downloads raw data, reformats it, and prepares it for embedding.
    """

    def __init__(self, download_url):
        """
        Parameters
        ----------
        download_url : str
            URL to download the raw binding affinity dataset.
        output_path : str
            Path to save the cleaned dataset.
        """
        self.download_url = download_url
        self.raw_data_path = "data/raw"
        self.output_path = "data/processed"

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
        #binding_data = BindingData(pd.read_csv(reformatter.reformated_path))

        # Example pipeline usage
        #binding_data.clean_affinity(min_value=0.0)
        

        # Save as pickle
        #binding_data.to_pickle_file("data/binding_data.pkl")
        #binding_data.to_csv_file("data/binding_data_cleaned.csv")
