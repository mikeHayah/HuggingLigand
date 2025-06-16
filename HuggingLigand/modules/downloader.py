import requests
import os
from tqdm import tqdm

class DataDownloader:
    """
    Download files from a given URL and save them to a specified folder.
    """
    def __init__(self, url, dest_folder):
        """
        Parameters
        ----------
        url : str
            The URL of the file to download.
        dest_folder : str
            The folder where the downloaded file will be saved.
        """
        self.url = url
        self.dest_folder = dest_folder
        os.makedirs(self.dest_folder, exist_ok=True)
        self.filename = os.path.join(self.dest_folder, self.url.split("/")[-1])

    def download(self):
        """
        Download the file from the URL and save it to the destination folder.
        Returns
        -------
        str
            The path to the downloaded file.
        """
        response = requests.get(self.url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=self.filename)
        
        with open(self.filename, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)

        progress_bar.close()

        if total_size != 0 and progress_bar.n != total_size:
            print("Download may be incomplete!")

