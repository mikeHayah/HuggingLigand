import requests
import os
import time
import logging
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
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 10
        os.makedirs(self.dest_folder, exist_ok=True)
        self.filename = os.path.join(self.dest_folder, self.url.split("/")[-1])

    def download(self):
        """
        Download the file from the URL and save it to the destination folder.
        Implements retry mechanism with exponential backoff for network failures.
        
        Returns
        -------
        str or None
            The path to the downloaded file if successful, None if failed after all retries.
        """
        if os.path.exists(self.filename):
            print(f"File {self.filename} already exists. Skipping download.")
            return self.filename
        
        for attempt in range(self.max_retries):
            try:
                logging.info(f"Starting download attempt {attempt + 1} of {self.max_retries}")
                return self._download_with_progress()
            except (requests.exceptions.RequestException, IOError) as e:
                if attempt == self.max_retries:
                    logging.error(f"Download failed after {self.max_retries + 1} attempts: {e}")
                    # Clean up partial file if it exists
                    if os.path.exists(self.filename):
                        os.remove(self.filename)
                    raise e
                else:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logging.warning(f"Download attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    # Clean up partial file before retry
                    if os.path.exists(self.filename):
                        os.remove(self.filename)
        
        return None

    def _download_with_progress(self):
        """
        Internal method to perform the actual download with progress bar.
        
        Returns
        -------
        str
            The path to the downloaded file.
        """
        # Use a longer timeout for large files - connection timeout is separate from read timeout
        response = requests.get(self.url, stream=True, timeout=(self.timeout, self.timeout * 3))
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(self.filename))
        
        try:
            with open(self.filename, 'wb') as f:
                for data in response.iter_content(block_size):
                    if data:  # Filter out keep-alive chunks
                        progress_bar.update(len(data))
                        f.write(data)
        finally:
            progress_bar.close()

        # Verify download completeness
        if total_size != 0 and progress_bar.n != total_size:
            raise IOError(f"Download incomplete: expected {total_size} bytes, got {progress_bar.n} bytes")
        
        logging.info(f"Successfully downloaded {self.filename}")
        return self.filename

