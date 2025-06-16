import os
import zipfile
import pandas as pd

class Reformatter:
    """
    Handles raw dataset files: detects format, unzips if necessary, 
    extracts relevant columns, and outputs standardized CSV files.
    """
    def __init__(self, input_path, reformated_path, required_columns):
        """
        Parameters:
        ----------
        input_path : str
            Path to the raw data file (could be zipped or plain .tsv).
        reformated_path : str
            Destination path for the cleaned CSV file.
        required_columns : list of str
            List of column names to keep in the reformatted dataset.
        """
        self.input_path = input_path
        self.reformated_path = reformated_path
        self.required_columns = required_columns

    def _unzip_if_needed(self):
        """
        Unzip the input file if it's a ZIP archive.
        """
        if self.input_path.endswith('.zip'):
            with zipfile.ZipFile(self.input_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(self.input_path))
            # Assume there's only one file in the zip for simplicity
            extracted_files = zip_ref.namelist()
            return os.path.join(os.path.dirname(self.input_path), extracted_files[0])
        else:
            return self.input_path

    def _read_file(self, file_path):
        """
        Read a file and return its content as a list of lines.
        Parameters:
        ----------
        file_path : str
            Path to the file to read.
        Returns:
        -------
        df : pandas.DataFrame
            DataFrame containing the file's content.
        """
        sep = '\t' if file_path.endswith('.tsv') else ','
        combined_data = []
        header = None
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line:
                   continue  # skip empty lines
                cells = line.strip()
                # If header is None or this line matches header
                if header is None:
                    header = cells
                    continue
                if cells == header:
                    # repeated header, skip
                    continue
                # Only add rows with correct length
                if len(cells) == len(header):
                    combined_data.append(cells)
        df = pd.DataFrame(combined_data, columns=header)
        return df

    def reformat(self):
        """
        Main method to reformat raw data into a clean CSV with selected columns.
        Returns:
        -------
        str
            Path to the cleaned CSV file.
        """
        file_path = self._unzip_if_needed()
        df = self._read_file(file_path)
        # Check columns and select required ones
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        df = df[self.required_columns]
        df.to_csv(self.reformated_path, index=False)
        print(f"Cleaned dataset saved to {self.reformated_path}")
