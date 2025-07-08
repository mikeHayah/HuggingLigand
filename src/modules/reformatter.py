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
        Parameters
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
        Parameters
        ----------
        file_path : str
            Path to the file to read.
        Returns
        -------
        df : pandas.DataFrame
            DataFrame containing the file's content.
        """
        sep = '\t' if file_path.endswith('.tsv') else ','
        # Initialize an list to store requied cols as header
        combined_data = []
        header = None
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                cells = line.split(sep)
                if cells[0] == 'BindingDB Reactant_set_id':
                    if(self.read_valid_header(cells)):
                        header = cells
                    else:
                        header = None
                else:
                    if header is None:
                        continue
                    else:
                        # look for values of cells where cols of header is self.required_columns
                        required_cells = [cells[header.index(col)] if col in header else '' for col in self.required_columns]
                        combined_data.append(required_cells)
        df = pd.DataFrame(combined_data, columns= [cols for cols in self.required_columns])
        return df

    def read_valid_header(self,cells):
        """Check if the header contains the required columns.

        Parameters
        ----------
        cells : list
            List of column names from the header.

        Returns
        -------
        bool
            True if the header contains the required columns, False otherwise.
        """
        # check if the header contain the required columns
        if any(col in cells for col in self.required_columns):
            return True
        else:
            return False

    def reformat(self):
        """Main method to reformat raw data into a clean CSV with selected columns.

        Returns
        -------
        str
            Path to the cleaned CSV file.
        """
        file_path = self._unzip_if_needed()
        df = self._read_file(file_path)
        df.to_csv(self.reformated_path, index=False)
