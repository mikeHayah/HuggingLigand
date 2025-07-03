import pandas as pd

class BindingData(pd.DataFrame):
    """
    A DataFrame wrapper with extended functionality for binding affinity datasets.
    """

    _metadata = ['operations']
    operations = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Map available functionality to internal methods
        self.operations = {
            'ligands': self.get_ligands,
            'proteins': self.get_proteins,
            'decouple': self.decouple,
            'couple': self.couple, 
            'csv': self.to_csv_file,
            'pickle': self.to_pickle_file,
            'jsonl': self.to_jsonl_file,
        }

    @property
    def _constructor(self):
        return BindingData

    def to_csv_file(self, path):
        self.to_csv(path, index=False)
        print(f"Data saved as CSV at {path}")

    def to_pickle_file(self, path):
        self.to_pickle(path)
        print(f"Data saved as pickle at {path}")

    def to_jsonl_file(self, path):
        self.to_json(path, orient="records", lines=True)
        print(f"Data saved as JSONL at {path}")

    def clean_affinity(self, min_value=0):
        """
        Example processing: remove invalid or negative affinity values.
        """
        if 'affinity' not in self.columns:
            raise ValueError("Affinity column not found in dataset.")
        self.dropna(subset=['affinity'], inplace=True)
        self = self[self['affinity'] >= min_value]
        return self

    def get_ligands(self):
        """
        Extract ligands from the dataset.
        Returns
        -------
        pd.Series
            Series of unique ligands.
        """
        if 'Ligand SMILES' not in self.columns:
            raise ValueError("Ligand SMILES column not found in dataset.")
        return self[['Ligand SMILES']]
    
    def get_proteins(self):
        """
        Extract proteins from the dataset.
        Returns
        -------
        pd.Series
            Series of unique proteins.
        """
        if 'BindingDB Target Chain Sequence' not in self.columns:
            raise ValueError("BindingDB Target Chain Sequence column not found in dataset.")
        return self[['BindingDB Target Chain Sequence']]
    
    def decouple(self):
        """
        Split the DataFrame into two separate DataFrames: one for ligands and one for proteins.
        Returns
        -------
        tuple
            A tuple containing two DataFrames: (ligands_df, proteins_df).
        """
        ligands_df = self[['Ligand SMILES']]
        proteins_df = self[['BindingDB Target Chain Sequence']]
        return ligands_df, proteins_df
    
    def couple(self, ligands_df, proteins_df):
        """
        Combine two DataFrames: one for ligands and one for proteins, into a single DataFrame.
        Parameters
        ----------
        ligands_df : pd.DataFrame
            DataFrame containing ligands.
        proteins_df : pd.DataFrame
            DataFrame containing proteins.
        Returns
        -------
        BindingData
            Combined DataFrame with ligands and proteins.
        """
        combined_df = pd.merge(ligands_df, proteins_df, left_index=True, right_index=True)
        return BindingData(combined_df)

    def apply(self, operation, *args, **kwargs):
        """Apply a specified operation to the DataFrame.

        Parameters
        ----------
        operation : str
            The name of the operation to apply.
        *args, **kwargs : additional arguments
            Additional arguments to pass to the operation.
        """
        if isinstance(operation, str):
            if operation in self.operations:
                return self.operations[operation](*args, **kwargs)

            else:
                raise ValueError(f"No operation named '{operation}'")
        else:
            raise TypeError("Operation must be a string referring to a cleaning function")

    def pipeline(self, ops):
        """Apply a series of operations to the DataFrame.

        Parameters
        ----------
        ops : list of str
            A list of operations to apply.
        """
        result = self
        for op in ops:
            result = result.apply(op)
            # If the operation returns something other than a DataFrame, stop and return it
            if not isinstance(result, BindingData):
                return result
        return result
