import pandas as pd

class BindingData(pd.DataFrame):
    """
    A DataFrame wrapper with extended functionality for binding affinity datasets.
    """

    _metadata = ['source']

    def __init__(self, *args, **kwargs):
        self.source = kwargs.pop('source', None)
        super().__init__(*args, **kwargs)

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

    def encode_smiles(self, encoder_func):
        """
        Apply a function to encode SMILES strings (for example tokenization).
        """
        if 'smiles' not in self.columns:
            raise ValueError("SMILES column not found.")
        self['encoded_smiles'] = self['smiles'].apply(encoder_func)
        return self

    def pipeline(self, ops):
        """
        Apply a sequence of operations by name.
        """
        for op in ops:
            method = getattr(self, op, None)
            if callable(method):
                method()
            else:
                raise ValueError(f"No method named {op} found.")
        return self
