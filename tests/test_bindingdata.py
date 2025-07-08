import pytest
import pandas as pd
import os
from src.modules.bindingdata import BindingData

@pytest.fixture
def sample_data():
    data = {
        "Ligand SMILES": ["C1=CC=CC=C1", "C2=CC=CN=C2"],
        "BindingDB Target Chain Sequence": ["MKVSA...", "GYHPA..."],
        "affinity": [7.5, -1.2]
    }
    return BindingData(data)

def test_get_ligands(sample_data):
    ligands = sample_data.get_ligands()
    assert isinstance(ligands, pd.DataFrame)
    assert list(ligands.columns) == ["Ligand SMILES"]
    assert len(ligands) == 2

def test_get_proteins(sample_data):
    proteins = sample_data.get_proteins()
    assert isinstance(proteins, pd.DataFrame)
    assert list(proteins.columns) == ["BindingDB Target Chain Sequence"]
    assert len(proteins) == 2

def test_decouple(sample_data):
    ligands, proteins = sample_data.decouple()
    assert isinstance(ligands, pd.DataFrame)
    assert isinstance(proteins, pd.DataFrame)
    assert list(ligands.columns) == ["Ligand SMILES"]
    assert list(proteins.columns) == ["BindingDB Target Chain Sequence"]

def test_couple(sample_data):
    ligands, proteins = sample_data.decouple()
    combined = sample_data.couple(ligands, proteins)
    assert isinstance(combined, BindingData)
    assert "Ligand SMILES" in combined.columns
    assert "BindingDB Target Chain Sequence" in combined.columns

def test_clean_affinity(sample_data):
    cleaned = sample_data.clean_affinity(min_value=0)
    assert all(cleaned["affinity"] >= 0)
    assert len(cleaned) == 1

def test_apply_valid(sample_data):
    ligands = sample_data.apply("ligands")
    assert isinstance(ligands, pd.DataFrame)
    assert list(ligands.columns) == ["Ligand SMILES"]

def test_apply_invalid_name(sample_data):
    with pytest.raises(ValueError):
        sample_data.apply("nonexistent_operation")

def test_apply_invalid_type(sample_data):
    with pytest.raises(TypeError):
        sample_data.apply(123)

def test_pipeline(sample_data):
    result = sample_data.pipeline(["ligands"])
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["Ligand SMILES"]

def test_to_csv_file(tmp_path, sample_data):
    file_path = tmp_path / "output.csv"
    sample_data.to_csv_file(file_path)
    assert os.path.exists(file_path)
    df = pd.read_csv(file_path)
    assert list(df.columns) == list(sample_data.columns)

def test_to_pickle_file(tmp_path, sample_data):
    file_path = tmp_path / "output.pkl"
    sample_data.to_pickle_file(file_path)
    assert os.path.exists(file_path)
    df = pd.read_pickle(file_path)
    assert list(df.columns) == list(sample_data.columns)

def test_to_jsonl_file(tmp_path, sample_data):
    file_path = tmp_path / "output.jsonl"
    sample_data.to_jsonl_file(file_path)
    assert os.path.exists(file_path)
    with open(file_path) as f:
        lines = f.readlines()
    assert len(lines) == len(sample_data)
