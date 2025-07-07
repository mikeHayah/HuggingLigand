import os
import zipfile
import pandas as pd
import pytest
from src.modules.reformatter import Reformatter


@pytest.fixture
def sample_tsv_file(tmp_path):
    file_path = tmp_path / "sample.tsv"
    content = (
        "BindingDB Reactant_set_id\tColA\tColB\n"
        "123\tValueA1\tValueB1\n"
        "124\tValueA2\tValueB2\n"
    )
    file_path.write_text(content)
    return file_path


def test_read_valid_header_positive():
    ref = Reformatter("input.tsv", "output.csv", ["ColA"])
    header = ["BindingDB Reactant_set_id", "ColA", "ColB"]
    assert ref.read_valid_header(header) is True


def test_read_valid_header_negative():
    ref = Reformatter("input.tsv", "output.csv", ["NonExistingCol"])
    header = ["BindingDB Reactant_set_id", "ColA", "ColB"]
    assert ref.read_valid_header(header) is False


def test_read_file_reads_correct_columns(sample_tsv_file, tmp_path):
    output_path = tmp_path / "cleaned.csv"
    required_columns = ["ColA", "ColB"]
    ref = Reformatter(str(sample_tsv_file), str(output_path), required_columns)

    df = ref._read_file(str(sample_tsv_file))

    assert list(df.columns) == required_columns
    assert len(df) == 2
    assert df.iloc[0]["ColA"] == "ValueA1"
    assert df.iloc[1]["ColB"] == "ValueB2"


def test_reformat_creates_clean_csv(sample_tsv_file, tmp_path):
    output_path = tmp_path / "result.csv"
    required_columns = ["ColA"]
    ref = Reformatter(str(sample_tsv_file), str(output_path), required_columns)

    ref.reformat()

    df = pd.read_csv(output_path)
    assert "ColA" in df.columns
    assert len(df) == 2
    assert df.iloc[0]["ColA"] == "ValueA1"


def test_unzip_if_needed(tmp_path):
    # Create dummy file to zip
    tsv_path = tmp_path / "data.tsv"
    tsv_path.write_text("BindingDB Reactant_set_id\tColX\n123\tA")

    # Zip it
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(tsv_path, arcname="data.tsv")

    ref = Reformatter(str(zip_path), "output.csv", ["ColX"])
    extracted_file = ref._unzip_if_needed()

    assert os.path.exists(extracted_file)
    assert extracted_file.endswith("data.tsv")
