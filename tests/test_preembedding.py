import pytest
from unittest.mock import patch
from src.pipeline_blocks.preembedding_block import PreEmbeddingBlock

@pytest.fixture
def dummy_url():
    return "https://example.com/fake_bindingdb.zip"

def test_preembeddingblock_initialization(dummy_url):
    block = PreEmbeddingBlock(dummy_url)
    assert block.download_url == dummy_url
    assert block.raw_data_path == "data/raw"
    assert block.output_path == "data/processed"
    assert block.ligand is None
    assert block.protein is None

@patch("src.pipeline_blocks.preembedding_block.pd.read_csv")
@patch("src.pipeline_blocks.preembedding_block.BindingData")
@patch("src.pipeline_blocks.preembedding_block.Reformatter")
@patch("src.pipeline_blocks.preembedding_block.DataDownloader")
def test_run_pipeline(mock_downloader, mock_reformatter, mock_bindingdata, mock_read_csv, dummy_url):
    # Mock DataDownloader behavior
    mock_downloader_instance = mock_downloader.return_value
    mock_downloader_instance.download.return_value = None
    mock_downloader_instance.filename = "data/raw/fake_bindingdb.zip"

    # Mock Reformatter behavior
    mock_reformatter_instance = mock_reformatter.return_value
    mock_reformatter_instance.reformat.return_value = None
    mock_reformatter_instance.reformated_path = "data/processed/fake_bindingdb_cleaned.csv"

    # Mock pandas.read_csv
    mock_read_csv.return_value = "mock_dataframe"

    # Mock BindingData behavior
    mock_bindingdata_instance = mock_bindingdata.return_value
    mock_bindingdata_instance.pipeline.return_value = ("mock_ligand_df", "mock_protein_df")

    # Initialize PreEmbeddingBlock
    block = PreEmbeddingBlock(dummy_url)

    # Run the pipeline
    block.run()

    # Assertions
    mock_downloader.assert_called_once_with(dummy_url, "data/raw")
    mock_downloader_instance.download.assert_called_once()

    mock_reformatter.assert_called_once_with(
        input_path=mock_downloader_instance.filename,
        reformated_path=mock_downloader_instance.filename.replace(".zip", "_cleaned.csv"),
        required_columns=["Ligand SMILES", "BindingDB Target Chain Sequence", "Ki (nM)", "IC50 (nM)", "Kd (nM)"]
    )
    mock_reformatter_instance.reformat.assert_called_once()

    mock_read_csv.assert_called_once_with(mock_reformatter_instance.reformated_path)

    mock_bindingdata.assert_called_once_with("mock_dataframe")
    mock_bindingdata_instance.pipeline.assert_called_once_with(["decouple"])

    assert block.ligand == "mock_ligand_df"
    assert block.protein == "mock_protein_df"


# Test the output retrieval
def test_get_output():
    block = PreEmbeddingBlock("https://example.com/fake_bindingdb.zip")
    block.ligand = "mock_ligand"
    block.protein = "mock_protein"
    ligand, protein = block.get_output()
    assert ligand == "mock_ligand"
    assert protein == "mock_protein"
    

    # Mock
