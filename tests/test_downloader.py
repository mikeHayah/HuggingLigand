import os
import pytest
from src.modules.downloader import DataDownloader  

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

def test_download_success(requests_mock, temp_dir):
    url = "http://example.com/testfile.txt"
    content = b"Hello, world!"
    
    # Mock HTTP GET request
    requests_mock.get(url, content=content, headers={"content-length": str(len(content))})
    
    downloader = DataDownloader(url, str(temp_dir))
    filepath = downloader.download()
    
    assert os.path.exists(filepath)
    with open(filepath, 'rb') as f:
        assert f.read() == content



def test_skip_existing_file(tmp_path):
    url = "http://example.com/testfile.txt"
    dest_folder = tmp_path
    file_path = os.path.join(dest_folder, "testfile.txt")

    # Create dummy file before test
    with open(file_path, 'w') as f:
        f.write("Existing content")

    downloader = DataDownloader(url, str(dest_folder))
    result_path = downloader.download()

    assert result_path == str(file_path)
    with open(file_path, 'r') as f:
        assert f.read() == "Existing content"
