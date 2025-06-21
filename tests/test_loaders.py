import pytest
from transformers import T5EncoderModel, T5Tokenizer

from modules.loaders import load_prott5_model


@pytest.mark.slow
def test_load_prott5_model():
    tokenizer, model = load_prott5_model(device="cpu")

    assert isinstance(tokenizer, T5Tokenizer), "Expected a T5Tokenizer"
    assert isinstance(model, T5EncoderModel), "Expected a T5EncoderModel"
    assert model.device.type == "cpu", "Model should be on CPU"

    print("Model and tokenizer loaded successfully")