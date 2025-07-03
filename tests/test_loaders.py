
import pytest
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, T5EncoderModel, T5Tokenizer

from src.modules.loaders import load_chemberta_model, load_prott5_model


@pytest.mark.slow
def test_load_prott5_model():
    tokenizer, model = load_prott5_model(device="cpu")

    assert isinstance(tokenizer, T5Tokenizer), "Expected a T5Tokenizer"
    assert isinstance(model, T5EncoderModel), "Expected a T5EncoderModel"
    assert model.device.type == "cpu", "Model should be on CPU"

    print("Model and tokenizer loaded successfully")


@pytest.mark.slow
def test_load_chemberta_model():
    tokenizer, model = load_chemberta_model(device="cpu")

    assert isinstance(tokenizer, RobertaTokenizerFast), "Expected a AutoTokenizer"
    assert isinstance(model, RobertaForMaskedLM), "Expected a AutoModelForMaskedLM"
    assert model.device.type == "cpu", "Model should be on CPU"

    print("Model and tokenizer loaded successfully")
