import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def load_chemberta_model(device: torch.device):
    """
    Load the ChemBERTa tokenizer and model.

    Parameters
    ----------
    model_name : str
        The name of the pre-trained ChemBERTa model to load.
    device : torch.device
        The computation device to move the model to.

    Returns
    -------
    tuple
        (tokenizer, model) ready for inference.
    """
    model_name = "seyonec/ChemBERTa-zinc-base-v1"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model
