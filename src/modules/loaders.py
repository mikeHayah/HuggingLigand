import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, T5EncoderModel, T5Tokenizer


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


def load_prott5_model(device: torch.device):
    """
    Load the ProtT5 tokenizer and encoder model.

    Parameters
    ----------
    device : torch.device
        The computation device to move the model to.

    Returns
    -------
    tuple
        (tokenizer, model) ready for inference.
    """
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model.to(device)
    model.eval()
    return tokenizer, model
