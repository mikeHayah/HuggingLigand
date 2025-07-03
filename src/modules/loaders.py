import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, T5EncoderModel, T5Tokenizer


def load_chemberta_model(device: torch.device, model_name: str = "seyonec/ChemBERTa-zinc-base-v1"):
    """
    Load the ChemBERTa tokenizer and model.

    Parameters
    ----------
    device : torch.device
        The computation device to move the model to.
    model_name : str
        The name of the pre-trained ChemBERTa model to load.
        Default is configured in config.ini under [models].ligand_model.

    Returns
    -------
    tuple
        (tokenizer, model) ready for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def load_prott5_model(device: torch.device, model_name: str = "Rostlab/prot_t5_xl_half_uniref50-enc"):
    """
    Load the ProtT5 tokenizer and encoder model.

    Parameters
    ----------
    device : torch.device
        The computation device to move the model to.
    model_name : str
        The name of the pre-trained ProtT5 model to load.
        Default is configured in config.ini under [models].protein_model.

    Returns
    -------
    tuple
        (tokenizer, model) ready for inference.
    """

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model
