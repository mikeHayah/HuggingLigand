from transformers import T5Tokenizer, T5EncoderModel
import torch
from typing import List
import logging

logger = logging.getLogger(__name__)


class ProtT5Embedder:
    """
    Generate protein embeddings using the ProtT5-XL-UniRef50 model.

    Parameters
    ----------
    device : str, optional
        Torch device to use ('cuda' or 'cpu'). Default is 'cpu'.

    Attributes
    ----------
    tokenizer : T5Tokenizer
        Tokenizer for the ProtT5 model.
    model : T5EncoderModel
        Encoder model for generating embeddings.
    device : torch.device
        Computation device.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        self.model.to(self.device)
        self.model.eval()
        self._cache = {}

    def embed(self, sequences: List[str]) -> List[torch.Tensor]:
        """
        Embed a list of protein sequences.

        Parameters
        ----------
        sequences : list of str
            Raw amino acid sequences.

        Returns
        -------
        list of torch.Tensor
            Mean-pooled sequence embeddings, one per input.
        """
        formatted_seqs = [" ".join(list(seq.strip())) for seq in sequences]
        inputs = self.tokenizer(formatted_seqs, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

        pooled = []
        for seq, emb, mask in zip(sequences, embeddings, attention_mask):
            if seq in self._cache:
                pooled.append(self._cache[seq])
            else:
                mean_emb = emb[:mask.sum()].mean(dim=0)
                self._cache[seq] = mean_emb
                pooled.append(mean_emb)

        return pooled