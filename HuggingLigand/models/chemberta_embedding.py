from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch


def get_chemberta_embeddings(smiles_list):
    """
    Get embeddings for a list of SMILES strings using the ChemBERTa model for ligands.

    Args:
        smiles_list (list of str): List of SMILES strings representing ligands.

    Returns:
        torch.Tensor: Embeddings for the ligands, shape (batch_size, embedding_dim).
    """

    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = AutoModelForMaskedLM.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

    # Tokenize the list of ligand strings
    # Use padding and truncation to handle varying lengths
    inputs = tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract the last hidden state (embeddings) from the model output
    # TODO: Check if we need to use a different layer or pooling strategy
    embeddings = outputs.hidden_states[-1] 

    return embeddings


# Test code
if __name__ == "__main__":

    embedding = get_chemberta_embeddings("CCO")
    print(embedding)
    print(embedding.shape)
    embedding = get_chemberta_embeddings(["CCO", "CCN(CC)CC", "C1=CC=CC=C1"])
    print(embedding)
    print(embedding.shape)