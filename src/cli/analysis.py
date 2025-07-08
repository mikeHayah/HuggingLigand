import os
import logging
import click
import numpy as np


def run_ligand_similarity_analysis(ligands_df, ligands_embd_df, text_only, output_dir):
    """
    Performs and visualizes a comparative analysis of chemical similarity
    versus embedding similarity for ligands.

    This function calculates two types of similarities for a sample of ligands:
    1.  **Tanimoto Similarity:** Based on Morgan fingerprints of the SMILES strings,
        representing chemical structural similarity.
    2.  **Cosine Similarity:** Based on the generated high-dimensional embeddings.

    It then plots these two similarity measures against each other to visualize
    their correlation and prints the Pearson correlation coefficient. A high
    correlation suggests that the embedding space preserves the chemical
    similarity space.
    """
    # --- 1. Dependency Check ---
    # Ensure all required libraries for this analysis are installed.
    try:
        from rdkit import Chem
        from rdkit.DataStructs import BulkTanimotoSimilarity
        from rdkit.Chem import AllChem
        from rdkit import rdBase
        import matplotlib
        if not text_only:
            matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        logging.warning(f"Skipping similarity analysis due to missing packages: {e}")
        click.echo("WARNING: Please install 'rdkit-pypi', 'scikit-learn', 'matplotlib', and 'seaborn' to run similarity analysis.")
        return

    # Suppress RDKit warnings to avoid cluttering the output
    rdBase.DisableLog('rdApp.warning')

    logging.info("Starting ligand similarity analysis...")

    # --- 2. Data Sampling ---
    # To avoid long computation times, we run the analysis on a random sample of the data.
    sample_size = min(100, len(ligands_embd_df))
    if sample_size < 2:
        logging.warning("Not enough data to run ligand similarity analysis.")
        rdBase.EnableLog('rdApp.warning')
        return
        
    sample_indices = np.random.choice(ligands_embd_df.index, sample_size, replace=False)
    
    sampled_ligands = ligands_df.loc[sample_indices]
    sampled_embeddings = ligands_embd_df.loc[sample_indices]

    # --- 3. Tanimoto Similarity (Chemical Structure) ---
    # Convert SMILES strings to RDKit molecule objects.
    smiles_list = sampled_ligands['Ligand SMILES'].tolist()
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    
    # Filter out any SMILES that could not be parsed.
    valid_mols_indices = [i for i, m in enumerate(mols) if m is not None]
    if len(valid_mols_indices) < sample_size:
        logging.warning(f"Could not parse {sample_size - len(valid_mols_indices)} SMILES strings.")
        mols = [mols[i] for i in valid_mols_indices]
        sampled_embeddings = sampled_embeddings.iloc[valid_mols_indices]
        sample_size = len(mols)
        if sample_size < 2:
            logging.warning("Not enough valid SMILES to run similarity analysis.")
            rdBase.EnableLog('rdApp.warning')
            return

    # Generate Morgan fingerprints for each molecule.
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]
    
    # Calculate the pairwise Tanimoto similarity matrix.
    tanimoto_sims_matrix = np.zeros((sample_size, sample_size))
    for i in range(sample_size):
        tanimoto_sims_matrix[i, :] = BulkTanimotoSimilarity(fps[i], fps)
    
    # Extract the upper triangle of the matrix to get a 1D array of similarity scores.
    tanimoto_sims = tanimoto_sims_matrix[np.triu_indices(sample_size, k=1)]

    # --- 4. Cosine Similarity (Embeddings) ---
    # Calculate the pairwise cosine similarity for the ligand embeddings.
    embeddings = np.vstack(sampled_embeddings['embedding'].values)
    cosine_sims_matrix = cosine_similarity(embeddings)
    # Extract the upper triangle for comparison.
    cosine_sims = cosine_sims_matrix[np.triu_indices(sample_size, k=1)]

    # --- 5. Comparison and Visualization ---
    # Create a scatter plot to compare the two similarity measures.
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=tanimoto_sims, y=cosine_sims, alpha=0.5)
    plt.title('Ligand Similarity: Tanimoto (Chemical) vs. Cosine (Embedding)')
    plt.xlabel('Tanimoto Similarity (Chemical Structure)')
    plt.ylabel('Cosine Similarity (Embeddings)')
    
    # Save the plot to a file.
    plot_path = os.path.join(output_dir, 'ligand_similarity_comparison.png')
    plt.savefig(plot_path)
    logging.info(f"Saved ligand similarity plot to {plot_path}")
    # Show the plot interactively if not in text-only mode.
    if not text_only:
        plt.show()
    plt.close()

    # --- 6. Report Correlation ---
    # Calculate and print the Pearson correlation coefficient.
    correlation = np.corrcoef(tanimoto_sims, cosine_sims)[0, 1]
    click.echo("\n--- Ligand Similarity Analysis ---")
    click.echo(f"Correlation between Tanimoto and Cosine similarity: {correlation:.4f}")
    click.echo("A high correlation suggests the embedding space preserves chemical similarity.")
    click.echo(f"Plot saved to: {plot_path}")
    click.echo("----------------------------------\n")
    # Re-enable RDKit warnings
    rdBase.EnableLog('rdApp.warning')


def run_protein_similarity_analysis(proteins_df, proteins_embd_df, text_only, output_dir):
    """
    Performs and visualizes a comparative analysis of protein sequence similarity
    versus embedding similarity.

    This function calculates two types of similarities for a sample of proteins:
    1.  **Sequence Similarity:** Based on the normalized global alignment score
        (Needleman-Wunsch) of the protein sequences.
    2.  **Cosine Similarity:** Based on the generated high-dimensional embeddings.

    It then plots these two similarity measures against each other to visualize
    their correlation and prints the Pearson correlation coefficient.
    """
    # --- 1. Dependency Check ---
    try:
        from Bio.Align import PairwiseAligner
        import matplotlib
        if not text_only:
            matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        logging.warning(f"Skipping protein similarity analysis due to missing packages: {e}")
        click.echo("WARNING: Please install 'biopython', 'scikit-learn', 'matplotlib', and 'seaborn' to run protein similarity analysis.")
        return

    logging.info("Starting protein similarity analysis...")

    # --- 2. Data Sampling ---
    sample_size = min(100, len(proteins_embd_df))
    if sample_size < 2:
        logging.warning("Not enough data to run protein similarity analysis.")
        return
        
    sample_indices = np.random.choice(proteins_embd_df.index, sample_size, replace=False)
    
    sampled_proteins = proteins_df.loc[sample_indices]
    sampled_embeddings = proteins_embd_df.loc[sample_indices]

    # --- 3. Sequence Similarity (Alignment Score) ---
    sequences = sampled_proteins['BindingDB Target Chain Sequence'].tolist()
    
    seq_sims_matrix = np.zeros((sample_size, sample_size))
    aligner = PairwiseAligner()
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = 0.0
    aligner.extend_gap_score = 0.0
    
    for i in range(sample_size):
        for j in range(i, sample_size):
            seq1 = sequences[i]
            seq2 = sequences[j]
            # Ensure sequences are valid strings before alignment.
            if not isinstance(seq1, str) or not isinstance(seq2, str) or len(seq1) == 0 or len(seq2) == 0:
                score = 0
            else:
                # Calculate global alignment score and normalize by the length of the shorter sequence.
                score = aligner.score(seq1, seq2)
                if min(len(seq1), len(seq2)) > 0:
                    score /= min(len(seq1), len(seq2))
                else:
                    score = 0
            seq_sims_matrix[i, j] = seq_sims_matrix[j, i] = score

    # Extract the upper triangle of the matrix.
    seq_sims = seq_sims_matrix[np.triu_indices(sample_size, k=1)]

    # --- 4. Cosine Similarity (Embeddings) ---
    embeddings = np.vstack(sampled_embeddings['embedding'].values)
    cosine_sims_matrix = cosine_similarity(embeddings)
    cosine_sims = cosine_sims_matrix[np.triu_indices(sample_size, k=1)]

    # --- 5. Comparison and Visualization ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=seq_sims, y=cosine_sims, alpha=0.5)
    plt.title('Protein Similarity: Sequence Alignment vs. Cosine (Embedding)')
    plt.xlabel('Sequence Similarity (Normalized Alignment Score)')
    plt.ylabel('Cosine Similarity (Embeddings)')
    
    plot_path = os.path.join(output_dir, 'protein_similarity_comparison.png')
    plt.savefig(plot_path)
    logging.info(f"Saved protein similarity plot to {plot_path}")
    if not text_only:
        plt.show()
    plt.close()

    # --- 6. Report Correlation ---
    correlation = np.corrcoef(seq_sims, cosine_sims)[0, 1]
    click.echo("\n--- Protein Similarity Analysis ---")
    click.echo(f"Correlation between Sequence and Cosine similarity: {correlation:.4f}")
    click.echo("A high correlation suggests the embedding space preserves sequence similarity.")
    click.echo(f"Plot saved to: {plot_path}")
    click.echo("-----------------------------------\n")
