import os
import torch
import multiprocessing as mp
import click
import logging

from src.pipeline_blocks.preembedding_block import PreEmbeddingBlock
from src.pipeline_blocks.prott5_embedding_block import Prott5EmbeddingBlock
from src.pipeline_blocks.chemberta_embedding_block import ChembertaEmbeddingBlock
from src.scripts.analysis import run_similarity_analysis, run_protein_similarity_analysis

@click.command()
@click.option('--source', default='https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202506_tsv.zip', help='URL to the dataset.')
@click.option('-v', '--verbose', count=True, help='Enable verbose output. -v for WARNING, -vv for INFO, -vvv for DEBUG.')
@click.option('--text-only', is_flag=True, help='Suppress graphical output.')
@click.option('--rows', type=int, default=None, help='Number of rows to process from the dataset.')
@click.option('--output-dir', default=None, help='Directory to save the embeddings and analysis results.')
@click.option('--embed', type=click.Choice(['ligand', 'protein', 'both']), default='both', help='Specify what to embed.')
def main(source, verbose, text_only, rows, output_dir, embed):
    """
    This script generates embeddings from a given dataset URL or local path.
    """
    # Set up logging
    log_level = logging.ERROR
    if verbose == 1:
        log_level = logging.WARNING
    elif verbose == 2:
        log_level = logging.INFO
    elif verbose >= 3:
        log_level = logging.DEBUG
    
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Optimize CPU settings before importing other modules
    cpu_cores = mp.cpu_count()
    torch.set_num_threads(cpu_cores)
    torch.set_num_interop_threads(1)
    os.environ['OMP_NUM_THREADS'] = str(cpu_cores)
    os.environ['MKL_NUM_THREADS'] = str(cpu_cores)

    logging.info(f"CPU optimization: Using {cpu_cores} threads")

    preembedding_block = PreEmbeddingBlock(source)
    output_file = "ligands_embeddings"
    preembedding_block.run()
    myligands, myproteins = preembedding_block.get_output()

    if rows:
        myligands = myligands.head(rows)
        myproteins = myproteins.head(rows)
        logging.info(f"Processing the first {rows} rows of the dataset.")

    myligands_embd = None
    if embed in ['ligand', 'both']:
        chemberta_embedding_block = ChembertaEmbeddingBlock()
        ligand_list = myligands['Ligand SMILES'].tolist()
        logging.info(f"Total number of ligands to process: {len(ligand_list)}")
        chemberta_embedding_block.set_input(ligand_list)
        chemberta_embedding_block.run(batch_size=128)
        myligands_embd = chemberta_embedding_block.get_output()

    myproteins_embd = None
    if embed in ['protein', 'both']:
        prott5_embedding_block = Prott5EmbeddingBlock()
        prott5_embedding_block.set_input(myproteins)
        prott5_embedding_block.run()
        myproteins_embd = prott5_embedding_block.get_output()

    if output_dir:
        data_directory = os.path.join(output_dir, 'embeddings')
        analysis_output_dir = output_dir
    else:
        result_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
        data_directory = os.path.join(result_dir, 'embeddings')
        analysis_output_dir = result_dir

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    
    if not os.path.exists(analysis_output_dir):
        os.makedirs(analysis_output_dir)

    if myligands_embd is not None:
        csv_path = os.path.join(data_directory, output_file + '.csv')
        
        # Save a version with flattened embeddings for CSV (optional)
        csv_df = myligands_embd.copy()
        csv_df['embedding_str'] = csv_df['embedding'].apply(lambda x: ','.join(map(str, x)))
        csv_df.drop('embedding', axis=1).to_csv(csv_path, index=False)
        
        logging.info(f"Saved embeddings for {len(myligands_embd)} ligands to {data_directory}")
        
        # Output statistics
        click.echo("\n--- Ligand Embedding Statistics ---")
        click.echo(f"Number of embeddings: {len(myligands_embd)}")
        if not myligands_embd.empty:
            click.echo(f"Embedding dimension: {len(myligands_embd['embedding'].iloc[0])}")
        click.echo(f"CSV file: {csv_path}")
        click.echo("------------------------------------\n")

    if myproteins_embd is not None:
        protein_csv_path = os.path.join(data_directory,'proteins_embeddings.csv')
        myproteins_embd.to_csv(protein_csv_path, index=False)
        
        logging.info(f"Saved {len(myproteins_embd)} protein embeddings to {protein_csv_path}")

        # Output statistics
        click.echo("--- Protein Embedding Statistics ---")
        click.echo(f"Number of embeddings: {len(myproteins_embd)}")
        click.echo(f"CSV file: {protein_csv_path}")
        click.echo("------------------------------------\n")

    # Run similarity analysis
    if myligands_embd is not None and not myligands_embd.empty:
        click.echo("Running ligand similarity analysis...")
        ligands_for_analysis = myligands.loc[myligands_embd.index]
        run_similarity_analysis(
            ligands_for_analysis,
            myligands_embd,
            text_only,
            analysis_output_dir
        )

    if myproteins_embd is not None and not myproteins_embd.empty:
        click.echo("Running protein similarity analysis...")
        proteins_for_analysis = myproteins.loc[myproteins_embd.index]
        run_protein_similarity_analysis(
            proteins_for_analysis,
            myproteins_embd,
            text_only,
            analysis_output_dir
        )

if __name__ == "__main__":
    main()
