import os
import torch
import multiprocessing as mp

from src.pipeline_blocks.preembedding_block import PreEmbeddingBlock
from src.pipeline_blocks.prott5_embedding_block import Prott5EmbeddingBlock
from src.pipeline_blocks.chemberta_embedding_block import ChembertaEmbeddingBlock

if __name__ == "__main__":

    # Optimize CPU settings before importing other modules
    cpu_cores = mp.cpu_count()
    torch.set_num_threads(cpu_cores)
    torch.set_num_interop_threads(1)
    os.environ['OMP_NUM_THREADS'] = str(cpu_cores)
    os.environ['MKL_NUM_THREADS'] = str(cpu_cores)

    print(f"CPU optimization: Using {cpu_cores} threads")

    # smaller dataset
    preembedding_block = PreEmbeddingBlock('https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202506_tsv.zip')
    output_file = "ligands_embeddings"
    # full dataset
    #preembedding_block = PreEmbeddingBlock('https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202506_tsv.zip')
    #output_file = "ligands_embeddings_full"
    preembedding_block.run()
    myligands, myproteins = preembedding_block.get_output()

    prott5_embedding_block = Prott5EmbeddingBlock()
    prott5_embedding_block.set_input(myproteins)
    prott5_embedding_block.run()
    myproteins_embd = prott5_embedding_block.get_output()

    chemberta_embedding_block = ChembertaEmbeddingBlock()
    ligand_list = myligands['Ligand SMILES'].tolist()
    print(f"Total number of ligands to process: {len(ligand_list)}")
    chemberta_embedding_block.set_input(ligand_list)
    # Increase batch size significantly for CPU processing - this is the main speedup
    chemberta_embedding_block.run(batch_size=128)  # Much larger batch size for CPU
    myligands_embd = chemberta_embedding_block.get_output()

    data_directory = 'data/embeddings'

    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    if myligands_embd is not None:
        # Save as pickle to preserve numpy arrays, and CSV for inspection
        myligands_embd.to_pickle(os.path.join(data_directory, output_file + '.pkl'))
        # Save a version with flattened embeddings for CSV (optional)
        csv_df = myligands_embd.copy()
        csv_df['embedding_str'] = csv_df['embedding'].apply(lambda x: ','.join(map(str, x)))
        csv_df.drop('embedding', axis=1).to_csv(os.path.join(data_directory, output_file + '.csv'), index=False)
        print(f"Saved embeddings for {len(myligands_embd)} ligands to {data_directory}")
    
    if myproteins_embd is not None:
        myproteins_embd.to_csv(os.path.join(data_directory,'proteins_embeddings.csv'), index=False)
