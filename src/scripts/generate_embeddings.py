import sys
import os
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pipeline_blocks.preembedding_block import PreEmbeddingBlock
from pipeline_blocks.embedding_block import EmbeddingBlock

prembdblk = PreEmbeddingBlock('https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202506_tsv.zip')
#prembdblk = PreEmbeddingBlock('https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202506_tsv.zip')
#optional for full dataset use url: https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202506_tsv.zip
prembdblk.run()
myligands, myproteins = prembdblk.get_output()
print("Sample proteins (first 2):", myproteins[:2])
print("Total number of proteins:", len(myproteins))
embdblk = EmbeddingBlock()
embdblk.set_input(myligands, myproteins)
embdblk.run()
myligands_embd, myproteins_embd = embdblk.get_output()
data_directory = 'data/embeddings'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

if myligands_embd is not None:
    print("Saving ligands embeddings...")
    df_lig = pd.DataFrame([t.tolist() for t in myligands_embd])
    df_lig.to_csv(os.path.join(data_directory, 'ligands_embeddings.csv'), index=False)

if myproteins_embd is not None:
    print("Saving protein embeddings...")
    df_prot = pd.DataFrame([t.tolist() for t in myproteins_embd])
    df_prot.to_csv(os.path.join(data_directory, 'proteins_embeddings_all.csv'), index=False)
