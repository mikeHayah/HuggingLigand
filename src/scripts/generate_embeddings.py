from pipeline_blocks.preembedding_block import PreEmbeddingBlock
from pipeline_blocks.embedding_block import EmbeddingBlock
import os

prembdblk = PreEmbeddingBlock('https://www.bindingdb.org/rwd/bind/downloads/BindingDB_Covid-19_202506_tsv.zip')
prembdblk.run()
myligands, myproteins = prembdblk.get_output()
embdblk = EmbeddingBlock()
embdblk.set_input(myligands, myproteins)
embdblk.run()
myligands_embd, myproteins_embd = embdblk.get_output()
data_directory = 'data/embeddings'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
myligands_embd.toself.reformated_path_csv(data_directory, index=False)
myproteins_embd.to_csv(data_directory, index=False)
