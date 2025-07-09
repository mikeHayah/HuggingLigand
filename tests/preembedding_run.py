from src.pipeline_blocks.preembedding_block import PreEmbeddingBlock

prembd = PreEmbeddingBlock('https://www.bindingdb.org/rwd/bind/downloads/BindingDB_Covid-19_202506_tsv.zip')
prembd.run()