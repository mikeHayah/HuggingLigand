# **HuggingLigand**: Learning the Language of a Molecular Hug
**A Deep Learning framework for protein–ligand affinity prediction.**

## Project Description
HuggingLigand is a modular deep learning framework for generating high-dimensional embeddings from protein sequences and small molecule structures — with the ultimate goal of predicting binding affinities between them. This prediction task is central to areas like drug discovery, biophysics, and computational biology, where determining how tightly a small molecule ligand binds to a protein target is crucial.  
  
This pipeline currently focuses on generating and managing protein and ligand embeddings using state-of-the-art transformer-based language models:
* ProtT5: a protein language model pretrained on millions of protein sequences.  
* ChemBERTa: a molecular language model trained on SMILES representations of chemical compounds. [![arXiv](https://img.shields.io/badge/arXiv-2209.01712-b31b1b.svg)](https://arxiv.org/abs/2209.01712) [![Dataset](https://img.shields.io/badge/HuggingFace-ChemBERTa-blue.svg)](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)


---

## Why HuggingLigand?

* **Modular Design**: Each pipeline block (preprocessing, embedding, postprocessing) is independent, extensible, and easy to maintain.

* **State-of-the-art Models**: Leverages powerful transformer architectures without requiring deep NLP knowledge.

* **CLI-Friendly**: Seamlessly integrates into workflows via a clean, intuitive command-line interface.

* **Reproducibility**: Fully configurable through Poetry-managed environments for reliable, consistent results. while our [**Hugging-Ligand-Embeddings**](https://huggingface.co/datasets/RSE-Group11/Hugging-Ligand-Embeddings) dataset card on the Hugging Face Hub providing versioned, reproducible embedding datasets for consistent experiments.

---

## Current Features

* Preprocessing and formatting of **BindingDB**  datasets. 
* A 'Bindingdata' class for extensible and efficient data manipulation.
* Automatic embedding generation for protein sequences and ligand SMILES strings.  
* Export of embedding datasets for downstream tasks.  
* Command-line interface (CLI) for seamless embedding generation and data preparation.  

## Project Roadmap

The current release covers embedding generation and data preparation steps. The next development phase will integrate these embeddings into a binding affinity prediction model, enabling end-to-end learning from raw molecular data to continuous affinity values(e.g., Kd, Ki, or IC50 values).
Future plans include:
- Developing an affinity prediction module.
- Implementing result visualization and interpretability tools.
- Expanding model support for additional molecular representations.

---

## Installation and Usage
You can install HuggingLigand either from source for development, or as a pre-built wheel for normal use.
### From Source
To install HuggingLigand including source codes, clone the repository and install the required dependencies with `poetry`:

```bash
git clone https://codebase.helmholtz.cloud/tud-rse-pojects-2025/group-11.git
cd group-11
poetry install
```

### From Wheel
Download the latest .whl file from the dist/ directory and install it in your virtual environment:

```bash
pip install huggingligand-1.0.0-py3-none-any.whl
```

### From [![TestPyPI](https://img.shields.io/badge/TestPyPI-huggingligand-blue)](https://test.pypi.org/project/huggingligand/)(Sandbox)
For testing the latest development build hosted on TestPyPI, you can install it directly via:

```bash
pip install -i https://test.pypi.org/simple/ huggingligand
```


### Usage (Basic check)
**Then execute the CLI script**:
```bash
huggingligand --help
```

---

## Dataset  

**BindingDB** — a curated dataset of experimentally measured protein-ligand binding affinities [![DOI](https://img.shields.io/badge/FAIRsharing-DOI-10.25504%2FFAIRsharing.3b36hk-blue.svg)](https://doi.org/10.25504/FAIRsharing.3b36hk)


**HuggingEmbeddings** - We store the embeddings dataset on zenodo 
Also, you can find versioned Dataset card on [![Dataset](https://img.shields.io/badge/HuggingFace-HuggingLigand_Embeddings-blue.svg)](https://huggingface.co/datasets/RSE-Group11/Hugging-Ligand-embeddings)

---

## Running the application

Once installed, you can run HuggingLigand to process data and generate embeddings.
You can create a folder to store data and results and run the the app with (it is recommended to use --rows argument to reduce runtime):

```bash
mkdir huggingligand
cd huggingligand
huggingligand --source https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202506_tsv.zip --rows 100
```

And get all available CLI options with:
```bash
huggingligand --help
```
You can also test it via:
```bash
poetry run python src/cli/cli.py --help
```

---

## HuggingLigand Directory Structure
<pre>  
'''
HuggingLigand/
├── src/
│   ├── scripts/                        # CLI scripts for training, inference, etc.
│   │   ├── train_huggingligand.py
│   │   └── evaluate_huggingligand.py
│   │
│   ├── pipeline_blocks/
│   │   ├── preembedding_block.py       # Data cleaning, transformation, splitting
│   │   ├── embedding_block.py          # Generate embeddings and save them
│   │   ├── postembedding_block.py      # Data hugging, and resplitting into train/valid/test
│   │   ├── training_block.py           # training for affinity model
│   │   └── evaluation_block.py         # infer the affinity of couple of protein and ligand
│   │
│   ├── models/
│   │   ├── protT5_embedding.py         # Wrapper & pipeline for ProtT5 embedding
│   │   ├── chemberta_embedding.py      # Wrapper & pipeline for ChemBERTa embedding
│   │   ├── affinity_predictor.py       # Third model for affinity prediction
│   │   └── utils/                      # Custom layers, loss functions, metrics etc.
│   │
│   ├── modules/                        # Any object to be used in pipeline blocks
│   │   ├── downloader                  # Data downloading
│   │   ├── reformmater                 # Put Data into good format
│   │   ├── preprocessor                # Apply processing functions on Data
│   │   └── ....
│   │
│   └──  config/
│       ├── config.yaml                 # General config for paths, hyperparameters
│       └── model_params.yaml           # Architecture, optimizer settings etc.
│   
├── tests/                    # Unit and integration tests for models, pipelines
│   ├── generate_embeddings.py
│   ├── train_affinity_model.py
│   ├── infer_affinity.py
│   └── evaluate_model.py
│
├── scripts/                  # CLI scripts for training, inference, etc.
│   ├── train_huggingligand.py
│   └── evaluate_huggingligand.py
│
├── logs/                     # Training and evaluation logs
│
├── results/                  # Model predictions, performance plots
│
├── requirements.txt
├── environment.yml           # Conda environment (if used)
├── .gitlab-ci.yml            # CI/CD configuration for GitLab
├── README.md
├── LICENSE
└── (Other metadata)
'''</pre> 

## Contributing



## Links & Resources


## License


## Acknowledgements


