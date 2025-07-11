# **HuggingLigand**: Learning the Language of a Molecular Hug
**A Deep Learning framework for protein–ligand affinity prediction.**

[![arXiv](https://img.shields.io/badge/arXiv-2209.01712-b31b1b.svg)](https://arxiv.org/abs/2209.01712) [![Dataset](https://img.shields.io/badge/HuggingFace-ChemBERTa-yellow.svg)](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)

## Project Description

HuggingLigand is a modular deep learning framework for generating high-dimensional embeddings from protein sequences and small molecule structures — with the ultimate goal of predicting binding affinities between them. This prediction task is central to areas like drug discovery, biophysics, and computational biology, where determining how tightly a small molecule ligand binds to a protein target is crucial.  
  
This pipeline currently focuses on generating and managing protein and ligand embeddings using state-of-the-art transformer-based language models:
* ProtT5: a protein language model pretrained on millions of protein sequences.  
* ChemBERTa: a molecular language model trained on SMILES representations of chemical compounds.  

## Why HuggingLigand?

* **Modular Design**: Each pipeline block (preprocessing, embedding, postprocessing) is independent, extensible, and easy to maintain.

* **State-of-the-art Models**: Leverages powerful transformer architectures without requiring deep NLP knowledge.

* **CLI-Friendly**: Seamlessly integrates into workflows via a clean, intuitive command-line interface.

* **Reproducibility**: Fully configurable through Poetry-managed environments for reliable, consistent results. while our [**Hugging-Ligand-Embeddings**](https://huggingface.co/datasets/RSE-Group11/Hugging-Ligand-Embeddings) dataset card on the Hugging Face Hub providing versioned, reproducible embedding datasets for consistent experiments.

## Current Features

* Preprocessing and formatting of **BindingDB**  datasets. 
* A 'Bindingdata' class for extensible and efficient data manipulation.
* Automatic embedding generation for protein sequences and ligand SMILES strings.  
* Export of embedding datasets for downstream tasks.  
* Command-line interface (CLI) for seamless embedding generation and data preparation.  

## Potential Future Steps

The current release covers embedding generation and data preparation steps. The next development phase will integrate these embeddings into a binding affinity prediction model, enabling end-to-end learning from raw molecular data to continuous affinity values(e.g., Kd, Ki, or IC50 values).
Future plans include:
- Developing an affinity prediction module.
- Implementing result visualization and interpretability tools.
- Expanding model support for additional molecular representations.

## Installation and Usage

You can install HuggingLigand either from source for development, or as a pre-built wheel for normal use.

### From Source
To install HuggingLigand including source codes, clone the repository and install the required dependencies with `poetry`:

```bash
git clone https://codebase.helmholtz.cloud/tud-rse-pojects-2025/group-11.git
cd group-11
poetry install
```

You can then run the CLI script directly from the source directory and get help with:

```bash
poetry run python src/cli/cli.py --help
```

**Potential options include:**

- `--source`: URL to the dataset (default is a BindingDB dataset).
- `-v` or `--verbose`: Increase verbosity level (e.g., `-v` for WARNING, `-vv` for INFO, `-vvv` for DEBUG).
- `--text-only`: Suppress graphical output.
- `--rows`: Number of rows to process from the dataset.
- `--output-dir`: Directory to save the embeddings and analysis results.
- `--embed`: Specify what to embed (options are 'ligand', 'protein', or 'both').

**To evaluate now a BindingDB dataset on 100 dataset rows, you can run:**

```bash
poetry run python src/cli/cli.py --source https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202506_tsv.zip --rows 100
```

### From TestPyPI
[![TestPyPI](https://img.shields.io/badge/TestPyPI-huggingligand-blue)](https://test.pypi.org/project/huggingligand/)  

For testing the latest build hosted on TestPyPI, you can install it directly (after having created a virtual environment) via:

```bash
pip install -i https://test.pypi.org/simple/ huggingligand
```

**Then execute the CLI script and run with options as a normal python package:**
```bash
huggingligand --help
```

## Dataset

**BindingDB** - A curated dataset of experimentally measured protein-ligand binding affinities.  
[![DOI](https://img.shields.io/badge/DOI-10.25504%2FFAIRsharing.3b36hk-blue.svg)](https://doi.org/10.25504/FAIRsharing.3b36hk)

**HuggingEmbeddings** - The embeddings dataset for proteins and ligands is stored on zenodo. There, a versioned dataset card can be found.  
[![DOI](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F5960-yellow)](https://doi.org/10.57967/hf/5960)

## HuggingLigand Directory Structure

```
HuggingLigand/
├── src/
│   ├── cli/
│   │   ├── cli.py
│   │   └── analysis.py
│   │
│   ├── scripts/                         # scripts for generating embeddings training, inference, etc.
│   │   ├── generate_embeddings.py                     
│   │   ├── train_huggingligand.py
│   │   └── evaluate_huggingligand.py
│   │
│   ├── pipeline_blocks/                # Blocks for building 
│   │   ├── preembedding_block.py       # Data cleaning, transformation, splitting
│   │   ├── prott5_embedding_block.py   # Generate embeddings for proteins
│   │   ├── chemberta_embedding_block.py# Generate embeddings for Ligands
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
│   ├── modules/                          # Utils
│   │   ├── downloader.py                 # Data downloading
│   │   ├── reformatter.py                # Put Data into good format
│   │   ├── bindingdata.py                # Apply processing functions on Data
│   │   ├── loader.py 
│   │   └── embedding_utils.py
│   │
│   └──  config/
│       └── config.yaml                 # General config for paths, hyperparameters
│   
├── tests/                              # Unit and integration tests for models, pipelines
│   ├── test_preembedding.py
│   ├── test_prott5_embedder.py
│   ├── test_chemberta_embedder.py
│   ├── test_embedding_utils.py
│   ├── test_bindingdata.py
│   ├── test_loaders.py
│   ├── test_downloader.py
│   └── test_reformatter.py
│
├── logs/                     # Training and evaluation logs
│
├── results/                  # Model predictions, performance plots
│
├── .gitlab-ci.yml            # CI/CD configuration for GitLab
├── CITATION.cff
├── CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── poetry.lock
├── pyproject.lock
├── pytest.ini
└── README.md
```

## Contributing
Thank you for your interest in contributing to HuggingLigand! We welcome contributions from the community to help improve protein-ligand binding affinity prediction.  

By participating in this project, you agree to foster a respectful, inclusive, and collaborative environment. Be considerate in your interactions with others, and help us maintain a positive community.  

For detailed guidelines on how to contribute — including setting up your development environment, reporting issues, and submitting pull requests — please refer to the CONTRIBUTING.md file.  

## Links & Resources
* Project Repository: [HuggingLigand on GitLab](https://codebase.helmholtz.cloud/tud-rse-pojects-2025/group-11/-/tree/main?ref_type=heads)

* TestPyPI Sandbox: [huggingligand on TestPyPI](https://test.pypi.org/project/huggingligand/)

* Dataset on Zenodo sandbox publication: [huggingligand on Zenodo]()

* Dataset on Hugging Face Hub: [Hugging-Ligand-Embeddings](https://huggingface.co/datasets/RSE-Group11/Hugging-Ligand-embeddings)

* ChemBERTa Pretrained Model: [ChemBERTa](https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1)

* ProtT5 Pretrained Model: [ProtT5-XL-UniRef50]()

* BindingDB Data Source: [BindingDB](https://doi.org/10.25504/FAIRsharing.3b36hk)

## License

This project is licensed under the terms of the MIT License.
It also makes use of third-party components, which are subject to their respective licenses:

* ProtT5: MIT License / Academic Free License v3.0
* ChemBERTa: MIT License
* BindingDB: Creative Commons Attribution 3.0 License

Please review the LICENSE file for full details.

## Acknowledgements

This work was carried out as part of the Lab Course: Research in Software Engineering at TU-Dresden, organized by the Computational Science Department at Helmholtz-Zentrum Dresden-Rossendorf (HZDR). We would like to sincerely thank Dr. Guido Juckeland and course coordinators for his supervision, guidance, and valuable feedback throughout the project.

Our gratitude also goes to the course coordinator Katja Linnemann for organizing and facilitating the course framework and providing an engaging environment for applied software engineering research.

We further acknowledge the use of codebase.helmholtz.cloud as the primary platform for version control, collaborative development, and project management.


