# HuggingLigand: Where proteins embrace ligands through deep learning
<<<<<<< HEAD
<pre>
HuggingLigand is a DeepLearning pipeline designed for predicting the binding affinity between proteins and ligands. This prediction task is central to areas like drug discovery, biophysics, and computational biology, where determining how tightly a small molecule ligand binds to a protein target is crucial.
The pipeline uses state-of-the-art transformer models to automatically convert molecular and sequence data into rich, high-dimensional embeddings:
ProtT5: a protein language model pretrained on millions of protein sequences.
ChemBERTa: a molecular language model trained on SMILES representations of chemical compounds.
These embeddings are concatenated and fed into a customizable  model that learns to predict continuous binding affinity values (e.g., Kd, Ki, or IC50 values).
</pre> 
=======

>>>>>>> 5cf6f60 (update file structure of the project)

## Dataset  

**BindingDB** — a curated dataset of experimentally measured protein-ligand binding affinities.

---


##  Workflow  

1. **Data Preprocessing**
2. **Protein and Ligand Embedding Generation**
3. **Data Postprocessing**
3. **Training Affinity Prediction Model**
4. **Model Evaluation**
5. **Visualization of Results**


---
<<<<<<< HEAD

## Directory Structure
<pre>  
'''
HuggingLigand/

Data-repo
├── data/
│   ├── raw/                  # Original BindingDB data 
│   ├── processed/            # Preprocessed/cleaned datasets
│   ├── embeddings/           # Precomputed protein and ligand embeddings
│   └── huggings/             # Combined embeddings for affinity model
│        ├── train/           # Training dataset
│        ├── valid/           # Validation dataset
│        └── test/            # testing dataset
└── Model_registery           # Save Model checkpoint

Code-repo
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
├── tests/                              # Unit and integration tests for models, pipelines
=======

## Directory Structure
<pre>  
'''
HuggingLigand/
│
├── data/
│   ├── raw/                  # Original BindingDB data 
│   ├── processed/            # Preprocessed/cleaned datasets
│   ├── embeddings/           # Precomputed protein and ligand embeddings
│   └── huggings/             # Combined embeddings for affinity model
│        ├── train/           # Training dataset
│        ├── valid/           # Validation dataset
│        └── test/            # testing dataset
│   ├── Model_registery       # Save Model checkpoint
│
├── models/
│   ├── protT5_embedding.py   # Wrapper & pipeline for ProtT5 embedding
│   ├── chemberta_embedding.py# Wrapper & pipeline for ChemBERTa embedding
│   ├── affinity_predictor.py # Third model for affinity prediction
│   └── utils/                # Custom layers, loss functions, metrics etc.
│
├── pipeline_blocks/
│   ├── preembedding_block.py       # Data cleaning, transformation, splitting
│   ├── embedding_block.py          # Generate embeddings and save them
│   ├── postembedding_block.py      # Data hugging, and resplitting into train/valid/test
│   ├── training_block.py           # training for affinity model
│   └── evaluation_block.py         # infer the affinity of couple of protein and ligand
│
├── modules/                        # Any object to be used in pipeline blocks
│   ├── downloader                  # Data downloading
│   ├── reformmater                 # Put Data into good format
│   ├── preprocessor                # Apply processing functions on Data
│   └── ....
│
├── config/
│   ├── config.yaml           # General config for paths, hyperparameters
│   └── model_params.yaml     # Architecture, optimizer settings etc.
│
├── tests/                    # Unit and integration tests for models, pipelines
>>>>>>> 5cf6f60 (update file structure of the project)
│   ├── generate_embeddings.py
│   ├── train_affinity_model.py
│   ├── infer_affinity.py
│   └── evaluate_model.py
│
<<<<<<< HEAD
=======
├── scripts/                  # CLI scripts for training, inference, etc.
│   ├── train_huggingligand.py
│   └── evaluate_huggingligand.py
│
>>>>>>> 5cf6f60 (update file structure of the project)
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