# Project pivoting — 12.06.2025

##  Introduction

Following our last meeting, we collectively decided to pivot from the *GitInScene* project to a new project focusing on **predicting binding affinity between proteins and ligands using deep learning models**. The primary motivation for this change was the opportunity to work on a problem with strong practical relevance in drug discovery, established datasets, and a broader application of state-of-the-art machine learning techniques.

This document formalizes the updated project scope, responsibilities, and technical decisions.

---

##  New Project Name  

**HuggingLigand: Where proteins embrace ligands through deep learning.**

---

## Project Tasks

- **Protein Embedding Implementation (ProtT5)** 
- **Ligand Embedding Implementation (ChemBERTa)** 
- **Affinity Prediction Model Implementation** 
- **Data Preprocessing and Pipeline Design** 
- **Benchmarking and Evaluation** 

---

## Dataset  

**BindingDB** — a curated dataset of experimentally measured protein-ligand binding affinities.

Data files to be processed into train/validation/test splits for model training.

---

## Preprocessing and Data Handling  

- Clean and format BindingDB data 
- Generate protein embeddings via **ProtT5**
- Generate ligand embeddings via **ChemBERTa**
- Merge embeddings with affinity values (Kd)

---

##  Analysis Workflow  

1. **Data Preprocessing**
2. **Protein and Ligand Embedding Generation**
3. **Data Postprocessing**
3. **Training Affinity Prediction Model**
4. **Model Evaluation**
5. **Visualization of Results**

---

## Software Packages  


`transformers`       Hugging Face models (ProtT5, ChemBERTa) 
`torch`              Deep Learning framework          |
`pandas`             Data handling                     
`numpy`              Numerical operations              
`scikit-learn`       Metrics, splitting, preprocessing 
`matplotlib / seaborn`   Visualization                 
`pytest`             Unit testing                      

**Python version:** 3.10+

---

## Planned Directory Structure
HuggingLigand/
│
├── data/
│   ├── raw/                  # Original BindingDB data 
│   ├── processed/            # Preprocessed/cleaned datasets
│   └── embeddings/           # Precomputed protein and ligand embeddings
│   └── huggings/             # Combined embeddings for affinity model
|       ├── train/            # Training dataset
|       ├── valid/            # Validation dataset
|       └── test/             # testing dataset
│
├── models/
│   ├── protT5_embedding.py   # Wrapper & pipeline for ProtT5 embedding
│   ├── chemberta_embedding.py# Wrapper & pipeline for ChemBERTa embedding
│   ├── affinity_predictor.py # Third model for affinity prediction
│   └── utils/                # Custom layers, loss functions, metrics etc.
│
├── pipelines/
│   ├── preembedding.py       # Data cleaning, transformation, splitting
│   ├── embedding_pipeline.py # Generate embeddings and save them
│   ├── postembedding.py      # Data hugging, and resplitting into train/valid/test
│   ├── training_pipeline.py  # Full training pipeline for affinity model
│   └── evaluation.py         # Model evaluation, plotting, performance metrics
│
│
├── config/
│   ├── config.yaml           # General config for paths, hyperparameters
│   └── model_params.yaml     # Architecture, optimizer settings etc.
│
├── tests/                    # Unit and integration tests for models, pipelines
│
├── scripts/                  # CLI scripts for training, inference, etc.
|   ├── generate_embeddings.py
|   ├── train_affinity_model.py
|   ├── infer_affinity.py
|   └── evaluate_model.py
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