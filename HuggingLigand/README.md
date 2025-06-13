# HuggingLigand: Where proteins embrace ligands through deep learning


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

<pre> ## Directory Structure '''

HuggingLigand/
│
├── data/
│   ├── raw/                  # Original BindingDB data 
│   ├── processed/            # Preprocessed/cleaned datasets
│   ├── embeddings/           # Precomputed protein and ligand embeddings
│   └── huggings/             # Combined embeddings for affinity model
│        ├── train/            # Training dataset
│        ├── valid/            # Validation dataset
│        └── test/             # testing dataset
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
│   ├── generate_embeddings.py
│   ├── train_affinity_model.py
│   ├── infer_affinity.py
│   └── evaluate_model.py
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
└── (Other metadata)'''</pre> 