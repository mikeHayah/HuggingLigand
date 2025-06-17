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
└── (Other metadata)
'''</pre> 