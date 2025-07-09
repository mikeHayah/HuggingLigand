# HuggingLigand: Where proteins embrace ligands through deep learning
<pre>
HuggingLigand is a DeepLearning pipeline designed for predicting the binding affinity between proteins and ligands. This prediction task is central to areas like drug discovery, biophysics, and computational biology, where determining how tightly a small molecule ligand binds to a protein target is crucial.
The pipeline uses state-of-the-art transformer models to automatically convert molecular and sequence data into rich, high-dimensional embeddings:
ProtT5: a protein language model pretrained on millions of protein sequences.
ChemBERTa: a molecular language model trained on SMILES representations of chemical compounds.
These embeddings are concatenated and fed into a customizable  model that learns to predict continuous binding affinity values (e.g., Kd, Ki, or IC50 values).
</pre> 


## Dataset  

**BindingDB** — a curated dataset of experimentally measured protein-ligand binding affinities (Liu et al., 2025) found at [BindingDB](https://www.bindingdb.org).

---


##  Workflow  

1. **Data Preprocessing**
2. **Protein and Ligand Embedding Generation**
3. **Data Postprocessing**
3. **Training Affinity Prediction Model**
4. **Model Evaluation**
5. **Visualization of Results**


---

## Datasets and Model 
We store the embeddings dataset on Hugging Face Hub:[Hugging-Ligand-Embeddings](https://huggingface.co/datasets/RSE-Group11/Hugging-Ligand-Embeddings)
and our Model [HuggingLigand Affinity Predictor](https://huggingface.co/RSE-Group11/hugging-ligand-affinity-predictor)
<pre>

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
</pre> 
---

## Running the application
To simply run the application, get the current *.whl file from the dist directory here and install it into a venv with pip by inserting the correct version number:
```bash
pip install huggingligand-1.0.0-py3-none-any.whl
```

You can then create a folder to store data and results and run the the app with (it is recommended to use --rows argument to reduce runtime):
```bash
mkdir huggingligand
cd huggingligand
huggingligand --source https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202506_tsv.zip --rows 100
```

And get all available options with:
```bash
huggingligand --help
```

---

## Installation and Usage
To install HuggingLigand including source codes, clone the repository and install the required dependencies with `poetry`:
```bash
git clone https://codebase.helmholtz.cloud/tud-rse-pojects-2025/group-11.git
cd group-11
poetry install
```

Then execute the CLI script:
```bash
poetry run python src/scripts/cli.py --source https://www.bindingdb.org/rwd/bind/downloads/BindingDB_BindingDB_Articles_202506_tsv.zip --rows 100
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

