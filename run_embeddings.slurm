#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=protT5_embeddings
#SBATCH --output=logs/protT5_embeddings_%j.out
#SBATCH --error=logs/protT5_embeddings_%j.err
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00

# Optional: Load your environment modules (if using modules)
module purge
module load GCC/11.3.0  Python/3.10.4  CUDA/11.7.0  cuDNN/8.4.0.27-CUDA-11.7.0

# Activate Poetry (adjust if installed differently)
export PATH="$HOME/.local/bin:$PATH"

# If using cached segments helps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to project directory
cd $HOME/group-11

# Run the script via poetry
poetry run python src/scripts/generate_embeddings.py