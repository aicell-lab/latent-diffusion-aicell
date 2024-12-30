#!/bin/bash
#SBATCH -A berzelius-2024-230    # Your project/account
#SBATCH --gpus=1                 # Number of GPUs you need
#SBATCH -t 1-00:00:00            # Time limit (e.g. 1 day)
#SBATCH --cpus-per-gpu=16        # Adjust CPU cores if needed
#SBATCH --mem=128G               # Adjust memory if needed (128GB as an example)
#SBATCH -J vae                   # Job name
#SBATCH -o logs/%x_%j.out        # Standard output log (optional)
#SBATCH -e logs/%x_%j.err        # Standard error log (optional)

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/ldm

# Run your training command
python main.py --base configs/autoencoder/autoencoder_kl_8x8x64_jump_cuda.yaml -t