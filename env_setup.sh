#!/usr/bin/env bash
conda create -y -n evo_embeddings python=3.12
conda activate evo_embeddings
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge flash-attn=2.7.4
pip install evo-model
pip install hf_transfer


# 1. Initialize conda for bash
#    (pick the install you actually have: miniconda3 or anaconda3)
# if [ -f "/workspace/miniconda3/etc/profile.d/conda.sh" ]; then
#     . "/workspace/miniconda3/etc/profile.d/conda.sh"
# elif [ -f "/workspace/anaconda3/etc/profile.d/conda.sh" ]; then
#     . "/workspace/anaconda3/etc/profile.d/conda.sh"
# fi

# # If for some reason conda.sh doesn't exist but conda is installed under ~/miniconda3,
# # you can fall back to PATH-based activation:
# if ! command -v conda >/dev/null 2>&1; then
#     if [ -d "/workspace/miniconda3/bin" ]; then
#         export PATH="/workspace/miniconda3/bin:$PATH"
#     elif [ -d "/workspace/anaconda3/bin" ]; then
#         export PATH="/workspace/anaconda3/bin:$PATH"
#     fi
# fi