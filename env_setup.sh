#!/usr/bin/env bash
conda create -y -n evo2_embeddings python=3.12
conda activate evo_embeddings
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge flash-attn=2.7.4
pip install evo-model
pip install hf_transfer