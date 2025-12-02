# Evo Embeddings Framework

## Overview

The Evo Embeddings framework provides tools to:
- Extract embeddings from FASTA sequences using pre-trained Evo models
- Process sequences in batches for efficient inference on single or multiple GPUs
- Support multiple model variants with varying context lengths
- Handle variable-length sequences with automatic padding and BOS token management

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for faster inference)
- Miniconda or Anaconda (for environment management)
- CUDA toolkit (for GPU support)

## Environment Setup

### 1. Install Miniconda (if not already installed)

Download and install Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

### 2. Create Virtual Environment and Install Dependencies

Run the environment setup script:

```bash
bash env_setup.sh
```

Or manually create and activate the environment:

```bash
conda create -n evo_embeddings python=3.12
conda activate evo_embeddings
```

### 3. Install the Package

Install the package in editable mode:

```bash
pip install -e .
```

This will install the `evo` and `retrieve_embeddings` packages along with their dependencies.

### 4. Model Download

Evo models are automatically downloaded from HuggingFace when first used. The models will be cached in your HuggingFace cache directory (typically `~/.cache/huggingface/`).

**Note:** Model files can be large (several GB). Ensure you have sufficient disk space and bandwidth.

## Available Models

| Model Name | Context Length | Description |
|------------|---------------|-------------|
| `evo-1.5-8k-base` | 8,192 (8K) | Latest 1.5 version with 8K context (default) |
| `evo-1-8k-base` | 8,192 (8K) | Base model with 8K context |
| `evo-1-131k-base` | 131,072 (131K) | Base model with extended 131K context |
| `evo-1-8k-crispr` | 8,192 (8K) | Model fine-tuned on CRISPR sequences |
| `evo-1-8k-transposon` | 8,192 (8K) | Model fine-tuned on transposon sequences |

**Notes:**
- **Context Length**: Maximum number of tokens/base pairs the model can process in a single forward pass
- **Model Selection**: Use `evo-1-8k-base` or `evo-1.5-8k-base` for standard tasks. Use `evo-1-131k-base` for longer sequences up to 131K base pairs
- **Specialized Models**: The CRISPR and transposon models are fine-tuned for specific biological contexts

## Installation

### Install as a Package

```bash
cd /workspace/gLMs/evo_embeddings
pip install -e .
```

After installation, you can use the package in multiple ways:

## Usage

### Extract Embeddings from FASTA Sequences

The main script for extracting embeddings is `retrieve_embeddings/retrieve_embeddings.py`.

#### Basic Usage

```bash
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path test_files/test.fasta \
    --output_path embeddings.npz
```

#### Full Command with All Options

```bash
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path <path-to-input.fasta> \
    --output_path <path-to-output.npz> \
    --model_name evo-1.5-8k-base \
    --batch_size 8 \
    --device cuda:0 \
    --prepend_bos \
    --mean_pooling
```

#### Command-Line Arguments

- `--fasta_path`, `-i` (required): Path to input FASTA file containing DNA sequences
- `--output_path`, `-o` (optional): Path to output `.npz` file where embeddings will be saved (default: `./embeddings.npz`)
- `--model_name`, `-m` (optional): Evo model name to use (default: `evo-1.5-8k-base`)
  - Options: `evo-1.5-8k-base`, `evo-1-8k-base`, `evo-1-131k-base`, `evo-1-8k-crispr`, `evo-1-8k-transposon`
- `--batch_size`, `-b` (optional): Batch size for processing sequences (default: 8)
- `--device`, `-d` (optional): Device to run inference on (default: `cuda:0`)
  - Options: `cuda:0`, `cuda:1`, `cpu`, etc.
- `--prepend_bos` (optional): Prepend BOS (Beginning of Sequence) token to sequences (default: False)
- `--mean_pooling` (optional): Apply mean pooling to embeddings across sequence length. Averages per-token embeddings into a single vector per sequence, significantly reducing memory usage and output file size (default: False)

#### Output Format

The script outputs a compressed NumPy archive (`.npz`) file containing:
- `ids`: Array of sequence IDs from the FASTA file (string array)
- `embeddings`: Array of embeddings
  - **Without mean pooling**: Variable-length embeddings per sequence with shape `(sequence_length, embedding_dim)`
  - **With mean pooling** (`--mean_pooling`): Single vector per sequence with shape `(embedding_dim,)` - averages all per-token embeddings
  - Embedding dimension depends on the model (typically 1024 or 2048)

**Memory Usage**: Using `--mean_pooling` significantly reduces memory usage during processing and output file size, especially for long sequences. Recommended for downstream tasks that don't require per-token embeddings.

#### Examples

```bash
# Extract embeddings using the default model
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path test_files/test.fasta \
    --output_path output/embeddings.npz

# Extract embeddings using the extended context model
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path test_files/test.fasta \
    --output_path output/embeddings.npz \
    --model_name evo-1-131k-base \
    --batch_size 4

# Extract embeddings with BOS token prepended
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path test_files/test.fasta \
    --output_path output/embeddings.npz \
    --prepend_bos

# Extract embeddings with mean pooling (reduces memory usage)
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path test_files/test.fasta \
    --output_path output/embeddings_pooled.npz \
    --mean_pooling

# Extract embeddings with mean pooling and larger batch size
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path test_files/test.fasta \
    --output_path output/embeddings_pooled.npz \
    --mean_pooling \
    --batch_size 16

# Use the CRISPR-specific model
python -m retrieve_embeddings.retrieve_embeddings \
    --fasta_path test_files/test.fasta \
    --output_path output/embeddings_crispr.npz \
    --model_name evo-1-8k-crispr
```

### Loading Embeddings

You can load the saved embeddings in Python:

```python
import numpy as np

# Load embeddings
data = np.load('output/embeddings.npz', allow_pickle=True)
sequence_ids = data['ids']
embeddings = data['embeddings']  # Object array of variable-length arrays

print(f"Loaded {len(sequence_ids)} sequences")
print(f"Sequence IDs: {sequence_ids}")

# Access embeddings for each sequence
for i, seq_id in enumerate(sequence_ids):
    seq_embedding = embeddings[i]
    print(f"Sequence {seq_id}: embedding shape {seq_embedding.shape}")
    # Without mean pooling: (sequence_length, embedding_dim)
    # With mean pooling: (embedding_dim,)
```

**Note**: When using `--mean_pooling`, each embedding is a 1D array of shape `(embedding_dim,)` instead of `(sequence_length, embedding_dim)`. This is useful for tasks like sequence classification, similarity search, or clustering where a single vector per sequence is sufficient.

### Using the Package Programmatically

You can also use the package directly in Python:

```python
from retrieve_embeddings.retrieve_embeddings import (
    setup_model_for_embeddings,
    extract_embeddings_batch,
    process_sequences,
)
from retrieve_embeddings.util import load_sequences_from_fasta

# Load sequences
sequences, sequence_ids = load_sequences_from_fasta('test_files/test.fasta')

# Setup model
model, tokenizer = setup_model_for_embeddings(
    model_name='evo-1.5-8k-base',
    device='cuda:0'
)

# Extract embeddings for a batch
embeddings, seq_lengths = extract_embeddings_batch(
    model=model,
    tokenizer=tokenizer,
    sequences=sequences[:5],  # First 5 sequences
    device='cuda:0',
    prepend_bos=False
)

print(f"Extracted embeddings for {len(sequences)} sequences")
```

## Project Structure

```
evo_embeddings/
├── evo/                          # Evo model package
│   ├── models.py                 # Evo model class and loading
│   ├── scoring.py                # Sequence scoring utilities
│   ├── generation.py             # Sequence generation utilities
│   ├── utils.py                  # Model utilities
│   └── configs/                  # Model configuration files
│       ├── evo-1-8k-base_inference.yml
│       └── evo-1-131k-base_inference.yml
├── retrieve_embeddings/          # Embedding extraction package
│   ├── retrieve_embeddings.py   # Main embedding extraction script
│   ├── util.py                   # Utility functions (FASTA loading, validation, saving)
│   └── __init__.py               # Package exports
├── scripts/                       # Utility scripts
│   ├── score.py                  # Sequence scoring script
│   └── generate.py               # Sequence generation script
├── test_files/                   # Test data
│   └── test.fasta                # Example input file
├── env_setup.sh                  # Environment setup script
├── setup.py                      # Package setup configuration
└── README.md                     # This file
```

## Input Format

The input FASTA file should contain DNA sequences with standard FASTA format:

```
>sequence_id_1
ATCGATCAGTACGATCAGATTTAGACGT
>sequence_id_2
TTTTGGGCGCGCGGCATCGATCAGTACGATCAGATTTAGACGTAAAAAA
>sequence_id_3
AGCTGATGCTAGCAGTGACGATGACAGTACAGTACAGAT
```

**Requirements:**
- Sequences must contain only valid DNA characters: A, T, C, G
- Sequences are automatically converted to uppercase
- Invalid sequences are skipped with a warning
- Maximum sequence length: 131,072 base pairs (for 131K context model) or 8,192 base pairs (for 8K models)
- Minimum sequence length: 1 base pair

## Additional Features

### Sequence Scoring

You can also score sequences using the Evo models:

```bash
python -m scripts.score \
    --input-fasta test_files/test.fasta \
    --output-tsv scores.tsv \
    --model-name evo-1.5-8k-base \
    --device cuda:0
```

### Sequence Generation

Generate new sequences using the Evo models:

```bash
python -m scripts.generate \
    --model-name evo-1.5-8k-base \
    --prompt ACGT \
    --n-samples 10 \
    --n-tokens 100 \
    --temperature 1.0 \
    --top-k 4 \
    --device cuda:0
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:
- Reduce `--batch_size` (try 1, 2, or 4)
- Use `--mean_pooling` to reduce memory usage
- Use a smaller model (e.g., `evo-1-8k-base` instead of `evo-1-131k-base`)

### Module Not Found Errors

If you get import errors, make sure the package is installed:

```bash
pip install -e .
```
