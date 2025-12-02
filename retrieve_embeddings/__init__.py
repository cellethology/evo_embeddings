"""
Retrieve embeddings from Evo models.

This package provides utilities for extracting embeddings from Evo models
for DNA sequences stored in FASTA files.
"""

from .retrieve_embeddings import (
    extract_embeddings_batch,
    process_sequences,
    setup_model_for_embeddings,
    main,
)
from .util import (
    load_sequences_from_fasta,
    save_embeddings_to_npz,
    validate_sequence,
)

__all__ = [
    "extract_embeddings_batch",
    "process_sequences",
    "setup_model_for_embeddings",
    "main",
    "load_sequences_from_fasta",
    "save_embeddings_to_npz",
    "validate_sequence",
]

