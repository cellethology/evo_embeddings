import argparse
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from evo import Evo
from evo.scoring import prepare_batch
from tqdm import tqdm

from retrieve_embeddings.util import (
    load_sequences_from_fasta,
    save_embeddings_to_npz,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_model_for_embeddings(
    model_name: str = "evo-1.5-8k-base",
    device: str = "cuda:0",
) -> Tuple[torch.nn.Module, object]:
    """
    Initialize Evo model and configure it to return embeddings instead of logits.
    
    Args:
        model_name: Name of the Evo model to load
        device: Device to run inference on (default: "cuda:0")
        
    Returns:
        Tuple of (model, tokenizer) where model is configured to return embeddings
    """
    logger.info(f"Loading Evo model: {model_name}...")
    evo_model = Evo(model_name)
    model, tokenizer = evo_model.model, evo_model.tokenizer
    model.to(device)
    model.eval()
    
    logger.info("Model loaded and configured for embedding extraction")
    return model, tokenizer


def extract_embeddings_batch(
    model: torch.nn.Module,
    tokenizer: object,
    sequences: List[str],
    device: str = "cuda:0",
    prepend_bos: bool = False,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Extract embeddings for a batch of sequences.

    Args:
        model: Evo model instance
        tokenizer: Evo tokenizer instance
        sequences: List of DNA sequences to process
        device: Device to run inference on (default: "cuda:0")
        prepend_bos: Whether to prepend BOS token (default: False)

    Returns:
        Tuple of (embeddings tensor, sequence_lengths list)
        Embeddings have shape [batch_size, max_seq_length + (1 if prepend_bos else 0), embedding_dim]
        sequence_lengths are the original sequence lengths (without BOS or padding)
    """
    # Prepare batch: tokenize and pad sequences
    input_ids, seq_lengths = prepare_batch(
        seqs=sequences,
        tokenizer=tokenizer,
        prepend_bos=prepend_bos,
        device=device,
    )

    # Extract embeddings using forward pass
    with torch.no_grad():
        embeddings, _ = model(input_ids)  # (batch, length, embed_dim)

    return embeddings, seq_lengths


def process_sequences(
    model: torch.nn.Module,
    tokenizer: object,
    sequences: List[str],
    sequence_ids: List[str],
    output_path: str,
    batch_size: int = 8,
    device: str = "cuda:0",
    prepend_bos: bool = False,
    mean_pooling: bool = False,
) -> None:
    """
    Process sequences in batches and extract embeddings.

    Args:
        model: Evo model instance
        tokenizer: Evo tokenizer instance
        sequences: List of DNA sequences to process
        sequence_ids: List of sequence identifiers corresponding to sequences
        output_path: Path to save the .npz file with all embeddings
        batch_size: Number of sequences to process per batch (default: 8)
        device: Device to run inference on (default: "cuda:0")
        prepend_bos: Whether to prepend BOS token (default: False)
        mean_pooling: If True, average embeddings across sequence length to get a single
            vector per sequence. Reduces memory usage significantly (default: False)

    Raises:
        ValueError: If sequences and sequence_ids have different lengths
    """
    if len(sequences) != len(sequence_ids):
        raise ValueError(
            f"Sequences ({len(sequences)}) and sequence_ids ({len(sequence_ids)}) "
            "must have the same length"
        )

    logger.info(
        f"Processing {len(sequences)} sequences in batches of {batch_size} "
        f"on device {device}"
    )
    if mean_pooling:
        logger.info("Mean pooling enabled: embeddings will be averaged across sequence length")

    # Lists to collect all embeddings and IDs
    all_ids: List[str] = []
    all_embeddings: List[np.ndarray] = []

    # Process sequences in batches
    num_batches = (len(sequences) + batch_size - 1) // batch_size

    with tqdm(total=len(sequences), desc="Extracting embeddings") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sequences))

            batch_sequences = sequences[start_idx:end_idx]
            batch_ids = sequence_ids[start_idx:end_idx]

            try:
                # Extract embeddings for the batch
                batch_embeddings, batch_seq_lengths = extract_embeddings_batch(
                    model=model,
                    tokenizer=tokenizer,
                    sequences=batch_sequences,
                    device=device,
                    prepend_bos=prepend_bos,
                )

                # Extract embeddings for each sequence in the batch
                for i, seq_id in enumerate(batch_ids):
                    # Extract embeddings for this specific sequence
                    # Note: embeddings are padded, so we need to extract the actual sequence length
                    seq_length = batch_seq_lengths[i]
                    if prepend_bos:
                        # Skip BOS token if prepended (first position)
                        # Take seq_length tokens after BOS
                        seq_embeddings = batch_embeddings[i, 1 : seq_length + 1]
                    else:
                        # Take seq_length tokens from the start
                        seq_embeddings = batch_embeddings[i, :seq_length]

                    # Apply mean pooling if requested (reduces memory usage)
                    if mean_pooling:
                        # Average across sequence length dimension: (seq_length, embedding_dim) -> (embedding_dim,)
                        seq_embeddings = torch.mean(seq_embeddings, dim=0)

                    # Convert to numpy and store
                    # Convert to float32 first to avoid BFloat16 issues with numpy
                    all_ids.append(seq_id)
                    all_embeddings.append(seq_embeddings.cpu().float().numpy())

                pbar.update(len(batch_sequences))

            except Exception as e:
                logger.error(
                    f"Error processing batch {batch_idx + 1}/{num_batches}: {str(e)}"
                )
                # Continue with next batch
                pbar.update(len(batch_sequences))
                continue

    # Save all embeddings to a single .npz file
    logger.info(f"Saving {len(all_ids)} embeddings to {output_path}...")
    save_embeddings_to_npz(
        ids=all_ids,
        embeddings=all_embeddings,
        output_path=output_path,
    )

    logger.info(f"Completed processing. Embeddings saved to {output_path}")


def main() -> None:
    """
    Main function to orchestrate sequence loading, embedding extraction, and saving.
    """
    parser = argparse.ArgumentParser(
        description="Single-GPU Evo embedding computation from FASTA files"
    )
    parser.add_argument(
        "--fasta_path",
        "-i",
        type=str,
        required=True,
        help="Path to input FASTA file",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default="./embeddings.npz",
        help="Output path for .npz file with embeddings (default: ./embeddings.npz)",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=8,
        help="Batch size for processing sequences (default: 8)",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="evo-1.5-8k-base",
        help="Evo model name to use (default: evo-1.5-8k-base)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        help="Device to run inference on (default: cuda:0)",
    )
    parser.add_argument(
        "--prepend_bos",
        action="store_true",
        help="Prepend BOS token to sequences (default: False)",
    )
    parser.add_argument(
        "--mean_pooling",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Apply mean pooling to embeddings (default: True). Use --no-mean_pooling to disable."
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.fasta_path):
        parser.error(f"Input FASTA file not found: {args.fasta_path}")

    # Validate CUDA availability if using GPU
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(
            f"CUDA is not available but device {args.device} was specified. "
            "Please use 'cpu' or ensure CUDA is properly configured."
        )

    # Validate batch size
    if args.batch_size < 1:
        parser.error(f"Batch size must be at least 1, got {args.batch_size}")

    logger.info(f"Processing file: {args.fasta_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {args.device}")

    try:
        # Load sequences from FASTA file
        sequences, sequence_ids = load_sequences_from_fasta(args.fasta_path)

        # Initialize model and configure for embedding extraction
        model, tokenizer = setup_model_for_embeddings(
            model_name=args.model_name,
            device=args.device,
        )

        # Process sequences and extract embeddings
        process_sequences(
            model=model,
            tokenizer=tokenizer,
            sequences=sequences,
            sequence_ids=sequence_ids,
            output_path=args.output_path,
            batch_size=args.batch_size,
            device=args.device,
            prepend_bos=args.prepend_bos,
            mean_pooling=args.mean_pooling,
        )

        logger.info("Embedding extraction completed successfully")

    except Exception as e:
        logger.error(f"Error during embedding extraction: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

