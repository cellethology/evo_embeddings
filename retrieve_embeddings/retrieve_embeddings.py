import argparse
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
from evo.scoring import logits_to_logprobs, prepare_batch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evo import Evo
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


class SequenceDataset(Dataset):
    """Simple dataset to wrap a list of sequences for DataLoader."""

    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


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


def process_sequences(
    model: torch.nn.Module,
    tokenizer: object,
    sequences: List[str],
    sequence_ids: List[str],
    output_path: str,
    batch_size: int = 8,
    device: str = "cuda:0",
    prepend_bos: bool = True,
    max_seq_length: int = None,
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
        max_seq_length: Maximum sequence length to pad to (default: None)
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

    dataset = SequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Lists to collect all embeddings and IDs
    all_embeddings: List[np.ndarray] = []

    with torch.inference_mode():
        for batch_sequences in tqdm(
            dataloader, desc="Extracting Features", unit="batch"
        ):
            input_ids, _ = prepare_batch(
                seqs=batch_sequences,
                tokenizer=tokenizer,
                prepend_bos=prepend_bos,
                device=device,
                max_seq_length=max_seq_length,
            )
            try:
                # The Evo model forward pass returns (logits, hidden_states)
                while True:
                    try:
                        logits, _ = model(input_ids)
                    except torch.cuda.OutOfMemoryError:
                        print("CUDA Out of Memory during model inference. Retrying...")
                        torch.cuda.empty_cache()
                        continue
                    break

                sequence_embeddings = logits_to_logprobs(
                    logits, input_ids, trim_bos=True
                )

                all_embeddings.append(
                    sequence_embeddings.to(dtype=torch.float32).cpu().numpy()
                )

            except torch.cuda.OutOfMemoryError:
                print("CUDA Out of Memory on a batch. Skipping batch.")
                torch.cuda.empty_cache()
                continue

    # check all_embeddings is same length as sequence_ids
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    if all_embeddings.shape[0] != len(sequence_ids):
        raise ValueError(
            f"All embeddings ({all_embeddings.shape[0]}) and sequence_ids ({len(sequence_ids)}) "
            "must have the same length"
        )

    # Save all embeddings to a single .npz file
    logger.info(f"Saving {len(sequence_ids)} embeddings to {output_path}...")
    save_embeddings_to_npz(
        ids=sequence_ids,
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

    logger.info(f"Processing file: {args.fasta_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {args.device}")

    try:
        # Load sequences from FASTA file
        sequences, sequence_ids, max_seq_length = load_sequences_from_fasta(
            args.fasta_path
        )

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
            max_seq_length=max_seq_length,
            output_path=args.output_path,
            batch_size=args.batch_size,
            device=args.device,
            prepend_bos=args.prepend_bos,
        )

        logger.info("Embedding extraction completed successfully")

    except Exception as e:
        logger.error(f"Error during embedding extraction: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
