"""
Select a diverse subset of BCR sequences from a large training file.

Diversity is measured using 7-mer count vectors (via ProkBERT tokenization).
Each batch is compared against a rolling pool of previously selected sequences;
the top-K most distant members of each batch are retained.

Usage:
    python 05_select_diverse_sequences.py \\
        --input  /path/to/shuffled_sequences.csv \\
        --output /path/to/diverse_sequences.csv \\
        --n-sequences 13252849 \\
        --batch-size 4096 \\
        --top-k 512 \\
        --pool-cap 100000
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from prokbert.prokbert_tokenizer import ProkBERTTokenizer


# ── Dataset ──────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """Streams sequences from a flat text file in fixed-size chunks."""

    def __init__(self, file_path, tokenizer, chunk_size=4096, approx_line_count=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.vocab)
        self.chunk_size = chunk_size
        self.line_count = approx_line_count if approx_line_count is not None else 10 ** 9

        self.current_chunk = []
        self.current_chunk_start_idx = -1

    def __len__(self):
        return self.line_count

    def _load_chunk(self, start_idx):
        self.current_chunk = []
        self.current_chunk_start_idx = start_idx
        with open(self.file_path, "r") as fh:
            for _ in range(start_idx):
                fh.readline()
            for _ in range(self.chunk_size):
                line = fh.readline()
                if not line:
                    break
                self.current_chunk.append(line.strip())

    def __getitem__(self, idx):
        if not (self.current_chunk_start_idx <= idx < self.current_chunk_start_idx + len(self.current_chunk)):
            self._load_chunk(idx)

        sample = self.current_chunk[idx - self.current_chunk_start_idx]

        tokenized = self.tokenizer.encode_plus(
            sample, max_length=512, padding="max_length", truncation=True, return_tensors="pt"
        )
        inputs = tokenized["input_ids"].squeeze(0)
        inputs = F.pad(inputs, (0, 512))
        inputs = torch.narrow(inputs, 0, 0, 512)

        count_vector = torch.bincount(inputs, minlength=self.vocab_size).float()
        return count_vector, sample


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",        required=True,  help="Path to shuffled input sequences (.csv, one sequence per row)")
    parser.add_argument("--output",       required=True,  help="Path for selected diverse sequences output (.csv)")
    parser.add_argument("--n-sequences",  type=int,   default=13_252_849, help="Approximate number of sequences in input (default: 13252849)")
    parser.add_argument("--batch-size",   type=int,   default=4096,       help="DataLoader batch size (default: 4096)")
    parser.add_argument("--top-k",        type=int,   default=512,        help="Sequences selected per batch (default: 512)")
    parser.add_argument("--pool-cap",     type=int,   default=100_000,    help="Max comparator pool size (default: 100000)")
    parser.add_argument("--num-workers",  type=int,   default=10,         help="DataLoader worker processes (default: 10)")
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = ProkBERTTokenizer(
        tokenization_params={"kmer": 7, "shift": 1},
        operation_space="sequence",
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    dataset = SequenceDataset(
        args.input, tokenizer,
        chunk_size=args.batch_size,
        approx_line_count=args.n_sequences,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    comparator_pool = None
    diverse_sequences = []

    for i, (count_matrix, samples) in enumerate(loader):
        count_matrix = count_matrix.to(device)

        if comparator_pool is None:
            comparator_pool = count_matrix.clone()
        else:
            comparator_pool = torch.cat((comparator_pool, selected_vectors), dim=0)
            if comparator_pool.size(0) > args.pool_cap:
                idx = torch.randint(0, comparator_pool.size(0), (args.pool_cap,))
                comparator_pool = comparator_pool[idx]

        sample_idx = torch.randint(0, comparator_pool.size(0), (args.top_k,))
        comparators = comparator_pool[sample_idx]

        dist_matrix = torch.cdist(count_matrix, comparators, p=2)
        total_distances = torch.sum(dist_matrix, dim=1)
        top_idx = torch.topk(total_distances, args.top_k)[1]

        selected_vectors = count_matrix[top_idx]
        diverse_sequences.extend([samples[k] for k in top_idx.cpu().numpy()])

        if i % 100 == 0:
            print(f"{i}/{len(loader)} batches processed")

    pd.DataFrame(diverse_sequences).to_csv(args.output, index=False)
    print(f"Done. {len(diverse_sequences)} sequences written to {args.output}")


if __name__ == "__main__":
    main()
