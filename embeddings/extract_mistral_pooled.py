"""
Extract mean-, max-, and weighted-mean-pooled embeddings from the 500M BCR-Mistral model (487M parameters; resumable).

Three output CSVs are written simultaneously (one per pooling strategy).
Weighted mean pooling assigns linearly increasing weights to token positions
(1, 2, ..., seq_len), downweighting early padding.

If output CSVs already exist and contain rows, the script resumes from where
it left off — sequences already embedded are skipped and new rows are appended.

Usage:
    python extract_mistral_pooled.py \\
        --input          /path/to/triplet_embedding_test_set_igblast.csv \\
        --out-mean        /path/to/mistral-meanpooled-embeddings.csv \\
        --out-max         /path/to/mistral-maxpooled-embeddings.csv \\
        --out-weighted    /path/to/mistral-weightedpooled-embeddings.csv \\
        --model-hf        /path/to/2025-HL-Foundation-1024x4096-HF \\
        --weights         /path/to/2025-HL-Foundation-1024x4096-gradClip-long-cp48h.pth \\
        --tokenizer       /path/to/base_only_bpe_tokenizer/tokenizer.json \\
        --sequence-col    sequence
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, MistralForCausalLM
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",         required=True)
    parser.add_argument("--out-mean",      required=True)
    parser.add_argument("--out-max",       required=True)
    parser.add_argument("--out-weighted",  required=True)
    parser.add_argument("--model-hf",      required=True)
    parser.add_argument("--weights",       required=True)
    parser.add_argument("--tokenizer",     required=True)
    parser.add_argument("--sequence-col",  default="sequence")
    parser.add_argument("--batch-size",    type=int, default=96)
    return parser.parse_args()


class SequenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, sequence_col, start_offset=0):
        self.df           = dataframe.reset_index(drop=True)
        self.tokenizer    = tokenizer
        self.sequence_col = sequence_col
        self.start_offset = start_offset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq     = self.df.iloc[idx][self.sequence_col]
        encoded = self.tokenizer.encode_plus(
            seq, max_length=512, return_attention_mask=True,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "index":          self.start_offset + idx,
        }


def weighted_mean_pooling(hidden_states, attention_mask):
    """Position-weighted mean pool: weights are 1, 2, ..., seq_len (masked for padding)."""
    batch_size, seq_len, _ = hidden_states.shape
    position_weights = torch.arange(1, seq_len + 1, device=hidden_states.device).unsqueeze(0)
    weights          = attention_mask * position_weights
    weighted_sum     = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
    denom            = torch.sum(weights, dim=-1, keepdim=True).clamp(min=1e-6)
    return weighted_sum / denom


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer, padding_side="right")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = MistralForCausalLM.from_pretrained(
        args.model_hf, max_position_embeddings=512, sliding_window=512
    )
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    df         = pd.read_csv(args.input)
    total_rows = len(df)

    # ── Resume support (use out-mean as reference; all 3 are written in lockstep) ─
    rows_done = 0
    output_path = Path(args.out_mean)
    if output_path.exists():
        with open(output_path, "r") as f:
            rows_done = sum(1 for _ in f) - 1  # subtract header
        rows_done = max(0, rows_done)
        if rows_done > 0:
            print(f"Resume: {rows_done:,} / {total_rows:,} rows already written.")
        if rows_done >= total_rows:
            print("All rows already embedded. Nothing to do.")
            return

    df_remaining = df.iloc[rows_done:]
    dataset    = SequenceDataset(df_remaining, tokenizer, args.sequence_col,
                                 start_offset=rows_done)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    write_header = (rows_done == 0)
    header = ["index"] + [f"hidden_state_{i}" for i in range(model.config.hidden_size)]
    if write_header:
        for path in [args.out_mean, args.out_max, args.out_weighted]:
            with open(path, "w", newline="") as f_out:
                csv.writer(f_out).writerow(header)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding",
                          initial=rows_done, total=total_rows):
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            indices        = batch["index"]

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            hidden_states = outputs.hidden_states[-1]
            mask          = attention_mask.unsqueeze(-1)

            # Mean pooling
            mean_pooled = ((hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).cpu().tolist()

            # Max pooling
            max_pooled = hidden_states.masked_fill(mask == 0, float("-inf")).max(dim=1).values.cpu().tolist()

            # Weighted mean pooling
            weighted_pooled = weighted_mean_pooling(hidden_states, attention_mask).cpu().tolist()

            with open(args.out_mean,     "a", newline="") as f_mean, \
                 open(args.out_max,      "a", newline="") as f_max,  \
                 open(args.out_weighted, "a", newline="") as f_weighted:
                w_mean, w_max, w_wt = csv.writer(f_mean), csv.writer(f_max), csv.writer(f_weighted)
                for i, mean_row, max_row, wt_row in zip(indices, mean_pooled, max_pooled, weighted_pooled):
                    w_mean.writerow([i] + mean_row)
                    w_max.writerow([i]  + max_row)
                    w_wt.writerow([i]   + wt_row)


if __name__ == "__main__":
    main()
