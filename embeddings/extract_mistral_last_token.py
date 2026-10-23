"""
Extract last-token hidden-state embeddings from the 500M BCR-Mistral model (487M parameters; resumable).

In a decoder-only model the last real token carries the most contextual information.
Sequences are padded on the right, so it is located from the attention mask.
Embeddings are written row-by-row to avoid holding the full matrix in RAM.

If the output CSV already exists and contains rows, the script resumes from
where it left off — sequences already embedded are skipped and new rows are
appended.

Usage:
    python extract_mistral_last_token.py \\
        --input    /path/to/triplet_embedding_test_set_igblast.csv \\
        --output   /path/to/mistral-embeddings-last-token.csv \\
        --model-hf /path/to/2025-HL-Foundation-1024x4096-HF \\
        --weights  /path/to/2025-HL-Foundation-1024x4096-gradClip-long-cp48h.pth \\
        --tokenizer /path/to/base_only_bpe_tokenizer/tokenizer.json \\
        --sequence-col sequence
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
    parser.add_argument("--input",        required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--model-hf",     required=True, help="Path to HuggingFace model directory")
    parser.add_argument("--weights",      required=True, help="Path to .pth state-dict file")
    parser.add_argument("--tokenizer",    required=True, help="Path to tokenizer.json")
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--batch-size",   type=int, default=96)
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


def main():
    args      = parse_args()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # ── Resume support ─────────────────────────────────────────────────────────
    rows_done = 0
    output_path = Path(args.output)
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
    open_mode    = "a" if rows_done > 0 else "w"

    header = ["index"] + [f"hidden_state_{i}" for i in range(model.config.hidden_size)]
    if write_header:
        with open(args.output, "w", newline="") as f_out:
            csv.writer(f_out).writerow(header)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding",
                          initial=rows_done, total=total_rows):
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            indices        = batch["index"]

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            hidden_states      = outputs.hidden_states[-1]
            last_token_indices = attention_mask.sum(dim=1) - 1
            hidden_vecs        = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices].cpu().tolist()

            with open(args.output, "a", newline="") as f_out:
                writer = csv.writer(f_out)
                for i, row in zip(indices, hidden_vecs):
                    writer.writerow([i] + row)


if __name__ == "__main__":
    main()
