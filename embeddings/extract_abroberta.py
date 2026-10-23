"""
Extract mean-pooled embeddings from Ab-RoBERTa (mogam-ai/Ab-RoBERTa).

Used as a baseline for comparison against BCR-Mistral embeddings.

Usage:
    python extract_abroberta.py \\
        --input        /path/to/triplet_embedding_test_set_igblast.csv \\
        --output       /path/to/abroberta-meanpooled-embeddings.csv \\
        --sequence-col sequence_aa \\
        --cache-dir    /path/to/hf_cache
"""

import argparse
import csv

import torch
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",        required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--sequence-col", default="sequence_aa")
    parser.add_argument("--batch-size",   type=int, default=96)
    parser.add_argument("--cache-dir",    default=None, help="HuggingFace model cache directory")
    return parser.parse_args()


class SequenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, sequence_col):
        self.df           = dataframe
        self.tokenizer    = tokenizer
        self.sequence_col = sequence_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq     = self.df.iloc[idx][self.sequence_col]
        encoded = self.tokenizer.encode_plus(
            seq, max_length=152, return_attention_mask=True,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "index":          idx,
        }


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "mogam-ai/Ab-RoBERTa"
    tokenizer  = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    model      = AutoModel.from_pretrained(model_name, cache_dir=args.cache_dir)
    model.to(device)
    model.eval()

    df         = pd.read_csv(args.input)
    dataset    = SequenceDataset(df, tokenizer, args.sequence_col)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    header = ["index"] + [f"hidden_state_{i}" for i in range(model.config.hidden_size)]
    with open(args.output, "w", newline="") as f_out:
        csv.writer(f_out).writerow(header)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            indices        = batch["index"]

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            hidden_states = outputs.hidden_states[-1]
            mask          = attention_mask.unsqueeze(-1)
            mean_pooled   = ((hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).cpu().tolist()

            with open(args.output, "a", newline="") as f_out:
                writer = csv.writer(f_out)
                for i, row in zip(indices, mean_pooled):
                    writer.writerow([i] + row)


if __name__ == "__main__":
    main()
