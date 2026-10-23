"""
Extract foundation-model embeddings for Owens holdout BCR junction sequences.

Uses the junction foundation model (shorter context, junction column) rather than
the full-sequence model used in extract_embeddings.py.
"""

from bcrm.paths import data_path, model_path, out_path, repo_path
import csv

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, MistralForCausalLM
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
TOKENIZER_PATH = repo_path("foundation-models/base_only_bpe_tokenizer/tokenizer.json")
MODEL_HF_PATH  = model_path("2025-HL-Foundation-1024x4096-HF")
WEIGHTS_PATH   = model_path("2025-HL-Foundation-1024x4096-gradClip-junction.pth")
INPUT_CSV      = data_path("owens/owens_sequences.csv")
OUTPUT_CSV     = out_path("owens/owens_junction_embeddings.csv")
SEQ_COL        = "junction"
MAX_SEQ_LEN    = 96
# ──────────────────────────────────────────────────────────────────────────────


class SequenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, sequence_col):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.sequence_col = sequence_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.iloc[idx][self.sequence_col]
        if pd.isna(seq) or str(seq).strip() == "":
            seq = "A"

        encoded = self.tokenizer.encode_plus(
            seq,
            max_length=MAX_SEQ_LEN,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "index":          idx,
        }


def extract_embeddings(model, tokenizer, input_csv, output_csv,
                       sequence_col, batch_size=256, num_workers=8):
    df = pd.read_csv(input_csv)
    dataset = SequenceDataset(df, tokenizer, sequence_col)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True,
                            num_workers=num_workers, shuffle=False)

    model.eval()
    model.cuda()
    hidden_size = model.config.hidden_size

    with open(output_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["index"] + [f"hidden_state_{i}" for i in range(hidden_size)])

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Embedding"):
                input_ids      = batch["input_ids"].cuda(non_blocking=True)
                attention_mask = batch["attention_mask"].cuda(non_blocking=True)
                indices        = batch["index"]

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                hidden_vecs = outputs.hidden_states[-1][:, -1, :].cpu().tolist()

                for i, hidden_row in zip(indices, hidden_vecs):
                    writer.writerow([i.item()] + hidden_row)

    print(f"Embeddings written to {output_csv} ({len(df)} rows)")


tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_PATH, padding_side="left"
)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

model = MistralForCausalLM.from_pretrained(
    MODEL_HF_PATH, max_position_embeddings=MAX_SEQ_LEN, sliding_window=MAX_SEQ_LEN
)
model.load_state_dict(torch.load(WEIGHTS_PATH))

extract_embeddings(
    model=model,
    tokenizer=tokenizer,
    input_csv=INPUT_CSV,
    output_csv=OUTPUT_CSV,
    sequence_col=SEQ_COL,
    batch_size=256,
    num_workers=8,
)
