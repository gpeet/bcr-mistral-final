from bcrm.paths import data_path, model_path, out_path, repo_path
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, MistralForCausalLM
import pandas as pd
from tqdm import tqdm
import csv
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
TOKENIZER_PATH = repo_path("foundation-models/base_only_bpe_tokenizer/tokenizer.json")
MODEL_HF_PATH  = model_path("2025-HL-Foundation-1024x4096-HF")
WEIGHTS_PATH   = model_path("2025-HL-Foundation-1024x4096-gradClip-long-cp48h.pth")

MAX_SEQ_LEN    = 512
SEQUENCE_COL   = "sequence"
BATCH_SIZE     = 96
NUM_WORKERS    = 8

# --variant contig (default): the full TRUST4 contigs behind the published Fig. 3B.
# --variant vj: the same sequences clipped to V-J like the classifier training data
# (dataset/09_treatment_vj.R); written to separately named files.
parser = argparse.ArgumentParser(description="Embed the pre/post treatment sequences")
parser.add_argument("--variant", choices=["contig", "vj"], default="contig")
VARIANT = parser.parse_args().variant
SUFFIX  = "" if VARIANT == "contig" else f"_{VARIANT}"

JOBS = [
    {
        "input_csv":  data_path(f"classifier/treatment_sequences{SUFFIX}_train.csv"),
        "output_csv": out_path(f"embeddings/treatment_embeddings{SUFFIX}_train.csv"),
    },
    {
        "input_csv":  data_path(f"classifier/treatment_sequences{SUFFIX}_test.csv"),
        "output_csv": out_path(f"embeddings/treatment_embeddings{SUFFIX}_test.csv"),
    },
]
# ───────────────────────────────────────────────────────────────────────────────


class SequenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, sequence_col, start_offset=0):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.sequence_col = sequence_col
        self.start_offset = start_offset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq = self.df.iloc[idx][self.sequence_col]

        if pd.isna(seq) or str(seq).strip() == '':
            seq = 'A'

        encoded = self.tokenizer.encode_plus(
            seq, max_length=MAX_SEQ_LEN, return_attention_mask=True,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'index': self.start_offset + idx
        }


def extract_embeddings(model, tokenizer, input_csv, output_csv,
                       sequence_col, batch_size=32, num_workers=4):
    """Extract last-token hidden states from the foundation model for treatment sequences.  Resumes automatically if interrupted."""
    df = pd.read_csv(input_csv)
    total_rows = len(df)

    # ── Resume support ────────────────────────────────────────────────────────
    rows_done = 0
    output_path = Path(output_csv)
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
    dataset = SequenceDataset(df_remaining, tokenizer, sequence_col,
                              start_offset=rows_done)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True,
        num_workers=num_workers, shuffle=False
    )

    write_header = (rows_done == 0)
    open_mode    = "a" if rows_done > 0 else "w"

    model.eval()
    model.cuda()
    hidden_size = model.config.hidden_size

    with open(output_csv, open_mode, newline="") as f_out:
        writer = csv.writer(f_out)

        if write_header:
            writer.writerow(["index"] + [f"hidden_state_{i}" for i in range(hidden_size)])

        with torch.no_grad():
            n_remaining = total_rows - rows_done
            n_batches = (n_remaining + batch_size - 1) // batch_size
            for batch in tqdm(dataloader, desc="Embedding", total=n_batches):
                input_ids = batch['input_ids'].cuda(non_blocking=True)
                attention_mask = batch['attention_mask'].cuda(non_blocking=True)
                indices = batch['index']

                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)

                # Final layer, last token (correct for left-padded sequences)
                hidden_vecs = outputs.hidden_states[-1][:, -1, :].cpu().tolist()

                for i, hidden_row in zip(indices, hidden_vecs):
                    writer.writerow([i.item()] + hidden_row)


tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=TOKENIZER_PATH, padding_side='left'
)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = MistralForCausalLM.from_pretrained(
    MODEL_HF_PATH, max_position_embeddings=MAX_SEQ_LEN, sliding_window=MAX_SEQ_LEN
)
model.load_state_dict(torch.load(WEIGHTS_PATH))

for job_i, job in enumerate(JOBS):
    print(f"\n{'='*70}")
    print(f"JOB {job_i+1}/{len(JOBS)}: {Path(job['input_csv']).name}")
    print(f"  -> {Path(job['output_csv']).name}")
    print(f"{'='*70}\n")

    extract_embeddings(
        model=model,
        tokenizer=tokenizer,
        input_csv=job["input_csv"],
        output_csv=job["output_csv"],
        sequence_col=SEQUENCE_COL,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

print("\nAll jobs complete.")
