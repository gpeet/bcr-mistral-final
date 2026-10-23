"""
Train the large ("500M") architecture on CDR3 junction sequences only: the junction model behind the junction classifiers.

Uses the same architecture as train_500M.py but with max_position_embeddings=96
and no cyclic rotation augmentation (junctions are short nucleotide sequences).
Input data is a flat CSV of junction sequences.
"""

from bcrm.paths import data_path, out_path, repo_path
import os
import numpy as np
import torch
import torch.nn.utils as nn_utils
import torch.optim as optim
from transformers import PreTrainedTokenizerFast, MistralConfig, MistralForCausalLM
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.cuda.amp import GradScaler, autocast

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH       = data_path("foundation/junction.csv")
TOKENIZER_PATH  = repo_path("foundation-models/base_only_bpe_tokenizer/tokenizer.json")
MODEL_SAVE_PATH = out_path("foundation-models/checkpoints/2025-HL-Foundation-1024x4096-gradClip-junction.pth")
LOSS_LOG_PATH   = out_path("foundation-models/logs/training_loss-1024x4096-gradClip-junction.txt")
MAX_STEPS       = int(os.environ.get("BCRM_MAX_STEPS", 0))  # >0 stops early (smoke tests)
# ─────────────────────────────────────────────────────────────────────────────

MAX_LEN = 96  # junction sequences are short


class BCRDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        with open(file_path, "r") as f:
            self.samples = [line.strip() for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample    = self.samples[idx]
        tokenized = self.tokenizer.encode_plus(
            sample, max_length=MAX_LEN, return_attention_mask=True,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return tokenized["input_ids"].squeeze(0), tokenized["attention_mask"].squeeze(0)


tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH, padding_side="left")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
vocab_size  = len(tokenizer)
dataset     = BCRDataset(DATA_PATH, tokenizer)
data_loader = DataLoader(dataset, batch_size=850, shuffle=True, num_workers=8, pin_memory=True)

configuration = MistralConfig(
    vocab_size=vocab_size, hidden_size=1024, intermediate_size=4096,
    num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
    hidden_act="silu", max_position_embeddings=MAX_LEN, initializer_range=0.02,
    rms_norm_eps=1e-06, use_cache=True, pad_token_id=3,
    bos_token_id=None, eos_token_id=None, rope_theta=10000.0, sliding_window=MAX_LEN,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MistralForCausalLM(configuration)
scaler = GradScaler()
model.to(device)
model.gradient_checkpointing_enable()

optimizer       = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, betas=(0.9, 0.95))
total_steps     = 14000 // 3
warmup_steps    = 100
cosine_steps    = total_steps - warmup_steps
warmup_scheduler  = LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps)
cosine_scheduler  = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-7)
scheduler         = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

accumulation_steps = 3
step_count         = 1
open(LOSS_LOG_PATH, "w").close()   # fresh log: BCRM_OUT may hold the published one (seeds/)
total_loss         = 0.0
num_epochs         = 100
print("Total batches per epoch:", len(data_loader))

for epoch in range(num_epochs):
    for i, (inputs, mask) in enumerate(data_loader):
        input_ids      = inputs.to(device, non_blocking=True)
        attention_mask = mask.to(device, non_blocking=True)
        if attention_mask.dtype != torch.long:
            attention_mask = attention_mask.long()

        with autocast():
            labels = input_ids.masked_fill(attention_mask == 0, -100)
            loss   = model(input_ids, attention_mask, labels=labels)["loss"]

        loss        = loss / accumulation_steps
        total_loss += loss.item()
        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            print(f"Step {step_count}, Loss: {total_loss}")
            with open(LOSS_LOG_PATH, "a") as f:
                f.write(f"Step {step_count}, Loss: {total_loss}\n")
            total_loss = 0.0

        step_count += 1
        if MAX_STEPS and step_count > MAX_STEPS:
            break

        if (i + 1) % 100 == 0:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    if MAX_STEPS and step_count > MAX_STEPS:
        break

torch.save(model.state_dict(), MODEL_SAVE_PATH)
