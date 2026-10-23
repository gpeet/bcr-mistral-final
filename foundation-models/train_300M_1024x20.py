"""
Train a medium BCR language model at its nominal size
(hidden 1024, MLP 4096, 20 layers, 32 heads / 8 KV heads: the large model's design with fewer
layers; 304M parameters).

Same recipe as train_300M.py (data, tokenizer, augmentation, optimizer, schedule, effective batch size,
24 h of training on one MI100); only the architecture differs. train_300M.py itself is 151M parameters
despite its name (see the model table in README.md), so these runs test whether correctly sized models
change Fig. 2B / S1A.

Augmentation:
  - Random cyclic rotation of each sequence at load time
  - Random block masking (0-2 blocks of 20 tokens) applied to the attention mask
"""

from bcrm.seed import seed_everything
from bcrm.paths import data_path, out_path, repo_path
import os
import time
import numpy as np
import torch
import torch.nn.utils as nn_utils
import torch.optim as optim
from transformers import PreTrainedTokenizerFast, MistralConfig, MistralForCausalLM
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.cuda.amp import GradScaler, autocast

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH       = data_path("foundation/diverse_foundation_data.csv")
TOKENIZER_PATH  = repo_path("foundation-models/base_only_bpe_tokenizer/tokenizer.json")
MODEL_SAVE_PATH = out_path("foundation-models/checkpoints/2026-Foundation-300M-1024x20.pth")
LOSS_LOG_PATH   = out_path("foundation-models/logs/training_loss-300M-1024x20.txt")
MAX_STEPS       = int(os.environ.get("BCRM_MAX_STEPS", 0))  # >0 stops early (smoke tests)
MAX_HOURS       = float(os.environ.get("BCRM_MAX_HOURS", 23.5))  # the originals trained for 24 h on one MI100
# ─────────────────────────────────────────────────────────────────────────────


class BCRDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        with open(file_path, "r") as f:
            self.samples = [line.strip() for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Random cyclic rotation
        split = np.random.randint(0, len(sample))
        sample = sample[split:] + sample[:split]

        tokenized = self.tokenizer.encode_plus(
            sample, max_length=512, return_attention_mask=True,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        inputs = tokenized["input_ids"].squeeze(0)
        mask   = tokenized["attention_mask"].squeeze(0)

        # Random block masking
        seq_len    = 512
        n_blocks   = np.random.randint(0, 3)
        block_mask = torch.ones(seq_len)
        block_size = 20
        for _ in range(n_blocks):
            start = np.random.randint(0, max(seq_len - block_size + 1, 1))
            block_mask[start:start + block_size] = 0
        mask = mask * block_mask

        return inputs, mask


seed_everything(42)

# ── Data ─────────────────────────────────────────────────────────────────────
tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH, padding_side="left")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
vocab_size  = len(tokenizer)
dataset     = BCRDataset(DATA_PATH, tokenizer)
# Batches of 64 accumulated 50 times: the same 3,200 sequences per optimizer step as the original
# 300M recipe (128 x 25), which does not fit in an MI100's 32 GB at this model size
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

# ── Model ─────────────────────────────────────────────────────────────────────
configuration = MistralConfig(
    vocab_size=vocab_size, hidden_size=1024, intermediate_size=4096,
    num_hidden_layers=20, num_attention_heads=32, num_key_value_heads=8,
    hidden_act="silu", max_position_embeddings=512, initializer_range=0.02,
    rms_norm_eps=1e-06, use_cache=True, pad_token_id=3,
    bos_token_id=None, eos_token_id=None, rope_theta=10000.0, sliding_window=512,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MistralForCausalLM(configuration)
scaler = GradScaler()
model.to(device)
model.gradient_checkpointing_enable()

# ── Optimizer & scheduler ────────────────────────────────────────────────────
optimizer       = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, betas=(0.9, 0.95))
total_steps     = 100000 // 50   # same 2,000 optimizer steps as the original
warmup_steps    = total_steps // 30
cosine_steps    = total_steps - warmup_steps
warmup_scheduler  = LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_steps)
cosine_scheduler  = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-7)
scheduler         = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

# ── Training loop ────────────────────────────────────────────────────────────
accumulation_steps = 50
step_count         = 1
open(LOSS_LOG_PATH, "w").close()   # fresh log: BCRM_OUT may hold the published one (seeds/)
start_time         = time.time()
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
        out_of_time = time.time() - start_time > MAX_HOURS * 3600
        if (MAX_STEPS and step_count > MAX_STEPS) or out_of_time:
            break

        if (i + 1) % 100 == 0:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    if (MAX_STEPS and step_count > MAX_STEPS) or out_of_time:
        break

torch.save(model.state_dict(), MODEL_SAVE_PATH)
