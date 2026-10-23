"""
Evaluate per-sequence language modelling loss for all BCR-Mistral model variants.

For each model, scores 5000 validation sequences (one random sample, seed 42, or each
model's sample recorded in --indices-from) with the causal LM loss on both the full sequence
and the CDR3 junction alone. Model names are checkpoint labels; the figures label each model by the parameter count of its
config: model_256x1024 = 50M, model_512x2048 = 151M, model_1024x4096 = 487M (its 48 h checkpoint,
the weights used throughout the paper). The 2026-Foundation-* checkpoints (model_768x11 = 104M,
model_1024x20 = 304M) back a supplementary size check, not a published panel.

Usage:
    python evaluate.py \\
        --eval-csv   /path/to/classifier_data_igh_sequence-val.csv \\
        --output-csv /path/to/foundation_models_eval_losses.csv \\
        --tokenizer  /path/to/base_only_bpe_tokenizer/tokenizer.json \\
        --model-dir  /path/to/models/ \\
        [--indices-from seeds/foundation-models/foundation_models_eval_losses.csv]
"""

import argparse
import gc

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import PreTrainedTokenizerFast, MistralForCausalLM, MistralConfig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-csv",   required=True, help="Validation CSV with 'sequence' and 'junction' columns")
    parser.add_argument("--output-csv", required=True, help="Output path for per-sample loss results")
    parser.add_argument("--tokenizer",  required=True, help="Path to tokenizer.json")
    parser.add_argument("--model-dir",  required=True, help="Directory containing .pth model checkpoints")
    parser.add_argument("--n-samples",  type=int, default=5000, help="Sequences to evaluate per model (default: 5000)")
    parser.add_argument("--indices-from", help="Earlier output CSV: re-score exactly the sequences it sampled")
    parser.add_argument("--models", nargs="+", help="Only these models (default: all)")
    return parser.parse_args()


# ── Dataset ──────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, csv_path, text_col, tokenizer):
        self.data    = pd.read_csv(csv_path)
        self.samples = self.data[text_col].astype(str).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokenized = self.tokenizer.encode_plus(
            self.samples[idx], max_length=512, truncation=True,
            padding="max_length", return_attention_mask=True, return_tensors="pt"
        )
        return (
            tokenized["input_ids"].squeeze(0),
            tokenized["attention_mask"].squeeze(0),
        )


# ── Loss ─────────────────────────────────────────────────────────────────────

def compute_loss(model, input_ids, attention_mask):
    with torch.no_grad():
        labels  = input_ids.masked_fill(attention_mask == 0, -100)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss.item()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer, padding_side="left")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    import os
    model_configs = {
        "model_256x1024": {
            "path": os.path.join(args.model_dir, "2025-HL-Foundation-100M-gradClip.pth"),
            "config": MistralConfig(
                vocab_size=len(tokenizer), hidden_size=512, intermediate_size=2048,
                num_hidden_layers=12, num_attention_heads=8, num_key_value_heads=8,
                hidden_act="silu", max_position_embeddings=512, initializer_range=0.02,
                rms_norm_eps=1e-06, use_cache=True, pad_token_id=3,
                bos_token_id=None, eos_token_id=None, rope_theta=10000.0, sliding_window=512,
            ),
        },
        "model_512x2048": {
            "path": os.path.join(args.model_dir, "2025-HL-Foundation-300M-gradClip.pth"),
            "config": MistralConfig(
                vocab_size=len(tokenizer), hidden_size=768, intermediate_size=3072,
                num_hidden_layers=16, num_attention_heads=12, num_key_value_heads=12,
                hidden_act="silu", max_position_embeddings=512, initializer_range=0.02,
                rms_norm_eps=1e-06, use_cache=True, pad_token_id=3,
                bos_token_id=None, eos_token_id=None, rope_theta=10000.0, sliding_window=512,
            ),
        },
        "model_512x2048-undiversified": {
            "path": os.path.join(args.model_dir, "2025-HL-Foundation-300M-gradClip-undiversified.pth"),
            "config": MistralConfig(
                vocab_size=len(tokenizer), hidden_size=768, intermediate_size=3072,
                num_hidden_layers=16, num_attention_heads=12, num_key_value_heads=12,
                hidden_act="silu", max_position_embeddings=512, initializer_range=0.02,
                rms_norm_eps=1e-06, use_cache=True, pad_token_id=3,
                bos_token_id=None, eos_token_id=None, rope_theta=10000.0, sliding_window=512,
            ),
        },
        "model_1024x4096": {
            "path": os.path.join(args.model_dir, "2025-HL-Foundation-1024x4096-gradClip-long-cp48h.pth"),  # 48 h checkpoint
            "config": MistralConfig(
                vocab_size=len(tokenizer), hidden_size=1024, intermediate_size=4096,
                num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8,
                hidden_act="silu", max_position_embeddings=512, initializer_range=0.02,
                rms_norm_eps=1e-06, use_cache=True, pad_token_id=3,
                bos_token_id=None, eos_token_id=None, rope_theta=10000.0, sliding_window=512,
            ),
        },
        # Correctly sized 104M / 304M models (train_*_768x11.py, train_*_1024x20.py); skipped if not trained
        "model_768x11": {
            "path": os.path.join(args.model_dir, "2026-Foundation-100M-768x11.pth"),
            "config": MistralConfig(
                vocab_size=len(tokenizer), hidden_size=768, intermediate_size=3072,
                num_hidden_layers=11, num_attention_heads=12, num_key_value_heads=12,
                hidden_act="silu", max_position_embeddings=512, initializer_range=0.02,
                rms_norm_eps=1e-06, use_cache=True, pad_token_id=3,
                bos_token_id=None, eos_token_id=None, rope_theta=10000.0, sliding_window=512,
            ),
        },
        "model_1024x20": {
            "path": os.path.join(args.model_dir, "2026-Foundation-300M-1024x20.pth"),
            "config": MistralConfig(
                vocab_size=len(tokenizer), hidden_size=1024, intermediate_size=4096,
                num_hidden_layers=20, num_attention_heads=32, num_key_value_heads=8,
                hidden_act="silu", max_position_embeddings=512, initializer_range=0.02,
                rms_norm_eps=1e-06, use_cache=True, pad_token_id=3,
                bos_token_id=None, eos_token_id=None, rope_theta=10000.0, sliding_window=512,
            ),
        },
        "model_1024x20-undiversified": {
            "path": os.path.join(args.model_dir, "2026-Foundation-300M-1024x20-undiversified.pth"),
            "config": MistralConfig(
                vocab_size=len(tokenizer), hidden_size=1024, intermediate_size=4096,
                num_hidden_layers=20, num_attention_heads=32, num_key_value_heads=8,
                hidden_act="silu", max_position_embeddings=512, initializer_range=0.02,
                rms_norm_eps=1e-06, use_cache=True, pad_token_id=3,
                bos_token_id=None, eos_token_id=None, rope_theta=10000.0, sliding_window=512,
            ),
        },
    }

    # Sequences to score: each model's sample recorded in --indices-from (the published file drew
    # a separate sample per model), otherwise one seeded sample shared by all models
    ref_dataset     = SequenceDataset(args.eval_csv, "junction", tokenizer)
    sampled_indices = np.random.default_rng(42).choice(len(ref_dataset), size=args.n_samples, replace=False)
    recorded = {}
    if args.indices_from:
        prev = pd.read_csv(args.indices_from)
        prev = prev[prev["text_col"] == "junction"]
        recorded = {m: g["sample_idx"].to_numpy() for m, g in prev.groupby("model", sort=False)}
    if args.models:
        model_configs = {k: v for k, v in model_configs.items() if k in args.models}
    for name in [k for k, v in model_configs.items() if not os.path.exists(v["path"])]:
        print(f"Skipping {name}: {model_configs.pop(name)['path']} not found")
    # A correctly sized model is scored on the sequences of the model it would replace
    SAMPLE_OF = {"model_768x11": "model_256x1024", "model_1024x20": "model_512x2048",
                 "model_1024x20-undiversified": "model_512x2048-undiversified"}

    results = []

    for model_name, model_info in model_configs.items():
        print(f"\n=== {model_name} ===")
        model_indices = recorded.get(model_name, recorded.get(SAMPLE_OF.get(model_name), sampled_indices))
        model = MistralForCausalLM(model_info["config"])
        model.load_state_dict(torch.load(model_info["path"], map_location=device))
        model.to(device)
        model.eval()

        for text_col in ["sequence", "junction"]:
            print(f"  evaluating: {text_col}")
            dataset = SequenceDataset(args.eval_csv, text_col, tokenizer)
            subset  = Subset(dataset, model_indices)
            loader  = DataLoader(subset, batch_size=1, shuffle=False)

            for i, (input_ids, attention_mask) in enumerate(tqdm(loader, desc=f"{model_name} | {text_col}")):
                input_ids      = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                loss = compute_loss(model, input_ids, attention_mask)
                results.append({
                    "model":      model_name,
                    "text_col":   text_col,
                    "sample_idx": int(model_indices[i]),
                    "loss":       loss,
                })

        del model
        torch.cuda.empty_cache()
        gc.collect()

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"\nSaved results to: {args.output_csv}")


if __name__ == "__main__":
    main()
