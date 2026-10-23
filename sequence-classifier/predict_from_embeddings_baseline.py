"""
Fold-aware ensemble prediction for pre/post treatment sequences (baseline classifier).

Applies the correct fold's models to each subset to avoid leakage:
  - Train-fold BioProjects → flipped models (trained on test, never saw these patients)
  - Test-fold BioProjects  → unflipped models (trained on train, never saw these patients)

Combines both subsets into a single output CSV.
"""

from bcrm.paths import data_path, model_path, out_path, repo_path
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────
# --variant contig (default): full TRUST4 contigs (published Fig. 3B); --variant vj: V-J clipped
# sequences (dataset/09_treatment_vj.R), read from and written to separately named files
parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
parser.add_argument("--variant", choices=["contig", "vj"], default="contig")
VARIANT = parser.parse_args().variant
SUFFIX  = "" if VARIANT == "contig" else f"_{VARIANT}"

BASE = out_path("sequence-classifier")

UNFLIPPED_CHECKPOINT_DIR = f"{BASE}/ensemble-output-baseline/unflipped/checkpoints"
FLIPPED_CHECKPOINT_DIR   = f"{BASE}/ensemble-output-baseline/flipped/checkpoints"

TRAIN_EMBEDDINGS_CSV = out_path(f"embeddings/treatment_embeddings{SUFFIX}_train.csv")
TEST_EMBEDDINGS_CSV  = out_path(f"embeddings/treatment_embeddings{SUFFIX}_test.csv")
TRAIN_METADATA_CSV   = data_path(f"classifier/treatment_sequences{SUFFIX}_train.csv")
TEST_METADATA_CSV    = data_path(f"classifier/treatment_sequences{SUFFIX}_test.csv")
OUTPUT_CSV           = f"{BASE}/treatment_predictions{SUFFIX}_baseline.csv"

EMBEDDING_DIM = 1024
BATCH_SIZE    = 4096
NUM_WORKERS   = 8

BINARY_LABELS = ["Multiple sclerosis", "Other"]
# ───────────────────────────────────────────────────────────────────────────────


# ==============================================================================
# Model (verbatim from train_embedding_classifier.py)
# ==============================================================================

class EmbeddingClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=None, num_classes=2,
                 dropout=0.5, input_dropout=0.2):
        super().__init__()
        layers = []
        if input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))

        prev_dim = input_dim
        if hidden_dims:
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.classifier(x)


# ==============================================================================
# Data loading
# ==============================================================================

class EmbeddingDataset(Dataset):
    """Wraps a float32 numpy array for DataLoader batching."""
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.from_numpy(self.embeddings[idx])


def load_embeddings(csv_path: str, chunk_size: int = 50_000) -> np.ndarray:
    """Read embedding columns from CSV into a contiguous float32 array."""
    embedding_cols = [f"hidden_state_{i}" for i in range(EMBEDDING_DIM)]
    print(f"Loading embeddings from {csv_path} ...")

    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=embedding_cols, chunksize=chunk_size):
        chunks.append(chunk[embedding_cols].values.astype(np.float32))

    embeddings = np.concatenate(chunks, axis=0)
    print(f"  Loaded {len(embeddings):,} rows × {embeddings.shape[1]} dims")
    return embeddings


# ==============================================================================
# Model loading
# ==============================================================================

def load_models_from_dir(checkpoint_dir: str) -> list:
    """Load all model_*.pt checkpoints from a directory."""
    paths = sorted(Path(checkpoint_dir).glob("model_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    models = []
    for ckpt_path in paths:
        ckpt    = torch.load(ckpt_path, map_location="cpu")
        hparams = ckpt["hparams"]
        model   = EmbeddingClassifier(
            input_dim    = EMBEDDING_DIM,
            hidden_dims  = hparams.get("hidden_dims"),
            dropout      = hparams.get("dropout", 0.5),
            input_dropout= hparams.get("input_dropout", 0.0),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
        print(f"  Loaded {ckpt_path.name}")

    return models


# ==============================================================================
# Inference
# ==============================================================================

def run_inference(models: list, embeddings: np.ndarray,
                  device: torch.device) -> np.ndarray:
    """
    Run all models on embeddings in eval mode, average softmax probabilities.
    Returns float32 array of shape (N, 2).
    """
    dataset = EmbeddingDataset(embeddings)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    all_probs = []
    for model_idx, model in enumerate(models):
        model.to(device)
        model_probs = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Model {model_idx + 1}/{len(models)}",
                              leave=False):
                logits = model(batch.to(device, non_blocking=True))
                model_probs.append(torch.softmax(logits, dim=1).cpu())
        all_probs.append(torch.cat(model_probs, dim=0))
        model.cpu()
        torch.cuda.empty_cache()

    mean_probs = torch.stack(all_probs).mean(dim=0)
    return mean_probs.numpy()


# ==============================================================================
# Main
# ==============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

label_encoder = LabelEncoder().fit(BINARY_LABELS)
ms_id = int(label_encoder.transform(["Multiple sclerosis"])[0])

print("\n=== Loading checkpoints ===")
print("Unflipped (trained on train BioProjects):")
unflipped_models = load_models_from_dir(UNFLIPPED_CHECKPOINT_DIR)
print("Flipped (trained on test BioProjects):")
flipped_models = load_models_from_dir(FLIPPED_CHECKPOINT_DIR)

# --- Train-fold BioProjects: predict with flipped models (never saw them) ---
print("\n=== Train fold → flipped models ===")
train_emb   = load_embeddings(TRAIN_EMBEDDINGS_CSV)
train_probs = run_inference(flipped_models, train_emb, device)
train_meta  = pd.read_csv(TRAIN_METADATA_CSV)
assert len(train_meta) == len(train_probs), (
    f"Row mismatch: metadata={len(train_meta)}, embeddings={len(train_probs)}")
train_meta["ms_score"]        = train_probs[:, ms_id]
train_meta["predicted_class"] = label_encoder.inverse_transform(train_probs.argmax(axis=1))
del train_emb, train_probs
torch.cuda.empty_cache()

# --- Test-fold BioProjects: predict with unflipped models (never saw them) ---
print("\n=== Test fold → unflipped models ===")
test_emb   = load_embeddings(TEST_EMBEDDINGS_CSV)
test_probs = run_inference(unflipped_models, test_emb, device)
test_meta  = pd.read_csv(TEST_METADATA_CSV)
assert len(test_meta) == len(test_probs), (
    f"Row mismatch: metadata={len(test_meta)}, embeddings={len(test_probs)}")
test_meta["ms_score"]        = test_probs[:, ms_id]
test_meta["predicted_class"] = label_encoder.inverse_transform(test_probs.argmax(axis=1))
del test_emb, test_probs

# --- Combine and write ---
print("\n=== Writing output ===")
combined = pd.concat([train_meta, test_meta], ignore_index=True)
combined.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions written to {OUTPUT_CSV} ({len(combined):,} rows)")
