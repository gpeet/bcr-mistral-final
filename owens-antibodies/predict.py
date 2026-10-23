"""
Ensemble prediction for Owens holdout sequences.

These are true holdout data (not in any training fold), so all models from
both folds are used. Runs both exclusion and baseline classifier variants
for both sequence and junction embeddings.
Outputs per-fold scores and max-fold scores alongside the all-model mean.
"""

from bcrm.paths import data_path, model_path, out_path, repo_path
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE = out_path("sequence-classifier")

CHECKPOINT_DIRS = {
    "exclusion": [
        f"{BASE}/ensemble-output-exclusion/unflipped/checkpoints",
        f"{BASE}/ensemble-output-exclusion/flipped/checkpoints",
    ],
    "baseline": [
        f"{BASE}/ensemble-output-baseline/unflipped/checkpoints",
        f"{BASE}/ensemble-output-baseline/flipped/checkpoints",
    ],
}

JUNCTION_CHECKPOINT_DIRS = {
    "exclusion_junction": [
        f"{BASE}/ensemble-output-junction-exclusion/unflipped/checkpoints",
        f"{BASE}/ensemble-output-junction-exclusion/flipped/checkpoints",
    ],
    "baseline_junction": [
        f"{BASE}/ensemble-output-junction-baseline/unflipped/checkpoints",
        f"{BASE}/ensemble-output-junction-baseline/flipped/checkpoints",
    ],
}

EMBEDDINGS_CSV          = out_path("owens/owens_embeddings.csv")
JUNCTION_EMBEDDINGS_CSV = out_path("owens/owens_junction_embeddings.csv")
METADATA_CSV            = data_path("owens/owens_sequences.csv")
OUTPUT_CSV              = out_path("owens/owens_predictions.csv")
PER_MODEL_CSV           = out_path("owens/owens_predictions_per_model.csv")

EMBEDDING_DIM = 1024
BATCH_SIZE    = 4096
NUM_WORKERS   = 4

BINARY_LABELS = ["Multiple sclerosis", "Other"]
# ──────────────────────────────────────────────────────────────────────────────


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
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.from_numpy(self.embeddings[idx])


def load_embeddings(csv_path: str, chunk_size: int = 50_000) -> np.ndarray:
    embedding_cols = [f"hidden_state_{i}" for i in range(EMBEDDING_DIM)]
    print(f"Loading embeddings from {csv_path} ...")

    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=embedding_cols, chunksize=chunk_size):
        chunks.append(chunk[embedding_cols].values.astype(np.float32))

    embeddings = np.concatenate(chunks, axis=0)
    print(f"  Loaded {len(embeddings):,} rows x {embeddings.shape[1]} dims")
    return embeddings


def load_models_from_dir(checkpoint_dir: str) -> list:
    paths = sorted(Path(checkpoint_dir).glob("model_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    models = []
    for ckpt_path in paths:
        ckpt    = torch.load(ckpt_path, map_location="cpu")
        hparams = ckpt["hparams"]
        model   = EmbeddingClassifier(
            input_dim     = EMBEDDING_DIM,
            hidden_dims   = hparams.get("hidden_dims"),
            dropout       = hparams.get("dropout", 0.5),
            input_dropout = hparams.get("input_dropout", 0.0),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)
        print(f"  Loaded {ckpt_path.name}")

    return models


def run_inference(models: list, embeddings: np.ndarray,
                  device: torch.device,
                  return_per_model: bool = False):
    """Average softmax probabilities across models. Returns (N, 2) float32.

    If return_per_model=True, also returns a list of (N, 2) per-model arrays
    (one entry per model, in load order) for uncertainty quantification.
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

    stacked    = torch.stack(all_probs)       # (n_models, N, 2)
    mean_probs = stacked.mean(dim=0).numpy()  # (N, 2)

    if return_per_model:
        return mean_probs, [p.numpy() for p in all_probs]
    return mean_probs


# ==============================================================================
# Main
# ==============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

label_encoder = LabelEncoder().fit(BINARY_LABELS)
ms_id = int(label_encoder.transform(["Multiple sclerosis"])[0])

embeddings = load_embeddings(EMBEDDINGS_CSV)
metadata   = pd.read_csv(METADATA_CSV)
assert len(metadata) == len(embeddings), (
    f"Row mismatch: metadata={len(metadata)}, embeddings={len(embeddings)}")

per_model_rows = []  # accumulates long-format per-model scores for uncertainty plots

for variant, (unflipped_dir, flipped_dir) in CHECKPOINT_DIRS.items():
    print(f"\n=== {variant} classifier ===")

    print(f"Unflipped models ({unflipped_dir}):")
    unflipped_models = load_models_from_dir(unflipped_dir)
    unflipped_probs, unflipped_per_model = run_inference(
        unflipped_models, embeddings, device, return_per_model=True)

    print(f"Flipped models ({flipped_dir}):")
    flipped_models = load_models_from_dir(flipped_dir)
    flipped_probs, flipped_per_model = run_inference(
        flipped_models, embeddings, device, return_per_model=True)

    # All-model mean (20 models combined)
    all_probs = (unflipped_probs + flipped_probs) / 2.0

    metadata[f"ms_score_{variant}"]           = all_probs[:, ms_id]
    metadata[f"ms_score_{variant}_unflipped"]  = unflipped_probs[:, ms_id]
    metadata[f"ms_score_{variant}_flipped"]    = flipped_probs[:, ms_id]
    metadata[f"ms_score_{variant}_maxfold"]    = np.maximum(
        unflipped_probs[:, ms_id], flipped_probs[:, ms_id]
    )
    metadata[f"predicted_class_{variant}"]     = label_encoder.inverse_transform(
        all_probs.argmax(axis=1)
    )

    # Collect individual model scores (long format) for uncertainty quantification
    for fold, per_model_list in [("unflipped", unflipped_per_model),
                                  ("flipped",   flipped_per_model)]:
        for model_idx, probs in enumerate(per_model_list):
            per_model_rows.append(pd.DataFrame({
                "sequence_id": metadata["sequence_id"].values,
                "variant":     variant,
                "fold":        fold,
                "model_idx":   model_idx,
                "ms_score":    probs[:, ms_id],
            }))

    del unflipped_models, flipped_models, unflipped_probs, flipped_probs, all_probs
    del unflipped_per_model, flipped_per_model
    torch.cuda.empty_cache()

# ==============================================================================
# Junction classifiers
# ==============================================================================

junction_embeddings = load_embeddings(JUNCTION_EMBEDDINGS_CSV)
assert len(metadata) == len(junction_embeddings), (
    f"Row mismatch: metadata={len(metadata)}, junction_embeddings={len(junction_embeddings)}")

# Flag rows where junction is missing — scores will be uninformative
junction_na = metadata["junction"].isna() | (metadata["junction"].astype(str).str.strip() == "")
metadata["junction_missing"] = junction_na
n_missing = junction_na.sum()
if n_missing > 0:
    print(f"Warning: {n_missing:,} sequences have missing junction — scores will be set to NaN")

for variant, (unflipped_dir, flipped_dir) in JUNCTION_CHECKPOINT_DIRS.items():
    print(f"\n=== {variant} classifier ===")

    print(f"Unflipped models ({unflipped_dir}):")
    unflipped_models = load_models_from_dir(unflipped_dir)
    unflipped_probs, unflipped_per_model = run_inference(
        unflipped_models, junction_embeddings, device, return_per_model=True)

    print(f"Flipped models ({flipped_dir}):")
    flipped_models = load_models_from_dir(flipped_dir)
    flipped_probs, flipped_per_model = run_inference(
        flipped_models, junction_embeddings, device, return_per_model=True)

    # All-model mean (20 models combined)
    all_probs = (unflipped_probs + flipped_probs) / 2.0

    metadata[f"ms_score_{variant}"]           = all_probs[:, ms_id]
    metadata[f"ms_score_{variant}_unflipped"]  = unflipped_probs[:, ms_id]
    metadata[f"ms_score_{variant}_flipped"]    = flipped_probs[:, ms_id]
    metadata[f"ms_score_{variant}_maxfold"]    = np.maximum(
        unflipped_probs[:, ms_id], flipped_probs[:, ms_id]
    )
    metadata[f"predicted_class_{variant}"]     = label_encoder.inverse_transform(
        all_probs.argmax(axis=1)
    )

    # NaN-out scores for sequences with missing junctions
    if n_missing > 0:
        for col in [f"ms_score_{variant}", f"ms_score_{variant}_unflipped",
                    f"ms_score_{variant}_flipped", f"ms_score_{variant}_maxfold"]:
            metadata.loc[junction_na, col] = np.nan
        metadata.loc[junction_na, f"predicted_class_{variant}"] = np.nan

    # Collect individual model scores with NaN masking for missing junctions
    for fold, per_model_list in [("unflipped", unflipped_per_model),
                                  ("flipped",   flipped_per_model)]:
        for model_idx, probs in enumerate(per_model_list):
            ms_scores = probs[:, ms_id].copy()
            if n_missing > 0:
                ms_scores[junction_na.values] = np.nan
            per_model_rows.append(pd.DataFrame({
                "sequence_id": metadata["sequence_id"].values,
                "variant":     variant,
                "fold":        fold,
                "model_idx":   model_idx,
                "ms_score":    ms_scores,
            }))

    del unflipped_models, flipped_models, unflipped_probs, flipped_probs, all_probs
    del unflipped_per_model, flipped_per_model
    torch.cuda.empty_cache()

metadata.to_csv(OUTPUT_CSV, index=False)
print(f"\nPredictions written to {OUTPUT_CSV} ({len(metadata):,} rows)")

per_model_df = pd.concat(per_model_rows, ignore_index=True)
per_model_df.to_csv(PER_MODEL_CSV, index=False)
print(f"Per-model predictions written to {PER_MODEL_CSV} ({len(per_model_df):,} rows)")
