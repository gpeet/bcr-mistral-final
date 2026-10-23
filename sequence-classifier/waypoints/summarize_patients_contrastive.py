#!/usr/bin/env python3
"""
Summarize Patients — Contrastive Waypoint Activation Features

Converts per-sequence embeddings into per-patient feature vectors using
the trained contrastive waypoint model.  For each patient repertoire the
mean and max of the temperature-scaled fingerprint activations are computed,
producing 2 * W features (W = number of waypoints).

The activation formula matches tune_waypoints_contrastive.py exactly:
    fingerprint = softmax(-cdist(x, waypoints) / temperature, dim=1)

Outputs:
  patient_features_contrastive.csv — one row per patient–tissue combination with columns:
      Author, Subject, Disease, BSource, n_seqs,
      ms_score_median, ms_score_85pct, ms_score_90pct, ms_score_95pct, ms_score_99pct,
      ms_score_85pct_weighted, ms_score_90pct_weighted, ms_score_95pct_weighted,
      ms_score_99pct_weighted, ms_score_sd,
      CW0_mean, CW0_max, ..., CW{W-1}_mean, CW{W-1}_max
"""

from bcrm.paths import data_path, model_path, out_path, repo_path
import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch


# ==============================================================================
# Configuration
# ==============================================================================

FOLD1_CSV          = out_path("embeddings/classifier_data_igblast_train_embeddings_weights.csv")
FOLD1_WEIGHTS_CSV  = data_path("classifier/classifier_data_igblast-train-weights.csv")
FOLD2_CSV          = out_path("embeddings/classifier_data_igblast_test_embeddings_weights.csv")
FOLD2_WEIGHTS_CSV  = data_path("classifier/classifier_data_igblast-test-weights.csv")

PREPROCESSED_DIR   = out_path("preprocessed")
TRAINING_INFO_JSON = out_path("sequence-classifier/waypoints-contrastive-final/training_info.json")
CHECKPOINT_PATH    = out_path("sequence-classifier/waypoints-contrastive-final/checkpoint.pt")
OUTPUT_CSV         = out_path("sequence-classifier/waypoints-contrastive-final/patient_features_contrastive.csv")

ENSEMBLE_DIR       = out_path("sequence-classifier/ensemble-output-exclusion")
PRED_UNFLIPPED     = os.path.join(ENSEMBLE_DIR, "unflipped/predictions/ensemble_predictions.csv")
PRED_FLIPPED       = os.path.join(ENSEMBLE_DIR, "flipped/predictions/ensemble_predictions.csv")

EMBEDDING_DIM      = 1024
AUTHOR_COL         = "Author"
SUBJECT_COL        = "Subject"
DISEASE_COL        = "Disease"
BSOURCE_COL        = "BSource"
MIN_SEQS_PER_REP   = 50       # minimum sequences to include a patient
PREPROCESS_CHUNK_SIZE = 50000


# ==============================================================================
# Preprocessing helpers (mirrors train_waypoints.py)
# ==============================================================================

def get_embedding_path(csv_path, preprocessed_dir):
    base_name       = os.path.splitext(os.path.basename(csv_path))[0]
    embeddings_path = os.path.join(preprocessed_dir, f"{base_name}_embeddings.npy")

    if os.path.exists(embeddings_path):
        n_rows = np.load(embeddings_path, mmap_mode='r').shape[0]
        print(f"Found preprocessed embeddings for {base_name} ({n_rows:,} rows)")
        return embeddings_path, n_rows

    return _preprocess_embeddings(csv_path, embeddings_path)


def _preprocess_embeddings(csv_path, embeddings_path):
    print(f"\nPreprocessing embeddings from {csv_path}...")
    os.makedirs(os.path.dirname(embeddings_path) or ".", exist_ok=True)

    embedding_cols = [f"hidden_state_{i}" for i in range(EMBEDDING_DIM)]

    # Single-pass: collect chunks then write mmap with correct size
    emb_arrays = []
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=PREPROCESS_CHUNK_SIZE, usecols=embedding_cols)):
        emb_arrays.append(chunk.values.astype(np.float32))
        print(f"  Chunk {i+1}: {sum(a.shape[0] for a in emb_arrays):,} rows")
    all_emb = np.concatenate(emb_arrays)
    del emb_arrays
    n_rows = all_emb.shape[0]
    print(f"  Total rows: {n_rows:,}")

    mmap = np.lib.format.open_memmap(
        embeddings_path, mode="w+", dtype=np.float32, shape=(n_rows, EMBEDDING_DIM)
    )
    mmap[:] = all_emb
    del all_emb, mmap
    return embeddings_path, n_rows


class CombinedEmbeddings:
    def __init__(self, fold1_path, fold2_path):
        self.fold1 = np.load(fold1_path, mmap_mode='r')
        self.fold2 = np.load(fold2_path, mmap_mode='r')
        self.n1    = len(self.fold1)

    def __len__(self):
        return self.n1 + len(self.fold2)

    def __getitem__(self, indices):
        indices = np.asarray(indices)
        result  = np.empty((len(indices), EMBEDDING_DIM), dtype=np.float32)
        fold1_mask = indices < self.n1
        fold2_mask = ~fold1_mask
        if fold1_mask.any():
            result[fold1_mask] = self.fold1[indices[fold1_mask]]
        if fold2_mask.any():
            result[fold2_mask] = self.fold2[indices[fold2_mask] - self.n1]
        return result


# ==============================================================================
# Nested index (Author -> Subject -> indices) + disease labels
# ==============================================================================

def build_nested_index(fold1_weights_csv, fold2_weights_csv,
                       fold1_n, min_seqs=MIN_SEQS_PER_REP):
    print("Building nested Author -> Subject -> BSource index...")

    df1 = pd.read_csv(fold1_weights_csv, usecols=[AUTHOR_COL, SUBJECT_COL, DISEASE_COL, BSOURCE_COL])
    df2 = pd.read_csv(fold2_weights_csv, usecols=[AUTHOR_COL, SUBJECT_COL, DISEASE_COL, BSOURCE_COL])

    subject_to_indices = defaultdict(list)
    subject_to_disease = {}

    for idx, row in enumerate(df1.itertuples(index=False)):
        subj_id = f"{row.Author}::{row.Subject}::{row.BSource}"
        subject_to_indices[subj_id].append(idx)
        subject_to_disease[subj_id] = row.Disease

    for idx, row in enumerate(df2.itertuples(index=False)):
        subj_id = f"{row.Author}::{row.Subject}::{row.BSource}"
        subject_to_indices[subj_id].append(fold1_n + idx)
        subject_to_disease[subj_id] = row.Disease

    n_before = len(subject_to_indices)
    subject_to_indices = {s: v for s, v in subject_to_indices.items()
                          if len(v) >= min_seqs}
    # Keep only disease labels for retained subjects
    subject_to_disease = {s: d for s, d in subject_to_disease.items()
                          if s in subject_to_indices}
    n_after = len(subject_to_indices)

    author_to_subjects = defaultdict(list)
    for subj_id in subject_to_indices:
        author_to_subjects[subj_id.split("::")[0]].append(subj_id)
    author_to_subjects = dict(author_to_subjects)

    sizes = [len(v) for v in subject_to_indices.values()]
    print(f"  {len(author_to_subjects)} authors  |  "
          f"{n_after} subjects retained ({n_before - n_after} dropped: <{min_seqs} seqs)")
    print(f"  Sequences: {sum(sizes):,}  "
          f"min={min(sizes):,}  median={int(np.median(sizes)):,}  max={max(sizes):,}")

    return author_to_subjects, subject_to_indices, subject_to_disease


# ==============================================================================
# Prediction-derived per-patient features
# ==============================================================================

def weighted_percentile(scores, weights, percentile):
    """Weighted percentile via sorted cumulative weights."""
    idx           = np.argsort(scores)
    sorted_scores = scores[idx]
    sorted_weights = weights[idx]
    cum_w         = np.cumsum(sorted_weights, dtype=float)
    cum_w        /= cum_w[-1]  # normalize to [0, 1]
    return float(np.interp(percentile / 100.0, cum_w, sorted_scores))


def build_prediction_index(pred_unflipped_csv, pred_flipped_csv,
                           fold2_weights_csv, fold1_weights_csv):
    """Load fold-aware ensemble predictions + consensus_count from weights CSVs.

    Unflipped models trained on fold 1 predict fold 2 (test).
    Flipped models trained on fold 2 predict fold 1 (train).

    Predictions and weights CSVs are row-aligned, so we join by row index
    then group by Author::Subject to compute per-patient summary stats.

    Returns dict: "Author::Subject" -> {ms_score_median, ms_score_85pct,
        ms_score_90pct, ms_score_95pct, ms_score_99pct,
        ms_score_85pct_weighted, ms_score_90pct_weighted,
        ms_score_95pct_weighted, ms_score_99pct_weighted, ms_score_sd}
    """
    print("\nBuilding prediction index...")

    def _load_fold(pred_csv, weights_csv):
        preds = pd.read_csv(pred_csv, usecols=["ms_score", AUTHOR_COL, SUBJECT_COL])
        weights = pd.read_csv(weights_csv, usecols=["consensus_count", BSOURCE_COL])
        assert len(preds) == len(weights), (
            f"Row count mismatch: {pred_csv} ({len(preds)}) vs "
            f"{weights_csv} ({len(weights)})"
        )
        preds["consensus_count"] = weights["consensus_count"].values
        preds[BSOURCE_COL] = weights[BSOURCE_COL].values
        return preds

    # Unflipped predicts fold 2 (test weights), flipped predicts fold 1 (train weights)
    df_unflipped = _load_fold(pred_unflipped_csv, fold2_weights_csv)
    df_flipped   = _load_fold(pred_flipped_csv,   fold1_weights_csv)
    combined     = pd.concat([df_unflipped, df_flipped], ignore_index=True)

    print(f"  Loaded {len(combined):,} sequence predictions "
          f"({len(df_unflipped):,} unflipped + {len(df_flipped):,} flipped)")

    subject_to_preds = {}
    for (author, subject, bsource), grp in combined.groupby([AUTHOR_COL, SUBJECT_COL, BSOURCE_COL]):
        scores = grp["ms_score"].values
        counts = grp["consensus_count"].values

        subject_to_preds[f"{author}::{subject}::{bsource}"] = {
            "ms_score_median":           float(np.median(scores)),
            "ms_score_85pct":            float(np.percentile(scores, 85)),
            "ms_score_90pct":            float(np.percentile(scores, 90)),
            "ms_score_95pct":            float(np.percentile(scores, 95)),
            "ms_score_99pct":            float(np.percentile(scores, 99)),
            "ms_score_85pct_weighted":   weighted_percentile(scores, counts, 85),
            "ms_score_90pct_weighted":   weighted_percentile(scores, counts, 90),
            "ms_score_95pct_weighted":   weighted_percentile(scores, counts, 95),
            "ms_score_99pct_weighted":   weighted_percentile(scores, counts, 99),
            "ms_score_sd":               float(scores.std()),
        }

    print(f"  {len(subject_to_preds):,} patient–tissue combinations with prediction features")
    return subject_to_preds


# ==============================================================================
# Waypoint model (must match tune_waypoints_contrastive.py exactly)
# ==============================================================================

class WaypointAutoencoder(torch.nn.Module):
    """
    Softmax-temperature waypoint autoencoder with a decoupled linear decoder.
    Must match tune_waypoints_contrastive.py exactly.

    Only the waypoints + temperature are used here (fingerprint extraction);
    the decoder is loaded for checkpoint compatibility but not called.
    """

    def __init__(self, num_waypoints, temperature=0.5, input_dim=EMBEDDING_DIM):
        super().__init__()
        self.temperature = temperature
        self.waypoints = torch.nn.Parameter(torch.randn(num_waypoints, input_dim))
        self.decoder = torch.nn.Linear(num_waypoints, input_dim, bias=False)

    def forward(self, x):
        dists       = torch.cdist(x, self.waypoints, p=2)
        fingerprint = torch.softmax(-dists / self.temperature, dim=1)  # (N, W)
        return fingerprint  # (N_seqs, W) — softmax-temperature activations


# ==============================================================================
# Main
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(PREPROCESSED_DIR, exist_ok=True)

    # --- Load training metadata ---
    print(f"\nLoading training info from {TRAINING_INFO_JSON}")
    with open(TRAINING_INFO_JSON) as f:
        training_info = json.load(f)

    hparams       = training_info['hparams']
    num_waypoints = hparams['num_waypoints']
    temperature   = hparams['temperature']
    print(f"  num_waypoints={num_waypoints}  temperature={temperature:.4f}")

    # --- Load checkpoint ---
    print(f"\nLoading checkpoint from {CHECKPOINT_PATH}")
    ckpt  = torch.load(CHECKPOINT_PATH, map_location=device)
    model = WaypointAutoencoder(num_waypoints=num_waypoints, temperature=temperature)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    waypoints_tensor = model.waypoints.detach()  # stays on device
    print(f"  Waypoints shape: {waypoints_tensor.shape}")

    # --- Embeddings ---
    print("\n=== Loading Fold 1 Embeddings ===")
    fold1_emb_path, fold1_n = get_embedding_path(FOLD1_CSV, PREPROCESSED_DIR)

    print("\n=== Loading Fold 2 Embeddings ===")
    fold2_emb_path, fold2_n = get_embedding_path(FOLD2_CSV, PREPROCESSED_DIR)

    print(f"\nCombined: {fold1_n + fold2_n:,} sequences")
    embeddings = CombinedEmbeddings(fold1_emb_path, fold2_emb_path)

    # --- Subject index + disease labels ---
    author_to_subjects, subject_to_indices, subject_to_disease = build_nested_index(
        FOLD1_WEIGHTS_CSV, FOLD2_WEIGHTS_CSV, fold1_n, min_seqs=MIN_SEQS_PER_REP
    )

    # --- Prediction-derived features ---
    subject_to_preds = build_prediction_index(
        PRED_UNFLIPPED, PRED_FLIPPED, FOLD2_WEIGHTS_CSV, FOLD1_WEIGHTS_CSV
    )

    # --- Compute per-patient features ---
    print(f"\n=== Computing contrastive patient features ({num_waypoints} waypoints) ===")
    patient_records = []
    all_subject_ids = [s for subjects in author_to_subjects.values() for s in subjects]

    for i, subject_id in enumerate(all_subject_ids):
        author, subject, bsource = subject_id.split("::", 2)
        indices = np.asarray(subject_to_indices[subject_id])

        x = torch.from_numpy(embeddings[indices].copy()).float().to(device)

        with torch.no_grad():
            # softmax-temperature fingerprints: (N_seqs, W)
            dists       = torch.cdist(x, waypoints_tensor, p=2)
            fingerprint = torch.softmax(-dists / temperature, dim=1)

        mean_acts = fingerprint.mean(dim=0).cpu().numpy()       # (W,)
        max_acts  = fingerprint.max(dim=0).values.cpu().numpy() # (W,)

        row = {
            AUTHOR_COL:  author,
            SUBJECT_COL: subject,
            DISEASE_COL: subject_to_disease[subject_id],
            BSOURCE_COL: bsource,
            "n_seqs":    len(indices),
        }

        # Prediction-derived features (NaN if patient has no predictions)
        pred_feats = subject_to_preds.get(subject_id, {})
        for key in ["ms_score_median", "ms_score_85pct", "ms_score_90pct",
                     "ms_score_95pct", "ms_score_99pct",
                     "ms_score_85pct_weighted", "ms_score_90pct_weighted",
                     "ms_score_95pct_weighted", "ms_score_99pct_weighted",
                     "ms_score_sd"]:
            row[key] = pred_feats.get(key, np.nan)

        for j in range(num_waypoints):
            row[f"CW{j}_mean"] = float(mean_acts[j])
            row[f"CW{j}_max"]  = float(max_acts[j])

        patient_records.append(row)

        if (i + 1) % 50 == 0 or (i + 1) == len(all_subject_ids):
            print(f"  {i + 1}/{len(all_subject_ids)} patients processed")

    # --- Save ---
    df = pd.DataFrame(patient_records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} patients × {len(df.columns)} columns to {OUTPUT_CSV}")

    # Sanity checks
    cw_mean_cols = [c for c in df.columns if c.startswith("CW") and c.endswith("_mean")]
    pred_cols = ["ms_score_median", "ms_score_85pct", "ms_score_90pct",
                 "ms_score_95pct", "ms_score_99pct",
                 "ms_score_85pct_weighted", "ms_score_90pct_weighted",
                 "ms_score_95pct_weighted", "ms_score_99pct_weighted",
                 "ms_score_sd"]
    n_with_preds = df["ms_score_median"].notna().sum()
    print(f"Contrastive waypoint features: {len(cw_mean_cols)} mean + {len(cw_mean_cols)} max = "
          f"{len(cw_mean_cols) * 2}")
    print(f"Prediction features: {len(pred_cols)} columns  "
          f"({n_with_preds}/{len(df)} patients have predictions)")
    print(f"n_seqs: min={df['n_seqs'].min()}  median={int(df['n_seqs'].median())}  "
          f"max={df['n_seqs'].max()}")
    print(f"Disease distribution:\n{df[DISEASE_COL].value_counts().to_string()}")


if __name__ == "__main__":
    main()
