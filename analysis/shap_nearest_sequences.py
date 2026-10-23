#!/usr/bin/env python3
"""
SHAP Waypoint → Nearest Sequence Finder

Finds the training/test sequences closest (L2 in embedding space) to the top
SHAP-ranked contrastive waypoints.

Outputs:
  shap_nearest_training_sequences.csv  — TOP_N_SEQS nearest seqs per waypoint
"""

from bcrm.paths import data_path, model_path, out_path, repo_path
import os
import json
import re

import numpy as np
import pandas as pd
import torch

# ==============================================================================
# Configuration
# ==============================================================================

BASE = out_path("sequence-classifier")

CHECKPOINT_PATH    = f"{BASE}/waypoints-contrastive-final/checkpoint.pt"
TRAINING_INFO_JSON = f"{BASE}/waypoints-contrastive-final/training_info.json"
SHAP_RANKING_CSV   = f"{BASE}/intermediates/shap_feature_ranking.csv"

TRAIN_WEIGHTS_CSV  = data_path("classifier/classifier_data_igblast-train-weights.csv")
TEST_WEIGHTS_CSV   = data_path("classifier/classifier_data_igblast-test-weights.csv")

PREPROCESSED_DIR   = out_path("preprocessed")

OUTPUT_DIR = f"{BASE}/intermediates"

EMBEDDING_DIM        = 1024
TOP_K                = 10      # top SHAP waypoints to analyze (TOP_K/2 each direction)
TOP_N_SEQS           = 1000    # nearest sequences per waypoint to report
CHUNK_SIZE           = 50_000  # rows per GPU batch
FDR_THRESHOLD        = 0.05    # BH threshold for top-K waypoint selection

# ==============================================================================
# Embedding helpers (mirrors summarize_patients_contrastive.py)
# ==============================================================================

def get_embedding_path(csv_path, preprocessed_dir):
    """Return path to preprocessed .npy file; create it if missing."""
    base_name       = os.path.splitext(os.path.basename(csv_path))[0]
    embeddings_path = os.path.join(preprocessed_dir, f"{base_name}_embeddings.npy")

    if os.path.exists(embeddings_path):
        n_rows = np.load(embeddings_path, mmap_mode="r").shape[0]
        print(f"Found preprocessed embeddings for {base_name} ({n_rows:,} rows)")
        return embeddings_path, n_rows

    return _preprocess_embeddings(csv_path, embeddings_path)


def _preprocess_embeddings(csv_path, embeddings_path):
    print(f"\nPreprocessing embeddings from {csv_path}...")
    os.makedirs(os.path.dirname(embeddings_path) or ".", exist_ok=True)

    embedding_cols = [f"hidden_state_{i}" for i in range(EMBEDDING_DIM)]

    emb_arrays = []
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE,
                                           usecols=embedding_cols)):
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
    """Memory-mapped access to train+test embedding arrays."""

    def __init__(self, fold1_path, fold2_path):
        self.fold1 = np.load(fold1_path, mmap_mode="r")
        self.fold2 = np.load(fold2_path, mmap_mode="r")
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

    def get_range(self, start, end):
        """Efficient contiguous range read across fold boundary."""
        if end <= self.n1:
            return self.fold1[start:end]
        elif start >= self.n1:
            return self.fold2[start - self.n1 : end - self.n1]
        else:
            return np.concatenate([
                self.fold1[start:self.n1],
                self.fold2[0 : end - self.n1],
            ])


# ==============================================================================
# Main
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load SHAP ranking → top-K contrastive waypoint indices ─────────────
    print(f"\n=== Loading SHAP ranking from {SHAP_RANKING_CSV} ===")
    shap_df = pd.read_csv(SHAP_RANKING_CSV)

    # Contrastive waypoint features; the patient model uses mean activations only
    cw_features = shap_df[shap_df["feature"].str.match(r"^CW\d+_mean$")].copy()
    cw_features["wp_index"] = cw_features["feature"].str.extract(r"(\d+)").astype(int)

    # Keep only features that are statistically significant across patients
    if "fdr_p" in cw_features.columns:
        n_before = len(cw_features)
        cw_features = cw_features[cw_features["fdr_p"] < FDR_THRESHOLD].copy()
        print(f"  FDR filter (< {FDR_THRESHOLD}): {n_before} → {len(cw_features)} features")
    else:
        print("  Warning: fdr_p column missing — re-run patient_classifier.Rmd to add significance")

    k_half = TOP_K // 2  # e.g. 5 from each direction when TOP_K = 10

    # Positive-direction waypoints (push toward MS): most significant fdr_p, deduplicated
    pos_cw = cw_features[cw_features["signed_mean_shap"] > 0].sort_values(
        "fdr_p", ascending=True, kind="stable"  # ties keep ranking order, as R's arrange() does
    )
    seen: set = set()
    pos_wp: list = []
    for idx in pos_cw["wp_index"]:
        if idx not in seen:
            seen.add(idx)
            pos_wp.append(idx)
        if len(pos_wp) >= k_half:
            break

    # Negative-direction waypoints (push toward Other): most significant fdr_p, deduplicated
    neg_cw = cw_features[cw_features["signed_mean_shap"] < 0].sort_values(
        "fdr_p", ascending=True, kind="stable"  # ties keep ranking order, as R's arrange() does
    )
    seen = set()
    neg_wp: list = []
    for idx in neg_cw["wp_index"]:
        if idx not in seen:
            seen.add(idx)
            neg_wp.append(idx)
        if len(neg_wp) >= k_half:
            break

    # Positive first, then negative — matches R notebook heatmap x-axis order
    top_wp_indices = pos_wp + neg_wp

    print(f"Positive-direction waypoints ({len(pos_wp)}): {pos_wp}")
    print(f"Negative-direction waypoints ({len(neg_wp)}): {neg_wp}")
    print(f"Combined ({len(top_wp_indices)}): {top_wp_indices}")

    # ── 2. Load contrastive waypoint checkpoint ───────────────────────────────
    print(f"\n=== Loading contrastive checkpoint ===")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    waypoints = ckpt["model_state_dict"]["waypoints"]  # (W, 1024)
    print(f"  Waypoints shape: {waypoints.shape}")

    with open(TRAINING_INFO_JSON) as f:
        hparams = json.load(f)["hparams"]
    print(f"  temperature={hparams['temperature']}, num_waypoints={hparams['num_waypoints']}")

    # Select top-K waypoint vectors
    wp_subset = waypoints[top_wp_indices]  # (K, 1024)

    # ── 3. Load training + test embeddings ────────────────────────────────────
    print("\n=== Loading training embeddings ===")

    # Per-sequence embeddings written by embeddings/extract_embeddings.py
    train_emb_csv = out_path("embeddings/classifier_data_igblast_train_embeddings_weights.csv")
    test_emb_csv  = out_path("embeddings/classifier_data_igblast_test_embeddings_weights.csv")

    # Use the same preprocessing dir as train_waypoints_contrastive.py
    fold1_emb_path, fold1_n = get_embedding_path(train_emb_csv, PREPROCESSED_DIR)
    fold2_emb_path, fold2_n = get_embedding_path(test_emb_csv, PREPROCESSED_DIR)

    n_total = fold1_n + fold2_n
    print(f"Combined: {n_total:,} sequences")
    embeddings = CombinedEmbeddings(fold1_emb_path, fold2_emb_path)

    # Load metadata from weights CSVs
    meta_cols = ["Author", "Subject", "Disease", "v_call", "j_call"]
    meta_cols_optional = ["junction", "sequence"]

    def _load_meta(csv_path):
        all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        use_cols = [c for c in meta_cols + meta_cols_optional if c in all_cols]
        return pd.read_csv(csv_path, usecols=use_cols)

    print("Loading metadata...")
    train_meta = _load_meta(TRAIN_WEIGHTS_CSV)
    test_meta  = _load_meta(TEST_WEIGHTS_CSV)

    train_meta["split"] = "train"
    test_meta["split"]  = "test"
    combined_meta = pd.concat([train_meta, test_meta], ignore_index=True)
    assert len(combined_meta) == n_total, (
        f"Metadata rows ({len(combined_meta)}) != embedding rows ({n_total})")
    print(f"  Metadata: {len(combined_meta):,} rows")

    # ── 4. Find nearest training sequences to each waypoint (GPU) ─────────────
    print(f"\n=== Finding {TOP_N_SEQS} nearest sequences per waypoint ===")
    nearest_records = []

    for j, wp_idx in enumerate(top_wp_indices):
        wp_vec = waypoints[wp_idx].unsqueeze(0).to(device)  # (1, 1024)

        all_dists = []
        for start in range(0, n_total, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n_total)
            chunk = torch.from_numpy(
                embeddings.get_range(start, end).copy()
            ).float().to(device)
            dists = torch.cdist(chunk, wp_vec).squeeze(1)  # (chunk_size,)
            all_dists.append(dists.cpu())

        all_dists = torch.cat(all_dists)  # (n_total,)

        # Top-N nearest (smallest distance)
        topk = torch.topk(all_dists, TOP_N_SEQS, largest=False)

        for rank, (dist_val, idx) in enumerate(zip(topk.values, topk.indices)):
            row_idx = idx.item()
            meta_row = combined_meta.iloc[row_idx]
            record = {
                "wp_index": wp_idx,
                "rank":     rank + 1,
                "distance": dist_val.item(),
            }
            for col in meta_row.index:
                record[col] = meta_row[col]
            nearest_records.append(record)

        print(f"  CW{wp_idx}: closest dist={topk.values[0]:.4f}, "
              f"farthest dist={topk.values[-1]:.4f}")

    nearest_df = pd.DataFrame(nearest_records)
    nearest_path = os.path.join(OUTPUT_DIR, "shap_nearest_training_sequences.csv")
    nearest_df.to_csv(nearest_path, index=False)
    print(f"\nSaved {len(nearest_df)} rows to {nearest_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"  Nearest training sequences: {nearest_path}")
    print(f"  Top-K waypoints: {top_wp_indices}")
    print(f"  TOP_N_SEQS per waypoint: {TOP_N_SEQS}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
