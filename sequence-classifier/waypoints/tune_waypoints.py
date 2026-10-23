#!/usr/bin/env python3
"""
Waypoint Hyperparameter Tuning — Repertoire-Level Reconstruction

Learns waypoints in 1024-dim embedding space such that for each sequence x,
a softmax-temperature fingerprint over distances to waypoints is passed through
a *decoupled* linear decoder to reconstruct x.

Changes from v1:
  - Replaced inverse-distance^power activation with softmax(-dist / temperature).
    This is numerically stable for any temperature, avoids d^p overflow at high
    powers, and has a clean interpretation: temperature → 0 is nearest-neighbor,
    temperature → ∞ is uniform mixing.
  - Added a separate linear decoder (nn.Linear, no bias) so the mixing basis
    and the anchor positions are independent parameters.  The waypoints remain
    pure anchors; the decoder learns how to project the fingerprint back into
    embedding space.
  - Tunes `temperature` (log-scale 0.01–2.0) instead of `power`.
  - num_waypoints is fixed at 128 (not a hyperparameter).

Data Strategy (Nested Grouping):
  1. K-Fold Cross Validation is split by AUTHOR (Study). This prevents batch
     effect leakage and forces the model to learn universal biological
     representations that generalize to completely unseen laboratories.
  2. Training / Gradient steps are performed by SUBJECT (Patient). Each step
     processes one human's repertoire (downsampled if needed), forcing the
     waypoints to map the distinct immune landscape of individuals.

Both fold1 (TRAIN) and fold2 (TEST) are combined into a single unsupervised
study — no labels are used.

Metric: mean validation reconstruction MSE across K folds (minimize).
"""

from bcrm.paths import data_path, model_path, out_path, repo_path
import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from optuna.pruners import MedianPruner


# ==============================================================================
# Configuration
# ==============================================================================

FOLD1_CSV         = out_path("embeddings/classifier_data_igblast_train_embeddings_weights.csv")
FOLD1_WEIGHTS_CSV = data_path("classifier/classifier_data_igblast-train-weights.csv")
FOLD2_CSV         = out_path("embeddings/classifier_data_igblast_test_embeddings_weights.csv")
FOLD2_WEIGHTS_CSV = data_path("classifier/classifier_data_igblast-test-weights.csv")

PREPROCESSED_DIR  = out_path("preprocessed")

STUDY_DIR         = out_path("sequence-classifier/waypoint-hparam-search")
STUDY_DB          = os.path.join(STUDY_DIR, "optuna_waypoints.db")
RESULTS_CSV       = os.path.join(STUDY_DIR, "trial_results_waypoints.csv")
BEST_CONFIG_JSON  = os.path.join(STUDY_DIR, "best_config_waypoints.json")

EMBEDDING_DIM      = 1024
AUTHOR_COL         = "Author"
SUBJECT_COL        = "Subject"

MIN_SEQS_PER_REP   = 50    # drop patient repertoires smaller than this
MAX_SEQS_PER_REP   = 1000  # downsample patient repertoires to this if larger

N_FOLDS            = 5
N_TRIALS           = int(os.environ.get("BCRM_TUNE_TRIALS", 200))  # lower it for smoke tests

MAX_EPOCHS            = 200
EARLY_STOP_PATIENCE   = 5
EARLY_STOP_MIN_DELTA  = 1e-6

MAX_GRAD_NORM         = 1.0
PREPROCESS_CHUNK_SIZE = 50000

NUM_WAYPOINTS = 128  # fixed — not a hyperparameter


# ==============================================================================
# Preprocessing — embeddings only
# ==============================================================================

def get_embedding_path(csv_path, preprocessed_dir):
    base_name       = os.path.splitext(os.path.basename(csv_path))[0]
    output_prefix   = os.path.join(preprocessed_dir, base_name)
    embeddings_path = f"{output_prefix}_embeddings.npy"

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
        print("  -> Loading Fold 1 into RAM to prevent Lustre hanging...")
        self.fold1 = np.load(fold1_path)
        print("  -> Loading Fold 2 into RAM to prevent Lustre hanging...")
        self.fold2 = np.load(fold2_path)
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
# Nested Index (Author -> Subject -> Indices)
# ==============================================================================

def build_nested_index(fold1_weights_csv, fold2_weights_csv, fold1_n, min_seqs=MIN_SEQS_PER_REP):
    """
    Build nested mapping to support grouping by Subject during training
    and splitting by Author during CV.
    """
    print("Building nested Author -> Subject index...")

    df1 = pd.read_csv(fold1_weights_csv, usecols=[AUTHOR_COL, SUBJECT_COL])
    df2 = pd.read_csv(fold2_weights_csv, usecols=[AUTHOR_COL, SUBJECT_COL])

    subject_to_indices = defaultdict(list)

    for idx, (author, subject) in enumerate(zip(df1[AUTHOR_COL], df1[SUBJECT_COL])):
        subj_id = f"{author}::{subject}"
        subject_to_indices[subj_id].append(idx)

    for idx, (author, subject) in enumerate(zip(df2[AUTHOR_COL], df2[SUBJECT_COL])):
        subj_id = f"{author}::{subject}"
        subject_to_indices[subj_id].append(fold1_n + idx)

    n_subj_before = len(subject_to_indices)
    subject_to_indices = {
        s: v for s, v in subject_to_indices.items() if len(v) >= min_seqs
    }
    n_subj_after = len(subject_to_indices)

    author_to_subjects = defaultdict(list)
    for subj_id in subject_to_indices.keys():
        author = subj_id.split("::")[0]
        author_to_subjects[author].append(subj_id)

    author_to_subjects = {a: s for a, s in author_to_subjects.items() if s}

    sizes = [len(v) for v in subject_to_indices.values()]
    print(f"  {len(author_to_subjects)} authors retained.")
    print(f"  {n_subj_after} subjects retained ({n_subj_before - n_subj_after} dropped: <{min_seqs} seqs).")
    print(f"  Sequences: {sum(sizes):,}  "
          f"min={min(sizes):,}  median={int(np.median(sizes)):,}  max={max(sizes):,}")

    return author_to_subjects, subject_to_indices


def create_author_kfold_splits(author_list, n_splits=N_FOLDS, seed=42):
    rng      = np.random.RandomState(seed)
    shuffled = list(author_list)
    rng.shuffle(shuffled)

    folds = [shuffled[i::n_splits] for i in range(n_splits)]

    splits = []
    for val_idx in range(n_splits):
        val_authors   = set(folds[val_idx])
        train_authors = set(a for i, fold in enumerate(folds) if i != val_idx for a in fold)
        splits.append((train_authors, val_authors))
    return splits


# ==============================================================================
# Model
# ==============================================================================

class WaypointAutoencoder(nn.Module):
    """
    Softmax-temperature waypoint autoencoder with a decoupled linear decoder.

    Forward pass:
      1. Compute L2 distances from each sequence to every waypoint.
      2. Apply softmax(-dist / temperature) to get a probability fingerprint.
         - temperature → 0 : one-hot (nearest-neighbor lookup)
         - temperature → ∞ : uniform mixture of all waypoints
         - Numerically stable for any positive temperature (no d^power overflow).
      3. Project the fingerprint through a learned linear decoder
         (num_waypoints → embedding_dim, no bias) to reconstruct x.
         The decoder is independent of waypoint positions, so the two sets of
         parameters can specialise without fighting each other.
    """

    def __init__(self, num_waypoints, temperature=0.5, input_dim=EMBEDDING_DIM):
        super().__init__()
        self.temperature = temperature
        # Waypoints: pure anchors for distance computation
        self.waypoints = nn.Parameter(torch.randn(num_waypoints, input_dim))
        # Decoder: maps fingerprint (num_waypoints,) → reconstruction (input_dim,)
        self.decoder = nn.Linear(num_waypoints, input_dim, bias=False)

    def forward(self, x):
        # 1. Pairwise L2 distances  (N, W)
        dists = torch.cdist(x, self.waypoints, p=2)

        # 2. Softmax fingerprint — numerically stable, sums to 1 automatically
        #    Dividing by temperature before softmax is equivalent to the
        #    original inverse-distance formulation at the limit, but is safe
        #    for any temperature > 0 because softmax handles large logits.
        fingerprint = torch.softmax(-dists / self.temperature, dim=1)  # (N, W)

        # 3. Decode: learned linear projection, no bias
        #    (N, W) @ (W, D)^T  →  (N, D)
        recon = self.decoder(fingerprint)

        return recon


# ==============================================================================
# Helpers
# ==============================================================================

def load_repertoire(subject_id, subject_to_indices, embeddings, n_seq_per_rep, rng):
    """Return a float32 tensor (N, D) for one patient, downsampled if needed."""
    indices = np.asarray(subject_to_indices[subject_id])
    if len(indices) > n_seq_per_rep:
        indices = rng.choice(indices, size=n_seq_per_rep, replace=False)
    return torch.from_numpy(embeddings[indices].copy()).float()


def sample_init_waypoints(num_waypoints, train_authors, author_to_subjects,
                          subject_to_indices, embeddings, seed=0):
    """
    K-Means++ initialisation (Farthest Point Sampling).
    Guarantees waypoints are spread across the biological manifold.
    """
    rng = np.random.RandomState(seed)

    all_indices = []
    for author in train_authors:
        for subj_id in author_to_subjects[author]:
            all_indices.extend(subject_to_indices[subj_id])

    sample_size    = min(10000, len(all_indices))
    chosen_indices = rng.choice(all_indices, size=sample_size, replace=False)
    subset_embs    = torch.from_numpy(embeddings[chosen_indices].copy()).float()

    waypoints = torch.zeros(num_waypoints, EMBEDDING_DIM, dtype=torch.float32)
    first_idx  = rng.randint(0, sample_size)
    waypoints[0] = subset_embs[first_idx]

    min_dists = torch.full((sample_size,), float('inf'), dtype=torch.float32)

    for i in range(1, num_waypoints):
        latest_wp = waypoints[i - 1].unsqueeze(0)
        dists     = torch.cdist(subset_embs, latest_wp, p=2).squeeze()
        min_dists = torch.min(min_dists, dists)

        probs = (min_dists ** 2).numpy()
        if probs.sum() == 0:
            probs = np.ones_like(probs)
        probs /= probs.sum()

        next_idx     = rng.choice(sample_size, p=probs)
        waypoints[i] = subset_embs[next_idx]

    return waypoints


# ==============================================================================
# Single-fold training
# ==============================================================================

def train_fold(model, train_authors, val_authors, author_to_subjects, subject_to_indices,
               embeddings, n_seq_per_rep, lr, weight_decay, device, fold_idx,
               trial=None, seed=0):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = nn.MSELoss()
    rng       = np.random.RandomState(seed)

    best_val_mse = float('inf')
    patience_ctr = 0
    num_waypoints = model.waypoints.shape[0]

    train_authors = list(train_authors)
    val_authors   = list(val_authors)

    for epoch in range(MAX_EPOCHS):
        # --- Training pass ---
        model.train()
        rng.shuffle(train_authors)
        train_loss_sum = 0.0
        n_train_reps   = 0

        usage_counts = torch.zeros(num_waypoints, dtype=torch.long, device=device)

        for author in train_authors:
            subjects = list(author_to_subjects[author])
            rng.shuffle(subjects)

            for subject_id in subjects:
                x     = load_repertoire(subject_id, subject_to_indices, embeddings, n_seq_per_rep, rng).to(device)
                recon = model(x)
                loss  = loss_fn(recon, x)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                train_loss_sum += loss.item()
                n_train_reps   += 1

                with torch.no_grad():
                    dists              = torch.cdist(x, model.waypoints, p=2)
                    closest_waypoints  = dists.argmin(dim=1)
                    usage_counts      += torch.bincount(closest_waypoints, minlength=num_waypoints)

        train_mse = train_loss_sum / max(n_train_reps, 1)

        # --- Validation pass ---
        model.eval()
        val_loss_sum = 0.0
        n_val_reps   = 0

        with torch.no_grad():
            for author in val_authors:
                for subject_id in author_to_subjects[author]:
                    x     = load_repertoire(subject_id, subject_to_indices, embeddings, n_seq_per_rep, rng).to(device)
                    recon = model(x)
                    val_loss_sum += loss_fn(recon, x).item()
                    n_val_reps   += 1

        val_mse  = val_loss_sum / max(n_val_reps, 1)
        improved = val_mse < best_val_mse - EARLY_STOP_MIN_DELTA

        print(f"  [F{fold_idx} E{epoch+1:>3}] "
              f"train={train_mse:.6f}  val={val_mse:.6f}"
              + (" *" if improved else ""))

        if trial is not None:
            trial.report(val_mse, epoch)
            if trial.should_prune():
                print(f"  [F{fold_idx}] Pruning {epoch + 1}")
                raise optuna.TrialPruned()

        if improved:
            best_val_mse = val_mse
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                print(f"  [F{fold_idx}] Early stopped at epoch {epoch + 1}")
                break

    return best_val_mse


# ==============================================================================
# Optuna objective
# ==============================================================================

def objective(trial, author_to_subjects, subject_to_indices, embeddings, device):

    # temperature replaces power: log-scale 0.01–2.0
    # Low temperature (~0.01) → near one-hot / sharp; high (~2.0) → soft mix
    temperature   = trial.suggest_float('temperature', 0.01, 2.0, log=True)
    lr            = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    weight_decay  = 1e-6
    n_seq_per_rep = trial.suggest_int('n_seq_per_rep', MIN_SEQS_PER_REP, MAX_SEQS_PER_REP, step=50)

    print(f"\n{'='*70}")
    print(f"Trial {trial.number}: W={NUM_WAYPOINTS}  temperature={temperature:.4f}")
    print(f"  lr={lr:.2e}  wd={weight_decay:.2e}  n_seq={n_seq_per_rep}")
    print(f"{'='*70}")

    cv_seed     = trial.number * 7919 + 42
    all_authors = list(author_to_subjects.keys())
    splits      = create_author_kfold_splits(all_authors, n_splits=N_FOLDS, seed=cv_seed)

    fold_metrics = []

    for fold_idx, (train_authors, val_authors) in enumerate(splits):
        n_train_subj = sum(len(author_to_subjects[a]) for a in train_authors)
        n_val_subj   = sum(len(author_to_subjects[a]) for a in val_authors)

        print(f"\n  --- Fold {fold_idx} "
              f"(train: {len(train_authors)} studies / {n_train_subj} subjects | "
              f"val: {len(val_authors)} studies / {n_val_subj} subjects) ---")

        model = WaypointAutoencoder(num_waypoints=NUM_WAYPOINTS, temperature=temperature)

        # K-Means++ initialisation for waypoints only;
        # decoder starts at default nn.Linear kaiming_uniform init.
        with torch.no_grad():
            init_w = sample_init_waypoints(
                NUM_WAYPOINTS, train_authors, author_to_subjects,
                subject_to_indices, embeddings, seed=cv_seed + fold_idx
            )
            model.waypoints.data.copy_(init_w)

        model = model.to(device)

        best_val_mse = train_fold(
            model, train_authors, val_authors, author_to_subjects, subject_to_indices,
            embeddings, n_seq_per_rep, lr, weight_decay, device, fold_idx=fold_idx,
            trial=trial if fold_idx == 0 else None, seed=cv_seed + fold_idx * 13,
        )

        fold_metrics.append(best_val_mse)
        print(f"  Fold {fold_idx} best val MSE: {best_val_mse:.6f}")

        del model
        torch.cuda.empty_cache()

    if not fold_metrics:
        raise optuna.TrialPruned()

    mean_mse = float(np.mean(fold_metrics))
    std_mse  = float(np.std(fold_metrics))

    print(f"\n  Trial {trial.number}: mean_val_mse={mean_mse:.6f} ± {std_mse:.6f} "
          f"(folds: {[f'{m:.6f}' for m in fold_metrics]})")

    trial_row = {
        'trial':         trial.number,
        'mean_val_mse':  mean_mse,
        'std_val_mse':   std_mse,
        'num_waypoints': NUM_WAYPOINTS,
        'temperature':   temperature,
        'lr':            lr,
        'weight_decay':  weight_decay,
        'n_seq_per_rep': n_seq_per_rep,
        **{f'fold{i}_mse': m for i, m in enumerate(fold_metrics)},
    }
    pd.DataFrame([trial_row]).to_csv(
        RESULTS_CSV, mode='a', header=not os.path.exists(RESULTS_CSV), index=False
    )

    return mean_mse


# ==============================================================================
# Main
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running {N_TRIALS} trials, {N_FOLDS}-fold author-grouped CV")
    print(f"Metric: val reconstruction MSE (minimize)")
    print(f"Model: WaypointAutoencoder (softmax-temperature + decoupled decoder)")

    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    os.makedirs(STUDY_DIR, exist_ok=True)

    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)

    # --- Embeddings ---
    print("\n=== Loading Fold 1 Embeddings ===")
    fold1_emb_path, fold1_n = get_embedding_path(FOLD1_CSV, PREPROCESSED_DIR)

    print("\n=== Loading Fold 2 Embeddings ===")
    fold2_emb_path, fold2_n = get_embedding_path(FOLD2_CSV, PREPROCESSED_DIR)

    print(f"\nCombined: {fold1_n + fold2_n:,} sequences across both folds")

    # --- Author -> Subject Index ---
    author_to_subjects, subject_to_indices = build_nested_index(
        FOLD1_WEIGHTS_CSV, FOLD2_WEIGHTS_CSV, fold1_n, min_seqs=MIN_SEQS_PER_REP
    )

    # --- Combined accessor ---
    embeddings = CombinedEmbeddings(fold1_emb_path, fold2_emb_path)

    # --- Baseline MSE ---
    print("\n=== Calculating Dataset Variance (Baseline MSE) ===")
    sample_size    = min(100000, len(embeddings))
    rng_base       = np.random.RandomState(42)
    sample_indices = rng_base.choice(len(embeddings), size=sample_size, replace=False)
    sample_data    = embeddings[sample_indices]

    global_mean      = np.mean(sample_data, axis=0)
    dataset_variance = float(np.mean((sample_data - global_mean) ** 2))

    print(f"Dataset Variance: {dataset_variance:.6f}")
    print(f"  -> If MSE is ~{dataset_variance:.6f}, the model predicts the mean (0% variance explained).")
    print(f"  -> If MSE is ~{dataset_variance * 0.5:.6f}, the model explains 50% of the variance.")
    print(f"  -> Target 'Good' MSE (80%+ variance explained): < {dataset_variance * 0.2:.6f}\n")

    # --- Optuna study ---
    print("=== Starting Hyperparameter Search ===")

    study = optuna.create_study(
        study_name="waypoint-repertoire-reconstruction-v2",
        direction="minimize",
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
        pruner=MedianPruner(n_startup_trials=15, n_warmup_steps=10, interval_steps=1),
    )

    def objective_wrapper(trial):
        return objective(trial, author_to_subjects, subject_to_indices, embeddings, device)

    study.optimize(objective_wrapper, n_trials=N_TRIALS, show_progress_bar=False)

    # Rebuild results CSV from DB
    study.trials_dataframe().to_csv(RESULTS_CSV, index=False)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("WAYPOINT HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 70)

    print(f"\nBest trial:   {study.best_trial.number}")
    print(f"Best val MSE: {study.best_value:.6f}")

    if dataset_variance > 0:
        pct_explained = (1.0 - (study.best_value / dataset_variance)) * 100
        print(f"Variance Explained: {pct_explained:.2f}%")

    print("\nBest hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    best_config = {
        'best_trial':   study.best_trial.number,
        'best_val_mse': study.best_value,
        'model':        'WaypointAutoencoder_softmax_decoupled_decoder',
        'params':       study.best_trial.params,
    }
    with open(BEST_CONFIG_JSON, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"\nBest config saved to {BEST_CONFIG_JSON}")

    print("\nTop 10 trials (lowest val MSE):")
    print(f"{'Trial':>6} {'MSE':>10} {'W':>5} {'temp':>8} {'lr':>10} {'n_seq':>7} {'wd':>10}")
    print("-" * 70)

    trials_sorted = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value
    )
    for t in trials_sorted[:10]:
        p = t.params
        print(f"{t.number:>6} {t.value:>10.6f} "
              f"{NUM_WAYPOINTS:>5} "
              f"{p.get('temperature', 0):>8.4f} "
              f"{p.get('lr', 0):>10.2e} "
              f"{p.get('n_seq_per_rep', 0):>7} "
              f"{p.get('weight_decay', 1e-6):>10.2e}")

    try:
        importance = optuna.importance.get_param_importances(study)
        print("\nParameter importance:")
        for param, imp in importance.items():
            print(f"  {param}: {imp:.3f}")
    except Exception:
        pass

    print(f"\nFull results: {RESULTS_CSV}")
    print(f"Optuna DB:    {STUDY_DB}")
    print(f"Best config:  {BEST_CONFIG_JSON}")


if __name__ == "__main__":
    main()
