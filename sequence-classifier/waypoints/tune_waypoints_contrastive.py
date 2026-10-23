#!/usr/bin/env python3
"""
Waypoint Hyperparameter Tuning — Joint Reconstruction + Contrastive Loss

Extends the reconstruction-only waypoint autoencoder with NT-Xent contrastive
loss on subject-level mean fingerprints.  The contrastive term encourages
waypoints to produce fingerprints that are discriminative across subjects,
while reconstruction loss anchors them to the data manifold.

Joint loss:
    total = recon_loss + LAMBDA_CONTRASTIVE * contrastive_loss

LAMBDA_CONTRASTIVE is computed once at startup from a random-init model
(BASELINE_RECON / BASELINE_CONTRASTIVE) and fixed for all trials.

Search space (4 params):
  - temperature:      waypoint softmax temperature (log 0.01–2.0)
  - lr:               learning rate (log 1e-4–1e-1)
  - n_seq_per_rep:    sequences per repertoire (50–1000, step 50)
  - contrastive_temp: NT-Xent temperature (log 0.05–1.0)

Data Strategy:
  - K-Fold CV split by Author (prevents batch-effect leakage).
  - Training by Subject in batches of SUBJECTS_PER_BATCH.
  - Validation metric: val_combined = val_recon + lambda * val_contrastive.
"""

from bcrm.paths import data_path, model_path, out_path, repo_path
import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

STUDY_DIR         = out_path("sequence-classifier/waypoint-contrastive-hparam-search")
STUDY_DB          = os.path.join(STUDY_DIR, "optuna_waypoints_contrastive.db")
RESULTS_CSV       = os.path.join(STUDY_DIR, "trial_results_waypoints_contrastive.csv")
BEST_CONFIG_JSON  = os.path.join(STUDY_DIR, "best_config_waypoints_contrastive.json")

EMBEDDING_DIM      = 1024
AUTHOR_COL         = "Author"
SUBJECT_COL        = "Subject"

MIN_SEQS_PER_REP   = 50
MAX_SEQS_PER_REP   = 1000

N_FOLDS            = 5
N_TRIALS           = int(os.environ.get("BCRM_TUNE_TRIALS", 200))  # lower it for smoke tests

MAX_EPOCHS            = 200
EARLY_STOP_PATIENCE   = 5
EARLY_STOP_MIN_DELTA  = 1e-6

MAX_GRAD_NORM         = 1.0
PREPROCESS_CHUNK_SIZE = 50000

NUM_WAYPOINTS      = 128
SUBJECTS_PER_BATCH = 32

# Module-level; set once in main() before Optuna
LAMBDA_CONTRASTIVE = 1.0


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
    Returns (recon, fingerprint) so the contrastive loss can use the fingerprint.
    """

    def __init__(self, num_waypoints, temperature=0.5, input_dim=EMBEDDING_DIM):
        super().__init__()
        self.temperature = temperature
        self.waypoints = nn.Parameter(torch.randn(num_waypoints, input_dim))
        self.decoder = nn.Linear(num_waypoints, input_dim, bias=False)

    def forward(self, x):
        dists = torch.cdist(x, self.waypoints, p=2)
        fingerprint = torch.softmax(-dists / self.temperature, dim=1)  # (N, W)
        recon = self.decoder(fingerprint)
        return recon, fingerprint


# ==============================================================================
# Contrastive Loss
# ==============================================================================

def contrastive_fingerprint_loss(model, subjects_data, contrastive_temperature):
    """
    NT-Xent on L2-normalised mean fingerprints from split halves.

    For each subject, split its sequences into two random halves, compute
    the mean fingerprint of each half, L2-normalise, then apply InfoNCE
    across all subjects in the batch.

    Returns (loss, B) where B = number of usable subjects.
    """
    fps_a, fps_b = [], []
    for x in subjects_data:
        n = x.size(0)
        if n < 4:
            continue
        half = n // 2
        perm = torch.randperm(n, device=x.device)
        _, fp_a = model(x[perm[:half]])
        _, fp_b = model(x[perm[half:2 * half]])
        fps_a.append(fp_a.mean(dim=0))
        fps_b.append(fp_b.mean(dim=0))

    if len(fps_a) < 2:
        return torch.tensor(0.0, device=subjects_data[0].device, requires_grad=True), 0

    za = F.normalize(torch.stack(fps_a), dim=1)
    zb = F.normalize(torch.stack(fps_b), dim=1)
    B = za.size(0)

    z = torch.cat([za, zb], dim=0)                          # (2B, W)
    sim_matrix = z @ z.T / contrastive_temperature           # (2B, 2B)
    mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)

    positive_indices = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])

    pos_sim = sim_matrix[torch.arange(2 * B, device=z.device), positive_indices]
    denom = (sim_matrix * mask.float()).exp().sum(dim=1)
    loss = (-pos_sim + torch.log(denom)).mean()

    return loss, B


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
    """K-Means++ initialisation (Farthest Point Sampling)."""
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
# Baseline Lambda Computation
# ==============================================================================

def compute_baseline_lambda(subject_to_indices, embeddings, device):
    """
    Compute LAMBDA_CONTRASTIVE = BASELINE_RECON / BASELINE_CONTRASTIVE
    using a random-init model on a random batch of subjects.
    """
    print("\n=== Computing Baseline Lambda ===")

    model = WaypointAutoencoder(num_waypoints=NUM_WAYPOINTS, temperature=0.5).to(device)
    model.eval()

    all_subjects = list(subject_to_indices.keys())
    rng = np.random.RandomState(999)
    rng.shuffle(all_subjects)
    sample_subjects = all_subjects[:SUBJECTS_PER_BATCH]

    subjects_data = []
    recon_losses = []
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        for subj_id in sample_subjects:
            x = load_repertoire(subj_id, subject_to_indices, embeddings,
                                MAX_SEQS_PER_REP, rng).to(device)
            recon, fp = model(x)
            recon_losses.append(loss_fn(recon, x).item())
            subjects_data.append(x)

        c_loss, B = contrastive_fingerprint_loss(model, subjects_data, 0.1)

    baseline_recon = float(np.mean(recon_losses))
    baseline_contrastive = c_loss.item() if B > 0 else 1.0

    lambda_val = baseline_recon / baseline_contrastive

    print(f"  BASELINE_RECON:        {baseline_recon:.6f}")
    print(f"  BASELINE_CONTRASTIVE:  {baseline_contrastive:.6f}")
    print(f"  LAMBDA_CONTRASTIVE:    {lambda_val:.6f}")

    del model
    torch.cuda.empty_cache()
    return lambda_val


# ==============================================================================
# Single-fold training
# ==============================================================================

def train_fold(model, train_authors, val_authors, author_to_subjects, subject_to_indices,
               embeddings, n_seq_per_rep, lr, weight_decay, contrastive_temp, device,
               fold_idx, trial=None, seed=0):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = nn.MSELoss()
    rng       = np.random.RandomState(seed)

    best_val_combined = float('inf')
    patience_ctr = 0

    # Collect all train/val subjects
    all_train_subjects = [s for a in train_authors for s in author_to_subjects[a]]
    all_val_subjects   = [s for a in val_authors   for s in author_to_subjects[a]]

    for epoch in range(MAX_EPOCHS):
        # --- Training pass ---
        model.train()
        rng.shuffle(all_train_subjects)

        epoch_recon_sum = 0.0
        epoch_contrastive_sum = 0.0
        n_recon = 0
        n_c_batches = 0

        for batch_start in range(0, len(all_train_subjects), SUBJECTS_PER_BATCH):
            batch_subjects = all_train_subjects[batch_start:batch_start + SUBJECTS_PER_BATCH]

            optimizer.zero_grad()

            # (a) Per-subject reconstruction loss
            subjects_data = []
            for subject_id in batch_subjects:
                x = load_repertoire(subject_id, subject_to_indices, embeddings,
                                    n_seq_per_rep, rng).to(device)
                recon, fp = model(x)
                rloss = loss_fn(recon, x)
                (rloss / len(batch_subjects)).backward()
                epoch_recon_sum += rloss.item()
                n_recon += 1
                subjects_data.append(x)

            # (b) Batch-wide contrastive loss
            c_loss, B = contrastive_fingerprint_loss(model, subjects_data, contrastive_temp)
            if B > 0:
                (LAMBDA_CONTRASTIVE * c_loss).backward()
                epoch_contrastive_sum += c_loss.item()
                n_c_batches += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        train_recon = epoch_recon_sum / max(n_recon, 1)
        train_contrastive = epoch_contrastive_sum / max(n_c_batches, 1)

        # --- Validation pass ---
        model.eval()
        val_recon_sum = 0.0
        val_contrastive_sum = 0.0
        n_val = 0
        n_val_c = 0

        with torch.no_grad():
            for batch_start in range(0, len(all_val_subjects), SUBJECTS_PER_BATCH):
                batch_subjects = all_val_subjects[batch_start:batch_start + SUBJECTS_PER_BATCH]
                subjects_data = []

                for subject_id in batch_subjects:
                    x = load_repertoire(subject_id, subject_to_indices, embeddings,
                                        n_seq_per_rep, rng).to(device)
                    recon, fp = model(x)
                    val_recon_sum += loss_fn(recon, x).item()
                    n_val += 1
                    subjects_data.append(x)

                c_loss, B = contrastive_fingerprint_loss(model, subjects_data, contrastive_temp)
                if B > 0:
                    val_contrastive_sum += c_loss.item()
                    n_val_c += 1

        val_recon = val_recon_sum / max(n_val, 1)
        val_contrastive = val_contrastive_sum / max(n_val_c, 1)
        val_combined = val_recon + LAMBDA_CONTRASTIVE * val_contrastive

        improved = val_combined < best_val_combined - EARLY_STOP_MIN_DELTA

        print(f"  [F{fold_idx} E{epoch+1:>3}] "
              f"train_r={train_recon:.6f} train_c={train_contrastive:.6f}  "
              f"val_r={val_recon:.6f} val_c={val_contrastive:.6f} "
              f"val_comb={val_combined:.6f}"
              + (" *" if improved else ""))

        if trial is not None:
            trial.report(val_combined, epoch)
            if trial.should_prune():
                print(f"  [F{fold_idx}] Pruning at epoch {epoch + 1}")
                raise optuna.TrialPruned()

        if improved:
            best_val_combined = val_combined
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                print(f"  [F{fold_idx}] Early stopped at epoch {epoch + 1}")
                break

    return best_val_combined


# ==============================================================================
# Optuna objective
# ==============================================================================

def objective(trial, author_to_subjects, subject_to_indices, embeddings, device):

    temperature       = trial.suggest_float('temperature', 0.01, 2.0, log=True)
    lr                = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    n_seq_per_rep     = trial.suggest_int('n_seq_per_rep', MIN_SEQS_PER_REP, MAX_SEQS_PER_REP, step=50)
    contrastive_temp  = trial.suggest_float('contrastive_temp', 0.05, 1.0, log=True)
    weight_decay      = 1e-6

    print(f"\n{'='*70}")
    print(f"Trial {trial.number}: W={NUM_WAYPOINTS}  temp={temperature:.4f}  "
          f"c_temp={contrastive_temp:.4f}")
    print(f"  lr={lr:.2e}  wd={weight_decay:.2e}  n_seq={n_seq_per_rep}")
    print(f"  LAMBDA_CONTRASTIVE={LAMBDA_CONTRASTIVE:.6f}")
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

        with torch.no_grad():
            init_w = sample_init_waypoints(
                NUM_WAYPOINTS, train_authors, author_to_subjects,
                subject_to_indices, embeddings, seed=cv_seed + fold_idx
            )
            model.waypoints.data.copy_(init_w)

        model = model.to(device)

        best_val = train_fold(
            model, train_authors, val_authors, author_to_subjects, subject_to_indices,
            embeddings, n_seq_per_rep, lr, weight_decay, contrastive_temp, device,
            fold_idx=fold_idx,
            trial=trial if fold_idx == 0 else None,
            seed=cv_seed + fold_idx * 13,
        )

        fold_metrics.append(best_val)
        print(f"  Fold {fold_idx} best val_combined: {best_val:.6f}")

        del model
        torch.cuda.empty_cache()

    if not fold_metrics:
        raise optuna.TrialPruned()

    mean_val = float(np.mean(fold_metrics))
    std_val  = float(np.std(fold_metrics))

    print(f"\n  Trial {trial.number}: mean_val_combined={mean_val:.6f} ± {std_val:.6f} "
          f"(folds: {[f'{m:.6f}' for m in fold_metrics]})")

    trial_row = {
        'trial':              trial.number,
        'mean_val_combined':  mean_val,
        'std_val_combined':   std_val,
        'temperature':        temperature,
        'contrastive_temp':   contrastive_temp,
        'lr':                 lr,
        'n_seq_per_rep':      n_seq_per_rep,
        'weight_decay':       weight_decay,
        'lambda_contrastive': LAMBDA_CONTRASTIVE,
        **{f'fold{i}_combined': m for i, m in enumerate(fold_metrics)},
    }
    pd.DataFrame([trial_row]).to_csv(
        RESULTS_CSV, mode='a', header=not os.path.exists(RESULTS_CSV), index=False
    )

    return mean_val


# ==============================================================================
# Main
# ==============================================================================

def main():
    global LAMBDA_CONTRASTIVE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running {N_TRIALS} trials, {N_FOLDS}-fold author-grouped CV")
    print(f"Metric: val_combined = recon + lambda * contrastive (minimize)")
    print(f"Model: WaypointAutoencoder (softmax-temperature + contrastive)")

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

    # --- Combined accessor (loaded into RAM) ---
    embeddings = CombinedEmbeddings(fold1_emb_path, fold2_emb_path)

    # --- Baseline Lambda ---
    LAMBDA_CONTRASTIVE = compute_baseline_lambda(
        subject_to_indices, embeddings, device
    )

    # --- Optuna study ---
    print("\n=== Starting Hyperparameter Search ===")

    study = optuna.create_study(
        study_name="waypoint-contrastive-v1",
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
    print("CONTRASTIVE WAYPOINT HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 70)

    print(f"\nBest trial:         {study.best_trial.number}")
    print(f"Best val_combined:  {study.best_value:.6f}")
    print(f"LAMBDA_CONTRASTIVE: {LAMBDA_CONTRASTIVE:.6f}")

    print("\nBest hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    best_config = {
        'best_trial':         study.best_trial.number,
        'best_val_combined':  study.best_value,
        'lambda_contrastive': LAMBDA_CONTRASTIVE,
        'model':              'WaypointAutoencoder_contrastive',
        'params':             study.best_trial.params,
    }
    with open(BEST_CONFIG_JSON, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"\nBest config saved to {BEST_CONFIG_JSON}")

    print(f"\nTop 10 trials (lowest val_combined):")
    print(f"{'Trial':>6} {'Combined':>10} {'temp':>8} {'c_temp':>8} {'lr':>10} {'n_seq':>7}")
    print("-" * 60)

    trials_sorted = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value
    )
    for t in trials_sorted[:10]:
        p = t.params
        print(f"{t.number:>6} {t.value:>10.6f} "
              f"{p.get('temperature', 0):>8.4f} "
              f"{p.get('contrastive_temp', 0):>8.4f} "
              f"{p.get('lr', 0):>10.2e} "
              f"{p.get('n_seq_per_rep', 0):>7}")

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
