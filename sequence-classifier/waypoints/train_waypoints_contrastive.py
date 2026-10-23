#!/usr/bin/env python3
"""
Train Final Waypoints — Joint Reconstruction + Contrastive Loss

Loads the best hyperparameters from tune_waypoints_contrastive.py, trains a
final WaypointAutoencoder with joint reconstruction + NT-Xent contrastive loss
on the full combined fold1+fold2 dataset, and saves the resulting waypoints
for downstream repertoire featurisation.

A 90/10 author-stratified hold-out is used only for early stopping —
no labels are involved anywhere.

Joint loss:
    total = recon_loss + LAMBDA_CONTRASTIVE * contrastive_loss

LAMBDA_CONTRASTIVE is computed at startup (same method as tune script).

Outputs:
  waypoints.csv       — (W, 1024) waypoint matrix, columns hidden_state_0..1023
  checkpoint.pt       — full model state dict + metadata
  training_info.json  — epoch counts, final metrics, hyperparameters, lambda
"""

from bcrm.seed import seed_everything
from bcrm.paths import data_path, model_path, out_path, repo_path
import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

seed_everything(42)  # waypoint init, sampling, batch order


# ==============================================================================
# Configuration
# ==============================================================================

FOLD1_CSV         = out_path("embeddings/classifier_data_igblast_train_embeddings_weights.csv")
FOLD1_WEIGHTS_CSV = data_path("classifier/classifier_data_igblast-train-weights.csv")
FOLD2_CSV         = out_path("embeddings/classifier_data_igblast_test_embeddings_weights.csv")
FOLD2_WEIGHTS_CSV = data_path("classifier/classifier_data_igblast-test-weights.csv")

PREPROCESSED_DIR  = out_path("preprocessed")

BEST_CONFIG_JSON  = out_path("sequence-classifier/waypoint-contrastive-hparam-search/best_config_waypoints_contrastive.json")
OUTPUT_DIR        = out_path("sequence-classifier/waypoints-contrastive-final")

EMBEDDING_DIM     = 1024
AUTHOR_COL        = "Author"
SUBJECT_COL       = "Subject"

MIN_SEQS_PER_REP  = 100
MAX_SEQS_PER_REP  = 1000   # default if not in best_config

MAX_EPOCHS        = int(os.environ.get("BCRM_MAX_EPOCHS", 500))    # generous ceiling; early stopping controls actual count
EARLY_STOP_PATIENCE   = 15  # more patience than tuning — we want the best final model
EARLY_STOP_MIN_DELTA  = 1e-6

VAL_AUTHOR_FRAC   = 0.10    # fraction of authors held out for early stopping only
MAX_GRAD_NORM     = 1.0
PREPROCESS_CHUNK_SIZE = 50000

SUBJECTS_PER_BATCH = 32


# ==============================================================================
# Preprocessing
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


# ==============================================================================
# Combined mmap accessor
# ==============================================================================

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
# Author index
# ==============================================================================

def build_nested_index(fold1_weights_csv, fold2_weights_csv,
                       fold1_n, min_seqs=MIN_SEQS_PER_REP):
    """Build Author -> Subject -> indices mapping."""
    print("Building nested Author -> Subject index...")

    df1 = pd.read_csv(fold1_weights_csv, usecols=[AUTHOR_COL, SUBJECT_COL])
    df2 = pd.read_csv(fold2_weights_csv, usecols=[AUTHOR_COL, SUBJECT_COL])

    subject_to_indices = defaultdict(list)
    for idx, (author, subject) in enumerate(zip(df1[AUTHOR_COL], df1[SUBJECT_COL])):
        subject_to_indices[f"{author}::{subject}"].append(idx)
    for idx, (author, subject) in enumerate(zip(df2[AUTHOR_COL], df2[SUBJECT_COL])):
        subject_to_indices[f"{author}::{subject}"].append(fold1_n + idx)

    n_before = len(subject_to_indices)
    subject_to_indices = {s: v for s, v in subject_to_indices.items()
                          if len(v) >= min_seqs}
    n_after = len(subject_to_indices)

    author_to_subjects = defaultdict(list)
    for subj_id in subject_to_indices:
        author_to_subjects[subj_id.split("::")[0]].append(subj_id)
    author_to_subjects = {a: s for a, s in author_to_subjects.items() if s}

    sizes = [len(v) for v in subject_to_indices.values()]
    print(f"  {len(author_to_subjects)} authors  |  "
          f"{n_after} subjects retained ({n_before - n_after} dropped: <{min_seqs} seqs)")
    print(f"  Sequences: {sum(sizes):,}  "
          f"min={min(sizes):,}  median={int(np.median(sizes)):,}  max={max(sizes):,}")
    return author_to_subjects, subject_to_indices


# ==============================================================================
# Waypoint model
# ==============================================================================

class WaypointAutoencoder(nn.Module):
    """
    Softmax-temperature waypoint autoencoder with a decoupled linear decoder.
    Returns (recon, fingerprint) for contrastive loss computation.
    """

    def __init__(self, num_waypoints, temperature=0.5, input_dim=EMBEDDING_DIM):
        super().__init__()
        self.temperature = temperature
        self.waypoints = nn.Parameter(torch.randn(num_waypoints, input_dim))
        self.decoder = nn.Linear(num_waypoints, input_dim, bias=False)

    def forward(self, x):
        dists       = torch.cdist(x, self.waypoints, p=2)
        fingerprint = torch.softmax(-dists / self.temperature, dim=1)  # (N, W)
        recon       = self.decoder(fingerprint)
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
# Repertoire loading
# ==============================================================================

def load_repertoire(subject_id, subject_to_indices, embeddings, n_seq_per_rep, rng):
    indices = np.asarray(subject_to_indices[subject_id])
    if len(indices) > n_seq_per_rep:
        indices = rng.choice(indices, size=n_seq_per_rep, replace=False)
    return torch.from_numpy(embeddings[indices].copy()).float()


# ==============================================================================
# Data-sample initialization
# ==============================================================================

def sample_init_waypoints(num_waypoints, train_authors,
                           author_to_subjects, subject_to_indices, embeddings, seed=0):
    """K-Means++ initialization: farthest-point sampling over a 10K-sequence subset."""
    rng = np.random.RandomState(seed)

    all_indices = [
        idx
        for a in train_authors
        for subj_id in author_to_subjects[a]
        for idx in subject_to_indices[subj_id]
    ]

    sample_size    = min(10000, len(all_indices))
    chosen_indices = rng.choice(all_indices, size=sample_size, replace=False)
    subset_embs    = torch.from_numpy(embeddings[chosen_indices].copy()).float()

    waypoints = torch.zeros(num_waypoints, EMBEDDING_DIM, dtype=torch.float32)
    waypoints[0] = subset_embs[rng.randint(0, sample_size)]
    min_dists    = torch.full((sample_size,), float('inf'), dtype=torch.float32)

    for i in range(1, num_waypoints):
        latest_wp = waypoints[i - 1].unsqueeze(0)
        dists     = torch.cdist(subset_embs, latest_wp, p=2).squeeze(1)
        min_dists = torch.min(min_dists, dists)
        probs     = (min_dists ** 2).numpy()
        if probs.sum() == 0:
            probs = np.ones_like(probs)
        probs    /= probs.sum()
        waypoints[i] = subset_embs[rng.choice(sample_size, p=probs)]

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

    num_waypoints = 128
    model = WaypointAutoencoder(num_waypoints=num_waypoints, temperature=0.5).to(device)
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
# Training
# ==============================================================================

def train(model, train_authors, val_authors, author_to_subjects, subject_to_indices,
          embeddings, hparams, contrastive_temp, lambda_contrastive, device,
          checkpoint_path):

    lr            = hparams['lr']
    weight_decay  = hparams['weight_decay']
    n_seq_per_rep = hparams['n_seq_per_rep']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    loss_fn   = nn.MSELoss()
    rng       = np.random.RandomState(42)

    best_val_combined = float('inf')
    best_epoch        = 0
    patience_ctr      = 0

    # Collect all train/val subjects
    all_train_subjects = [s for a in train_authors for s in author_to_subjects[a]]
    all_val_subjects   = [s for a in val_authors   for s in author_to_subjects[a]]

    n_train_subj = len(all_train_subjects)
    n_val_subj   = len(all_val_subjects)
    print(f"  Train: {len(train_authors)} authors / {n_train_subj} subjects  |  "
          f"Val: {len(val_authors)} authors / {n_val_subj} subjects")
    print(f"  Epochs up to {MAX_EPOCHS}, patience={EARLY_STOP_PATIENCE}")

    for epoch in range(MAX_EPOCHS):
        # --- Train (batched per-subject gradient steps + contrastive) ---
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
                (lambda_contrastive * c_loss).backward()
                epoch_contrastive_sum += c_loss.item()
                n_c_batches += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        train_recon = epoch_recon_sum / max(n_recon, 1)
        train_contrastive = epoch_contrastive_sum / max(n_c_batches, 1)

        # --- Val ---
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
        val_combined = val_recon + lambda_contrastive * val_contrastive

        improved = val_combined < best_val_combined - EARLY_STOP_MIN_DELTA

        print(f"  Epoch {epoch+1:>4}  "
              f"train_r={train_recon:.6f} train_c={train_contrastive:.6f}  "
              f"val_r={val_recon:.6f} val_c={val_contrastive:.6f} "
              f"val_comb={val_combined:.6f}"
              + (" *" if improved else ""))

        if improved:
            best_val_combined = val_combined
            best_epoch        = epoch + 1
            patience_ctr      = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch':            best_epoch,
                'val_combined':     best_val_combined,
                'val_recon':        val_recon,
                'val_contrastive':  val_contrastive,
                'hparams':          hparams,
                'contrastive_temp': contrastive_temp,
                'lambda_contrastive': lambda_contrastive,
            }, checkpoint_path)
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                print(f"  Early stopped at epoch {epoch + 1} "
                      f"(best epoch {best_epoch})")
                break

    print(f"\n  Best val_combined: {best_val_combined:.6f} at epoch {best_epoch}")
    return best_val_combined, best_epoch


# ==============================================================================
# Main
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    if os.environ.get("BCRM_RETRAIN") != "1" and os.path.exists(os.path.join(OUTPUT_DIR, "waypoints.csv")):
        print(f"Using the existing waypoints in {OUTPUT_DIR} (set BCRM_RETRAIN=1 to retrain)")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load best hyperparameters ---
    print(f"\nLoading best config from {BEST_CONFIG_JSON}")
    with open(BEST_CONFIG_JSON) as f:
        best_config = json.load(f)

    params = best_config['params']
    hparams = {
        'num_waypoints': int(params.get('num_waypoints', 128)),
        'temperature':   float(params['temperature']),
        'lr':            float(params['lr']),
        'weight_decay':  float(params.get('weight_decay', 1e-6)),
        'n_seq_per_rep': int(params['n_seq_per_rep']),
        'init_strategy': params.get('init_strategy', 'data_sample'),
    }
    contrastive_temp = float(params['contrastive_temp'])

    print(f"  num_waypoints={hparams['num_waypoints']}  "
          f"temperature={hparams['temperature']:.4f}  "
          f"contrastive_temp={contrastive_temp:.4f}")
    print(f"  lr={hparams['lr']:.2e}  wd={hparams['weight_decay']:.2e}  "
          f"n_seq={hparams['n_seq_per_rep']}  init={hparams['init_strategy']}")
    print(f"  (tuning best val_combined: {best_config.get('best_val_combined', '?'):.6f})")

    # --- Embeddings ---
    print("\n=== Loading Fold 1 Embeddings ===")
    fold1_emb_path, fold1_n = get_embedding_path(FOLD1_CSV, PREPROCESSED_DIR)

    print("\n=== Loading Fold 2 Embeddings ===")
    fold2_emb_path, fold2_n = get_embedding_path(FOLD2_CSV, PREPROCESSED_DIR)

    print(f"\nCombined: {fold1_n + fold2_n:,} sequences")
    embeddings = CombinedEmbeddings(fold1_emb_path, fold2_emb_path)

    # --- Nested index ---
    author_to_subjects, subject_to_indices = build_nested_index(
        FOLD1_WEIGHTS_CSV, FOLD2_WEIGHTS_CSV,
        fold1_n, min_seqs=MIN_SEQS_PER_REP
    )

    # --- Baseline Lambda ---
    lambda_contrastive = compute_baseline_lambda(
        subject_to_indices, embeddings, device
    )

    # --- Train/val split (for early stopping only) ---
    rng         = np.random.RandomState(0)
    all_authors = list(author_to_subjects.keys())
    rng.shuffle(all_authors)
    n_val         = max(1, int(len(all_authors) * VAL_AUTHOR_FRAC))
    val_authors   = set(all_authors[:n_val])
    train_authors = set(all_authors[n_val:])

    print(f"\nTrain/val split ({int(100*(1-VAL_AUTHOR_FRAC))}%/{int(100*VAL_AUTHOR_FRAC)}%):")
    print(f"  Train: {len(train_authors)} authors")
    print(f"  Val:   {len(val_authors)} authors  (early stopping only)")

    # --- Build model ---
    model = WaypointAutoencoder(
        num_waypoints=hparams['num_waypoints'],
        temperature=hparams['temperature'],
    )

    if hparams['init_strategy'] == 'data_sample':
        print("\nInitializing waypoints from data samples (K-Means++)...")
        with torch.no_grad():
            init_w = sample_init_waypoints(
                hparams['num_waypoints'], train_authors,
                author_to_subjects, subject_to_indices, embeddings, seed=0
            )
            model.waypoints.data.copy_(init_w)
    else:
        print("\nInitializing waypoints from random normal...")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {hparams['num_waypoints']} waypoints  |  {n_params:,} parameters")
    print(f"LAMBDA_CONTRASTIVE: {lambda_contrastive:.6f}")

    # --- Train ---
    print("\n=== Training ===")
    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint.pt")

    best_val_combined, best_epoch = train(
        model, train_authors, val_authors,
        author_to_subjects, subject_to_indices,
        embeddings, hparams, contrastive_temp, lambda_contrastive,
        device, checkpoint_path
    )

    # --- Load best checkpoint and save waypoints ---
    print(f"\nLoading best checkpoint (epoch {best_epoch})...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    waypoints = model.waypoints.detach().cpu().numpy()  # (W, 1024)

    waypoints_csv = os.path.join(OUTPUT_DIR, "waypoints.csv")
    cols = [f"hidden_state_{i}" for i in range(EMBEDDING_DIM)]
    pd.DataFrame(waypoints, columns=cols).to_csv(waypoints_csv, index=False)
    print(f"Waypoints saved to {waypoints_csv}  (shape: {waypoints.shape})")

    # Save training metadata
    training_info = {
        'hparams':              hparams,
        'contrastive_temp':     contrastive_temp,
        'lambda_contrastive':   lambda_contrastive,
        'best_val_combined':    best_val_combined,
        'best_val_recon':       ckpt.get('val_recon'),
        'best_val_contrastive': ckpt.get('val_contrastive'),
        'best_epoch':           best_epoch,
        'n_train_authors':      len(train_authors),
        'n_val_authors':        len(val_authors),
        'n_total_authors':      len(all_authors),
        'waypoints_shape':      list(waypoints.shape),
        'waypoints_csv':        'waypoints.csv',   # relative to the output folder
    }
    info_path = os.path.join(OUTPUT_DIR, "training_info.json")
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"Training info saved to {info_path}")

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"  Waypoints:      {waypoints_csv}")
    print(f"  Checkpoint:     {checkpoint_path}")
    print(f"  Best val_combined: {best_val_combined:.6f}  (epoch {best_epoch})")
    print(f"  LAMBDA_CONTRASTIVE: {lambda_contrastive:.6f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
