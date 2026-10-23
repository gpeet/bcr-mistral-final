#!/usr/bin/env python3
"""
Hyperparameter Tuning: Label-Exclusion Variant

Replaces soft-relabeling with a hard exclusion mechanism that progressively
drops MS-labeled samples the model confidently predicts as Other.

Fixed architecture: [256, 128, 64].
Tunable: dropout, lr, weight_decay, label_smoothing, flip_rate schedule.
Optimizes AUGC; logs lift@0.2 as secondary metric.
"""

from bcrm.paths import data_path, model_path, out_path, repo_path
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from collections import Counter, defaultdict
import optuna
from optuna.pruners import MedianPruner
import json
import time

# ==============================================================================
# Configuration
# ==============================================================================

TRAIN_CSV = out_path("embeddings/classifier_data_igblast_train_embeddings_weights.csv")
TRAIN_WEIGHTS_CSV = data_path("classifier/classifier_data_igblast-train-weights.csv")
TEST_CSV = out_path("embeddings/classifier_data_igblast_test_embeddings_weights.csv")
TEST_WEIGHTS_CSV = data_path("classifier/classifier_data_igblast-test-weights.csv")

PREPROCESSED_DIR = out_path("preprocessed")
SPLITS_DIR = os.path.join(PREPROCESSED_DIR, "cv_splits")

# Outputs
STUDY_DIR = out_path("sequence-classifier/hparam-search-exclusion")

EMBEDDING_DIM = 1024
LABEL_COL = "Disease"
WEIGHT_COL = "weight"
AUTHOR_COL = "Author"

ORIGINAL_LABELS = ['Autoimmune Disease', 'Healthy', 'Infectious Disease', 'Multiple sclerosis']
MS_LABEL = 'Multiple sclerosis'
BINARY_LABELS = ['Multiple sclerosis', 'Other']

N_FOLDS = 3
N_TRIALS = int(os.environ.get("BCRM_TUNE_TRIALS", 200))  # lower it for smoke tests
PREPROCESS_CHUNK_SIZE = 50000

MAX_STEPS_PER_FOLD = 20000
VAL_INTERVAL = 150
VAL_SAMPLES = 16384
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.005

NUM_WORKERS = 8
MAX_GRAD_NORM = 0.5

GAIN_FRACTIONS = [0.01, 0.05, 0.1, 0.2]
TOP_LIFT_FRACTIONS = [0.2]


# ==============================================================================
# Balanced Cross-Entropy Loss with Label Smoothing
# ==============================================================================

class BalancedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy with per-sample weights, optional label smoothing, and
    50/50 class-balanced averaging (matching the original tuning script's
    loss structure).
    """
    def __init__(self, other_id, ms_id, label_smoothing=0.0):
        super().__init__()
        self.other_id = other_id
        self.ms_id = ms_id
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, sample_weights=None):
        device = logits.device
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        if batch_size == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if sample_weights is None:
            sample_weights = torch.ones(batch_size, device=device)

        log_probs = F.log_softmax(logits, dim=1)

        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(logits).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smooth_targets = ((1.0 - self.label_smoothing) * smooth_targets
                              + self.label_smoothing / num_classes)
            per_sample_loss = -(smooth_targets * log_probs).sum(dim=1)
        else:
            per_sample_loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        weighted_loss = per_sample_loss * sample_weights

        other_mask = targets == self.other_id
        ms_mask = targets == self.ms_id
        n_other = other_mask.sum().item()
        n_ms = ms_mask.sum().item()

        other_mean = (weighted_loss[other_mask].mean()
                      if n_other > 0 else torch.tensor(0.0, device=device))
        ms_mean = (weighted_loss[ms_mask].mean()
                   if n_ms > 0 else torch.tensor(0.0, device=device))

        return 0.5 * other_mean + 0.5 * ms_mean


# ==============================================================================
# Gain Curve
# ==============================================================================

def compute_gain_curve(ms_scores, labels_binary, weights=None, fractions=None):
    if fractions is None:
        fractions = GAIN_FRACTIONS

    n = len(ms_scores)
    if n == 0:
        return {'augc': 0.0, 'top20_augc': 0.0, 'mean_top_lift': 0.0}

    sorted_indices = np.argsort(-ms_scores)
    sorted_labels = labels_binary[sorted_indices]

    if weights is not None:
        sorted_weights = weights[sorted_indices]
    else:
        sorted_weights = np.ones(n, dtype=np.float32)

    total_ms_weight = (sorted_labels * sorted_weights).sum()
    total_weight = sorted_weights.sum()
    baseline_rate = total_ms_weight / (total_weight + 1e-8)

    results = {}

    for frac in fractions:
        k = max(1, int(n * frac))
        top_k_labels = sorted_labels[:k]
        top_k_weights = sorted_weights[:k]

        if top_k_weights.sum() > 0:
            precision = (top_k_labels * top_k_weights).sum() / top_k_weights.sum()
        else:
            precision = 0.0

        lift = precision / (baseline_rate + 1e-8)
        results[f'precision@{frac}'] = float(precision)
        results[f'lift@{frac}'] = float(lift)

    cum_ms = np.cumsum(sorted_labels * sorted_weights)
    cum_samples = np.arange(1, n + 1, dtype=np.float32)
    cum_ms_frac = cum_ms / (total_ms_weight + 1e-8)
    cum_samples_frac = cum_samples / n

    augc = np.trapz(cum_ms_frac, cum_samples_frac) - 0.5
    results['augc'] = float(augc)

    cutoff = max(1, int(n * 0.20))
    partial_cum_ms_frac = cum_ms_frac[:cutoff]
    partial_cum_samples_frac = cum_samples_frac[:cutoff]
    partial_area = np.trapz(partial_cum_ms_frac, partial_cum_samples_frac)
    partial_diagonal = 0.5 * (partial_cum_samples_frac[-1] ** 2)
    results['top20_augc'] = float(partial_area - partial_diagonal)

    top_fracs = [f for f in fractions if f in TOP_LIFT_FRACTIONS]
    if top_fracs:
        mean_top_lift = np.mean([results[f'lift@{f}'] for f in top_fracs])
    else:
        mean_top_lift = 0.0
    results['mean_top_lift'] = float(mean_top_lift)
    results['baseline_rate'] = float(baseline_rate)

    return results


# ==============================================================================
# Model
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
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==============================================================================
# Early Stopping
# ==============================================================================

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


# ==============================================================================
# Preprocessing
# ==============================================================================

def preprocess_csv_to_mmap(csv_path, weights_csv_path, output_prefix,
                           chunk_size=PREPROCESS_CHUNK_SIZE):
    print(f"\nPreprocessing {csv_path}...")
    os.makedirs(os.path.dirname(output_prefix) or '.', exist_ok=True)

    embeddings_path = f"{output_prefix}_embeddings.npy"
    labels_path = f"{output_prefix}_labels.npy"
    weights_path = f"{output_prefix}_weights.npy"
    embedding_cols = [f"hidden_state_{i}" for i in range(EMBEDDING_DIM)]

    # Single-pass: collect chunks then write mmap with correct size
    emb_arrays = []
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, usecols=embedding_cols)):
        emb_arrays.append(chunk.values.astype(np.float32))
        print(f"    Embeddings chunk {i+1}: {sum(a.shape[0] for a in emb_arrays):,} rows")
    all_emb = np.concatenate(emb_arrays)
    del emb_arrays
    n_rows = all_emb.shape[0]
    print(f"  Total rows: {n_rows:,}")

    embeddings_mmap = np.lib.format.open_memmap(
        embeddings_path, mode='w+', dtype=np.float32, shape=(n_rows, EMBEDDING_DIM)
    )
    embeddings_mmap[:] = all_emb
    del all_emb, embeddings_mmap

    # Single-pass weights + labels
    wt_arrays, labels_list = [], []
    for chunk in pd.read_csv(weights_csv_path, chunksize=chunk_size,
                             usecols=[WEIGHT_COL, LABEL_COL]):
        wt_arrays.append(chunk[WEIGHT_COL].values.astype(np.float32))
        labels_list.extend(chunk[LABEL_COL].values.tolist())
    all_wt = np.concatenate(wt_arrays)
    del wt_arrays
    assert all_wt.shape[0] == n_rows, (
        f"Weights CSV has {all_wt.shape[0]} rows but embeddings has {n_rows}"
    )

    weights_mmap = np.lib.format.open_memmap(
        weights_path, mode='w+', dtype=np.float32, shape=(n_rows,)
    )
    weights_mmap[:] = all_wt
    del all_wt, weights_mmap
    np.save(labels_path, np.array(labels_list, dtype=object))
    return embeddings_path, labels_path, weights_path


def get_preprocessed_paths(csv_path, weights_csv_path, preprocessed_dir):
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_prefix = os.path.join(preprocessed_dir, base_name)

    embeddings_path = f"{output_prefix}_embeddings.npy"
    labels_path = f"{output_prefix}_labels.npy"
    weights_path = f"{output_prefix}_weights.npy"

    if all(os.path.exists(p) for p in [embeddings_path, labels_path, weights_path]):
        print(f"Found preprocessed files for {base_name}")
        return embeddings_path, labels_path, weights_path

    return preprocess_csv_to_mmap(csv_path, weights_csv_path, output_prefix)


# ==============================================================================
# Dataset
# ==============================================================================

class MMapEmbeddingDataset(Dataset):
    def __init__(self, embeddings_path, labels_path, weights_path, label_encoder,
                 ms_label='Multiple sclerosis', indices=None, quiet=False):
        self.embeddings = np.load(embeddings_path, mmap_mode='r')
        self.weights = np.load(weights_path, mmap_mode='r')
        self.labels_raw = np.load(labels_path, allow_pickle=True)
        self.label_encoder = label_encoder

        if indices is not None:
            self.indices = np.array(indices)
            self.labels_raw_subset = self.labels_raw[self.indices]
        else:
            self.indices = np.arange(len(self.labels_raw))
            self.labels_raw_subset = self.labels_raw

        binary_labels = np.where(self.labels_raw_subset == ms_label, ms_label, 'Other')
        self.labels = label_encoder.transform(binary_labels)

        if not quiet:
            bin_unique, bin_counts = np.unique(self.labels, return_counts=True)
            for u, c in zip(bin_unique, bin_counts):
                print(f"    {label_encoder.classes_[u]}: {c:,} ({100*c/len(self.labels):.1f}%)")
            print(f"    Total: {len(self):,} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        embedding = torch.from_numpy(self.embeddings[real_idx].copy()).float()
        label = self.labels[idx]
        weight = torch.tensor(self.weights[real_idx], dtype=torch.float32)
        return embedding, label, weight


# ==============================================================================
# Author-Grouped K-Fold Splits
# ==============================================================================

def create_author_kfold_splits(weights_csv_path, n_splits=3, seed=None):
    if seed is None:
        seed = random.randint(0, 2**31)

    df = pd.read_csv(weights_csv_path, usecols=[LABEL_COL, AUTHOR_COL])
    authors = df[AUTHOR_COL].values

    author_labels = df.groupby(AUTHOR_COL)[LABEL_COL].first().to_dict()

    ms_authors = [a for a, l in author_labels.items() if l == MS_LABEL]
    other_authors = [a for a, l in author_labels.items() if l != MS_LABEL]

    rng = np.random.RandomState(seed)
    rng.shuffle(ms_authors)   # randomizes tie-breaking before sort
    rng.shuffle(other_authors)
    
    ms_author_counts = {a: (df[AUTHOR_COL] == a).sum() for a in ms_authors}
    ms_authors_sorted = sorted(ms_authors, key=lambda a: ms_author_counts[a], reverse=True)
    
    fold_authors = defaultdict(list)
    ms_fold_seq_counts = {i: 0 for i in range(n_splits)}
    for author in ms_authors_sorted:
        min_fold = min(ms_fold_seq_counts, key=ms_fold_seq_counts.get)
        fold_authors[min_fold].append(author)
        ms_fold_seq_counts[min_fold] += ms_author_counts[author]

    other_author_counts = {a: (df[AUTHOR_COL] == a).sum() for a in other_authors}
    other_authors_sorted = sorted(other_authors,
                                  key=lambda a: other_author_counts[a], reverse=True)

    fold_seq_counts = {i: 0 for i in range(n_splits)}
    for author in other_authors_sorted:
        min_fold = min(fold_seq_counts, key=fold_seq_counts.get)
        fold_authors[min_fold].append(author)
        fold_seq_counts[min_fold] += other_author_counts[author]

    author_to_indices = defaultdict(list)
    for idx, author in enumerate(authors):
        author_to_indices[author].append(idx)

    splits = []
    for val_fold in range(n_splits):
        val_authors = set(fold_authors[val_fold])
        train_authors = set()
        for fold_idx in range(n_splits):
            if fold_idx != val_fold:
                train_authors.update(fold_authors[fold_idx])

        val_indices = []
        for author in val_authors:
            val_indices.extend(author_to_indices[author])
        train_indices = []
        for author in train_authors:
            train_indices.extend(author_to_indices[author])

        splits.append((np.array(sorted(train_indices)), np.array(sorted(val_indices))))

    return splits


# ==============================================================================
# Flip Rate Schedule
# ==============================================================================

def compute_flip_rate(step, flip_rate_init, flip_rate_final, flip_rate_delta):
    """
    Compute the current exclusion rate.

    flip_rate = fraction of conflicting MS samples to DROP from the loss.
      0.0 = keep all (standard training)
      1.0 = drop all conflicting MS samples

    Rate moves from init toward final at delta per step, clamped appropriately.
    """
    raw = flip_rate_init + flip_rate_delta * step

    if flip_rate_delta >= 0:
        # Increasing: clamp to [init, final]
        return min(raw, flip_rate_final)
    else:
        # Decreasing: clamp to [final, init]
        return max(raw, flip_rate_final)


# ==============================================================================
# Label Exclusion Step
# ==============================================================================

def apply_label_exclusion(outputs, labels, case_weights, other_id, ms_id, flip_rate):
    """
    Hard exclusion: among MS-labeled samples that the model currently predicts
    as Other, drop a fraction (flip_rate) of the most confident ones from the
    loss computation.

    Returns filtered (outputs, labels, weights) and exclusion stats dict.
    """
    device = outputs.device
    batch_size = outputs.shape[0]

    if batch_size == 0 or flip_rate <= 0.0:
        return outputs, labels, case_weights, {
            'n_conflicting': 0, 'n_excluded': 0, 'n_remaining': batch_size
        }

    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)
        # MS-labeled samples that the model predicts as Other
        conflicting = (preds == other_id) & (labels == ms_id)
        n_conflicting = conflicting.sum().item()

    if n_conflicting <= 1:
        return outputs, labels, case_weights, {
            'n_conflicting': n_conflicting, 'n_excluded': 0,
            'n_remaining': batch_size
        }

    with torch.no_grad():
        other_probs = torch.softmax(outputs, dim=1)[:, other_id]
        conflicting_indices = torch.where(conflicting)[0]

        # Sort conflicting samples by Other-probability, descending
        # (most confident mispredictions first)
        sorted_order = torch.argsort(other_probs[conflicting_indices], descending=True)

        # Drop the top flip_rate fraction of these
        n_to_drop = math.ceil(n_conflicting * flip_rate)
        n_to_drop = min(n_to_drop, n_conflicting)
        drop_indices = conflicting_indices[sorted_order[:n_to_drop]]

        # Build exclusion mask
        exclude_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        exclude_mask[drop_indices] = True

    keep_mask = ~exclude_mask
    n_excluded = exclude_mask.sum().item()

    if keep_mask.sum().item() == 0:
        # Safety: don't drop the entire batch
        return outputs, labels, case_weights, {
            'n_conflicting': n_conflicting, 'n_excluded': 0,
            'n_remaining': batch_size
        }

    return (
        outputs[keep_mask],
        labels[keep_mask],
        case_weights[keep_mask],
        {
            'n_conflicting': n_conflicting,
            'n_excluded': n_excluded,
            'n_remaining': keep_mask.sum().item(),
        }
    )


# ==============================================================================
# Single Fold Training
# ==============================================================================

def train_single_fold(
    model, train_dataset, val_dataset, label_encoder, other_id, ms_id,
    hparams, device, fold_idx, trial=None
):
    # Weighted sampler
    label_counts = Counter(train_dataset.labels)
    sample_weights = np.array([1.0 / label_counts[l] for l in train_dataset.labels])
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=hparams['batch_size'], sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    criterion = BalancedCrossEntropyLoss(
        other_id=other_id, ms_id=ms_id,
        label_smoothing=hparams['label_smoothing']
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=hparams['lr'],
        weight_decay=hparams['weight_decay'],
        betas=(0.9, 0.99)
    )
    scaler = GradScaler()

    warmup_steps = hparams['warmup_steps']
    total_steps = MAX_STEPS_PER_FOLD

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7)
        ],
        milestones=[warmup_steps]
    )

    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        mode='max'
    )

    best_metric = 0.0
    step_count = 0
    running_loss = 0.0
    running_excluded = 0
    running_conflicting = 0
    running_batches = 0

    allow_pruning = False #(fold_idx == 0)

    model.train()
    for epoch in range(200):
        for embeddings, labels, case_weights in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            case_weights = case_weights.to(device)

            with autocast():
                outputs = model(embeddings)

            # Compute current flip rate
            flip_rate = compute_flip_rate(
                step_count,
                hparams['flip_rate_init'],
                hparams['flip_rate_final'],
                hparams['flip_rate_delta']
            )

            # Apply exclusion
            outputs_f, labels_f, weights_f, excl_stats = apply_label_exclusion(
                outputs, labels, case_weights, other_id, ms_id, flip_rate
            )

            running_excluded += excl_stats['n_excluded']
            running_conflicting += excl_stats['n_conflicting']
            running_batches += 1

            if outputs_f.shape[0] == 0:
                step_count += 1
                continue

            with autocast():
                loss = criterion(outputs_f, labels_f, weights_f)

            running_loss += loss.detach().item()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Logging
            if step_count % 200 == 0 and step_count > 0:
                avg_loss = running_loss / max(running_batches, 1)
                avg_excl = running_excluded / max(running_batches, 1)
                avg_conf = running_conflicting / max(running_batches, 1)
                #print(f"  [F{fold_idx} S{step_count}] loss={avg_loss:.5f} "
                      #f"flip_rate={flip_rate:.3f} "
                      #f"excluded={avg_excl:.0f}/{avg_conf:.0f} per batch")
                running_loss = 0.0
                running_excluded = 0
                running_conflicting = 0
                running_batches = 0

            # Validation
            if step_count % VAL_INTERVAL == 0 and step_count > 0:
                gain_results = evaluate_fold(
                    model, val_dataset, label_encoder, ms_id, device
                )
                # Primary metric: AUGC
                metric = gain_results['augc']
                lift_20 = gain_results.get('lift@0.2', 0)

                if metric > best_metric:
                    best_metric = metric

                print(f"  [F{fold_idx} S{step_count}] augc={metric:.4f} "
                      f"(best={best_metric:.4f}) "
                      f"lift@20%={lift_20:.2f}x "
                      f"flip_rate={flip_rate:.3f}")

                if trial is not None and allow_pruning:
                    trial.report(best_metric, step_count)
                    if trial.should_prune():
                        print(f"  [F{fold_idx}] Pruned at step {step_count}")
                        return best_metric

                if early_stopping(metric):
                    print(f"  [F{fold_idx}] Early stopped at step {step_count}")
                    break

                model.train()

            step_count += 1
            if step_count >= total_steps:
                break

        if step_count >= total_steps or early_stopping.early_stop:
            break

    return best_metric


def evaluate_fold(model, val_dataset, label_encoder, ms_id, device):
    model.eval()

    val_indices = random.sample(range(len(val_dataset)),
                                k=min(VAL_SAMPLES, len(val_dataset)))
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=16384, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    all_probs, all_labels, all_weights = [], [], []

    with torch.no_grad():
        for embeddings, labels, weights in val_loader:
            embeddings = embeddings.to(device)
            outputs = model(embeddings)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_weights.append(weights.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_weights = np.concatenate(all_weights)

    ms_scores = all_probs[:, ms_id]
    ms_binary = (all_labels == ms_id).astype(np.float32)

    return compute_gain_curve(ms_scores, ms_binary, all_weights)


# ==============================================================================
# Optuna Objective
# ==============================================================================

def objective(trial, emb_path, labels_path, weights_path, weights_csv,
              results_csv, label_encoder, other_id, ms_id, device):

    hidden_dims = [256, 128, 64]
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    input_dropout = trial.suggest_float('input_dropout', 0.0, 0.5)

    # --- Training ---
    batch_size = 4096 #trial.suggest_categorical('batch_size', [2048, 4096])
    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-3, 0.2, log=True)
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)
    warmup_steps = 1000 #trial.suggest_int('warmup_steps', 100, 10000, step=100)

    # --- Label exclusion schedule ---
    flip_rate_init = trial.suggest_float('flip_rate_init', 0.0, 0.25)
    flip_rate_final = trial.suggest_float('flip_rate_final', 0.25, 1.0)
    flip_rate_delta = trial.suggest_float('flip_rate_delta', 0, 0.001)

    hparams = {
        'hidden_dims': hidden_dims,
        'dropout': dropout,
        'input_dropout': input_dropout,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'label_smoothing': label_smoothing,
        'warmup_steps': warmup_steps,
        'flip_rate_init': flip_rate_init,
        'flip_rate_final': flip_rate_final,
        'flip_rate_delta': flip_rate_delta,
    }

    # Check parameter count
    test_model = EmbeddingClassifier(
        input_dim=EMBEDDING_DIM,
        hidden_dims=hidden_dims if hidden_dims else None,
        num_classes=2,
        dropout=dropout,
        input_dropout=input_dropout
    )
    n_params = count_parameters(test_model)
    del test_model

    trial.set_user_attr('n_params', n_params)
    trial.set_user_attr('hidden_dims', hidden_dims)

    if abs(flip_rate_delta) > 1e-8:
        steps_to_final = abs(flip_rate_final - flip_rate_init) / abs(flip_rate_delta)
    else:
        steps_to_final = float('inf')

    print(f"\n{'='*70}")
    print(f"Trial {trial.number}: {hidden_dims or 'linear'} ({n_params:,} params)")
    print(f"  lr={lr:.2e} wd={weight_decay:.3f} bs={batch_size} "
          f"smooth={label_smoothing:.3f}")
    print(f"  dropout={dropout:.2f} input_dropout={input_dropout:.2f}")
    print(f"  flip: {flip_rate_init:.3f} -> {flip_rate_final:.3f} "
          f"(delta={flip_rate_delta:.5f}, ~{steps_to_final:.0f} steps to final)")
    print(f"{'='*70}")

    cv_seed = 42 #trial.number * 7919 + 42
    splits = create_author_kfold_splits(weights_csv, n_splits=N_FOLDS, seed=cv_seed)

    fold_metrics = []
    fold_n_ms = []

    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        print(f"\n  --- Fold {fold_idx} (train={len(train_indices):,}, "
              f"val={len(val_indices):,}) ---")

        train_dataset = MMapEmbeddingDataset(
            emb_path, labels_path, weights_path, label_encoder,
            ms_label=MS_LABEL, indices=train_indices, quiet=True
        )
        val_dataset = MMapEmbeddingDataset(
            emb_path, labels_path, weights_path, label_encoder,
            ms_label=MS_LABEL, indices=val_indices, quiet=True
        )

        n_val_ms = (val_dataset.labels == ms_id).sum()
        if n_val_ms < 10:
            print(f"  Fold {fold_idx}: only {n_val_ms} MS samples in val, skipping")
            del train_dataset, val_dataset
            continue

        model = EmbeddingClassifier(
            input_dim=EMBEDDING_DIM,
            hidden_dims=hidden_dims if hidden_dims else None,
            num_classes=2,
            dropout=dropout,
            input_dropout=input_dropout
        ).to(device)

        best_fold_metric = train_single_fold(
            model, train_dataset, val_dataset, label_encoder,
            other_id, ms_id, hparams, device, fold_idx, trial if fold_idx == 0 else None
        )

        fold_metrics.append(best_fold_metric)
        fold_n_ms.append(n_val_ms)
        print(f"  Fold {fold_idx} best augc: {best_fold_metric:.4f}")

        del model, train_dataset, val_dataset
        torch.cuda.empty_cache()

    if len(fold_metrics) == 0:
        raise optuna.TrialPruned()

    mean_metric = np.mean(fold_metrics)
    std_metric = np.std(fold_metrics)

    print(f"\n  Trial {trial.number} result: augc={mean_metric:.4f} ± {std_metric:.4f} "
          f"(folds: {[f'{m:.4f}' for m in fold_metrics]}, "
          f"ms_counts: {[int(n) for n in fold_n_ms]})")

    trial_row = {
        'trial': trial.number,
        'augc': mean_metric,
        'std_augc': std_metric,
        'n_params': n_params,
        **{f'fold{i}': m for i, m in enumerate(fold_metrics)},
        **{f'fold{i}_n_ms': int(n) for i, n in enumerate(fold_n_ms)},
        **{k: v for k, v in hparams.items() if k != 'hidden_dims'},
        'hidden_dims': str(hidden_dims),
    }
    pd.DataFrame([trial_row]).to_csv(
        results_csv, mode='a', header=not os.path.exists(results_csv), index=False
    )

    return mean_metric


# ==============================================================================
# Main
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running {N_TRIALS} trials per fold, {N_FOLDS}-fold inner CV")
    print(f"Max {MAX_STEPS_PER_FOLD} steps per fold, val every {VAL_INTERVAL} steps")
    print(f"Optimizing: AUGC — separate study per outer CV fold")

    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    os.makedirs(STUDY_DIR, exist_ok=True)

    # --- Label Encoder ---
    label_encoder = LabelEncoder()
    label_encoder.fit(BINARY_LABELS)
    other_id = label_encoder.transform(['Other'])[0]
    ms_id = label_encoder.transform(['Multiple sclerosis'])[0]
    print(f"Binary labels: {BINARY_LABELS}")
    print(f"Other ID: {other_id}, MS ID: {ms_id}")

    # Each outer fold gets its own Optuna study so hyperparameters are chosen
    # using only that fold's data.  The held-out half is never touched.
    fold_configs = [
        {
            'name': 'fold1',
            'emb_csv': TRAIN_CSV,
            'weights_csv': TRAIN_WEIGHTS_CSV,
            'study_name': 'ms-classifier-exclusion-fold1',
            'study_db': os.path.join(STUDY_DIR, 'optuna_study_exclusion_fold1.db'),
            'results_csv': os.path.join(STUDY_DIR, 'trial_results_exclusion_fold1.csv'),
            'best_config_json': os.path.join(STUDY_DIR, 'best_config_exclusion_fold1.json'),
        },
        {
            'name': 'fold2',
            'emb_csv': TEST_CSV,
            'weights_csv': TEST_WEIGHTS_CSV,
            'study_name': 'ms-classifier-exclusion-fold2',
            'study_db': os.path.join(STUDY_DIR, 'optuna_study_exclusion_fold2.db'),
            'results_csv': os.path.join(STUDY_DIR, 'trial_results_exclusion_fold2.csv'),
            'best_config_json': os.path.join(STUDY_DIR, 'best_config_exclusion_fold2.json'),
        },
    ]

    for fold_cfg in fold_configs:
        fold_name = fold_cfg['name']
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER SEARCH: {fold_name.upper()}")
        print(f"  Training data: {os.path.basename(fold_cfg['emb_csv'])}")
        print(f"{'='*70}")

        if os.path.exists(fold_cfg['results_csv']):
            os.remove(fold_cfg['results_csv'])

        # --- Load preprocessed data ---
        print(f"\n=== Loading Data ({fold_name}) ===")
        emb_path, labels_path, weights_path = get_preprocessed_paths(
            fold_cfg['emb_csv'], fold_cfg['weights_csv'], PREPROCESSED_DIR
        )

        labels_raw = np.load(labels_path, allow_pickle=True)
        unique, counts = np.unique(labels_raw, return_counts=True)
        print(f"{fold_name} dataset distribution:")
        for u, c in zip(unique, counts):
            print(f"  {u}: {c:,}")
        del labels_raw

        print(f"\n{fold_name} author distribution:")
        author_df = pd.read_csv(fold_cfg['weights_csv'], usecols=[LABEL_COL, AUTHOR_COL])
        print(f"  Total authors: {author_df[AUTHOR_COL].nunique()}")
        ms_authors_n = author_df[author_df[LABEL_COL] == MS_LABEL][AUTHOR_COL].nunique()
        print(f"  MS authors: {ms_authors_n}")
        for disease in ORIGINAL_LABELS:
            n_auth = author_df[author_df[LABEL_COL] == disease][AUTHOR_COL].nunique()
            n_seq = (author_df[LABEL_COL] == disease).sum()
            print(f"  {disease}: {n_auth} authors, {n_seq:,} sequences")
        del author_df

        # --- Optuna Study ---
        print(f"\n=== Starting Hyperparameter Search ({fold_name}) ===")

        study = optuna.create_study(
            study_name=fold_cfg['study_name'],
            direction="maximize",
            storage=f"sqlite:///{fold_cfg['study_db']}",
            load_if_exists=True,
            pruner=MedianPruner(
                n_startup_trials=25,
                n_warmup_steps=2000,
                interval_steps=VAL_INTERVAL
            )
        )

        def objective_wrapper(trial):
            return objective(
                trial, emb_path, labels_path, weights_path, fold_cfg['weights_csv'],
                fold_cfg['results_csv'], label_encoder, other_id, ms_id, device
            )

        study.optimize(objective_wrapper, n_trials=N_TRIALS, show_progress_bar=False)

        # Rebuild clean results CSV from DB
        df = study.trials_dataframe()
        df.to_csv(fold_cfg['results_csv'], index=False)

        # --- Results Summary ---
        print(f"\n{'='*70}")
        print(f"SEARCH COMPLETE: {fold_name.upper()}")
        print(f"{'='*70}")

        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best AUGC: {study.best_value:.4f}")
        print(f"\nBest hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")

        n_params = study.best_trial.user_attrs.get('n_params', 'unknown')
        hidden_dims = study.best_trial.user_attrs.get('hidden_dims', 'unknown')
        print(f"  architecture: {hidden_dims} ({n_params:,} params)")

        bp = study.best_trial.params
        init_r = bp.get('flip_rate_init', 0)
        final_r = bp.get('flip_rate_final', 0)
        delta_r = bp.get('flip_rate_delta', 0)
        if abs(delta_r) > 1e-8:
            steps_needed = abs(final_r - init_r) / abs(delta_r)
            print(f"  flip schedule: {init_r:.3f} -> {final_r:.3f} "
                  f"(~{steps_needed:.0f} steps)")
        else:
            print(f"  flip schedule: constant at {init_r:.3f}")

        best_config = {
            'fold': fold_name,
            'best_trial': study.best_trial.number,
            'best_augc': study.best_value,
            'params': study.best_trial.params,
            'n_params': n_params,
            'hidden_dims': hidden_dims,
        }
        with open(fold_cfg['best_config_json'], 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"\nBest config saved to {fold_cfg['best_config_json']}")

        # Top 10 trials
        print(f"\nTop 10 trials ({fold_name}):")
        print(f"{'Trial':>6} {'AUGC':>8} {'Params':>10} {'Architecture':<25} "
              f"{'Flip Init':>10} {'Flip Final':>11} {'Delta':>10}")
        print("-" * 90)

        trials_sorted = sorted(study.trials,
                               key=lambda t: t.value if t.value is not None else -1,
                               reverse=True)
        for t in trials_sorted[:10]:
            if t.value is None:
                continue
            dims = t.user_attrs.get('hidden_dims', '?')
            n_p = t.user_attrs.get('n_params', 0)
            fi = t.params.get('flip_rate_init', 0)
            ff = t.params.get('flip_rate_final', 0)
            fd = t.params.get('flip_rate_delta', 0)
            print(f"{t.number:>6} {t.value:>8.4f} {n_p:>10,} {str(dims):<25} "
                  f"{fi:>10.3f} {ff:>11.3f} {fd:>10.5f}")

        try:
            importance = optuna.importance.get_param_importances(study)
            print(f"\nParameter importance ({fold_name}):")
            for param, imp in importance.items():
                print(f"  {param}: {imp:.3f}")
        except Exception:
            pass

        print(f"\nFull results in {fold_cfg['results_csv']}")
        print(f"Optuna DB in {fold_cfg['study_db']}")
        print(f"Best config in {fold_cfg['best_config_json']}")

    print(f"\n{'='*70}")
    print("ALL FOLDS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
    
