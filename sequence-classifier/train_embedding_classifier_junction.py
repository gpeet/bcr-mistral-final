#!/usr/bin/env python3
"""
Ensemble Training & Stacked Inference — Junction CDR3 Embeddings, Label-Exclusion Variant

Identical to train_embedding_classifier.py but operates on CDR3/junction-only embeddings.
Reuses hyperparameters from the full-sequence exclusion Optuna study (no re-tuning).

2-fold outer CV between the pre-split train and test halves:
  unflipped: train on TRAIN_CSV, predict on TEST_CSV
  flipped:   train on TEST_CSV,  predict on TRAIN_CSV

Loss: BalancedCrossEntropyLoss + hard label exclusion (replaces soft relabeling).
Checkpoint metric: AUGC (matching the exclusion tuning objective).
"""

from bcrm.seed import seed_everything
from bcrm.paths import data_path, model_path, out_path, repo_path
import os
import math
import random
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as nn_utils
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder


# ==============================================================================
# Configuration
# ==============================================================================

TRAIN_CSV         = out_path("embeddings/classifier_data_igblast_train_embeddings_weights_junction.csv")
TRAIN_WEIGHTS_CSV = data_path("classifier/classifier_data_igblast-train-weights.csv")
TEST_CSV          = out_path("embeddings/classifier_data_igblast_test_embeddings_weights_junction.csv")
TEST_WEIGHTS_CSV  = data_path("classifier/classifier_data_igblast-test-weights.csv")

# Reuse full-sequence hyperparameter results — no re-tuning for junction models
PREPROCESSED_DIR  = out_path("preprocessed")
BEST_CONFIG_FOLD1_JSON = out_path("sequence-classifier/hparam-search-exclusion/best_config_exclusion_fold1.json")
BEST_CONFIG_FOLD2_JSON = out_path("sequence-classifier/hparam-search-exclusion/best_config_exclusion_fold2.json")
OUTPUT_DIR        = out_path("sequence-classifier/ensemble-output-junction-exclusion")

EMBEDDING_DIM     = 1024
LABEL_COL         = "Disease"
WEIGHT_COL        = "weight"
AUTHOR_COL        = "Author"
SUBJECT_COL       = "Subject"
MS_LABEL          = "Multiple sclerosis"
BINARY_LABELS     = ["Multiple sclerosis", "Other"]
PREPROCESS_CHUNK_SIZE = 50000

# Ensemble
N_MODELS = 10
RETRAIN  = os.environ.get("BCRM_RETRAIN") == "1"   # otherwise reuse existing checkpoints (the released models)
N_FOLDS  = 3

# Training — longer budget than tuning; early stopping is the real control
MAX_STEPS_PER_FOLD    = int(os.environ.get("BCRM_MAX_STEPS", 50000))  # lower it for smoke tests
VAL_INTERVAL          = 500
VAL_SAMPLES           = 65536
EARLY_STOPPING_PATIENCE   = 15
EARLY_STOPPING_MIN_DELTA  = 0.005

# Inference
INFERENCE_BATCH_SIZE = 16384 * 4

NUM_WORKERS   = 8
MAX_GRAD_NORM = 0.5

GAIN_FRACTIONS     = [0.01, 0.05, 0.1, 0.2, 0.5]
TOP_LIFT_FRACTIONS = [0.5]


# ==============================================================================
# Balanced Cross-Entropy Loss with Label Smoothing
# ==============================================================================

class BalancedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy with per-sample weights, optional label smoothing, and
    50/50 class-balanced averaging.
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
        ms_mask    = targets == self.ms_id
        n_other = other_mask.sum().item()
        n_ms    = ms_mask.sum().item()

        other_mean = (weighted_loss[other_mask].mean()
                      if n_other > 0 else torch.tensor(0.0, device=device))
        ms_mean    = (weighted_loss[ms_mask].mean()
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
        return {"augc": 0.0, "top20_augc": 0.0, "mean_top_lift": 0.0}

    sorted_indices = np.argsort(-ms_scores)
    sorted_labels  = labels_binary[sorted_indices]
    sorted_weights = weights[sorted_indices] if weights is not None else np.ones(n, dtype=np.float32)

    total_ms_weight = (sorted_labels * sorted_weights).sum()
    total_weight    = sorted_weights.sum()
    baseline_rate   = total_ms_weight / (total_weight + 1e-8)

    results = {}
    for frac in fractions:
        k = max(1, int(n * frac))
        top_k_labels  = sorted_labels[:k]
        top_k_weights = sorted_weights[:k]
        precision = ((top_k_labels * top_k_weights).sum() / top_k_weights.sum()
                     if top_k_weights.sum() > 0 else 0.0)
        lift = precision / (baseline_rate + 1e-8)
        results[f"precision@{frac}"] = float(precision)
        results[f"lift@{frac}"]      = float(lift)

    cum_ms          = np.cumsum(sorted_labels * sorted_weights)
    cum_samples     = np.arange(1, n + 1, dtype=np.float32)
    cum_ms_frac     = cum_ms / (total_ms_weight + 1e-8)
    cum_samples_frac = cum_samples / n

    augc = np.trapz(cum_ms_frac, cum_samples_frac) - 0.5
    results["augc"] = float(augc)

    cutoff = max(1, int(n * 0.20))
    partial_area     = np.trapz(cum_ms_frac[:cutoff], cum_samples_frac[:cutoff])
    partial_diagonal = 0.5 * (cum_samples_frac[cutoff - 1] ** 2)
    results["top20_augc"] = float(partial_area - partial_diagonal)

    top_fracs = [f for f in fractions if f in TOP_LIFT_FRACTIONS]
    results["mean_top_lift"] = float(np.mean([results[f"lift@{f}"] for f in top_fracs])
                                     if top_fracs else 0.0)
    results["baseline_rate"] = float(baseline_rate)
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
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.classifier(x)


# ==============================================================================
# Early Stopping
# ==============================================================================

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode="max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.counter    = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        improved = (score > self.best_score + self.min_delta if self.mode == "max"
                    else score < self.best_score - self.min_delta)
        if improved:
            self.best_score = score
            self.counter    = 0
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
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    embeddings_path = f"{output_prefix}_embeddings.npy"
    labels_path     = f"{output_prefix}_labels.npy"
    weights_path    = f"{output_prefix}_weights.npy"
    embedding_cols  = [f"hidden_state_{i}" for i in range(EMBEDDING_DIM)]

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
        embeddings_path, mode="w+", dtype=np.float32, shape=(n_rows, EMBEDDING_DIM)
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
        weights_path, mode="w+", dtype=np.float32, shape=(n_rows,)
    )
    weights_mmap[:] = all_wt
    del all_wt, weights_mmap
    np.save(labels_path, np.array(labels_list, dtype=object))
    return embeddings_path, labels_path, weights_path


def get_preprocessed_paths(csv_path, weights_csv_path, preprocessed_dir):
    base_name      = os.path.splitext(os.path.basename(csv_path))[0]
    output_prefix  = os.path.join(preprocessed_dir, base_name)
    embeddings_path = f"{output_prefix}_embeddings.npy"
    labels_path     = f"{output_prefix}_labels.npy"
    weights_path    = f"{output_prefix}_weights.npy"

    if all(os.path.exists(p) for p in [embeddings_path, labels_path, weights_path]):
        print(f"Found preprocessed files for {base_name}")
        return embeddings_path, labels_path, weights_path

    return preprocess_csv_to_mmap(csv_path, weights_csv_path, output_prefix)


# ==============================================================================
# Dataset
# ==============================================================================

class MMapEmbeddingDataset(Dataset):
    def __init__(self, embeddings_path, labels_path, weights_path, label_encoder,
                 ms_label="Multiple sclerosis", indices=None, quiet=False):
        self.embeddings    = np.load(embeddings_path, mmap_mode="r")
        self.weights       = np.load(weights_path, mmap_mode="r")
        self.labels_raw    = np.load(labels_path, allow_pickle=True)
        self.label_encoder = label_encoder

        if indices is not None:
            self.indices          = np.array(indices)
            self.labels_raw_subset = self.labels_raw[self.indices]
        else:
            self.indices          = np.arange(len(self.labels_raw))
            self.labels_raw_subset = self.labels_raw

        binary_labels = np.where(self.labels_raw_subset == ms_label, ms_label, "Other")
        self.labels   = label_encoder.transform(binary_labels)

        if not quiet:
            bin_unique, bin_counts = np.unique(self.labels, return_counts=True)
            for u, c in zip(bin_unique, bin_counts):
                print(f"    {label_encoder.classes_[u]}: {c:,} ({100*c/len(self.labels):.1f}%)")
            print(f"    Total: {len(self):,} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx  = self.indices[idx]
        embedding = torch.from_numpy(self.embeddings[real_idx].copy()).float()
        label     = self.labels[idx]
        weight    = torch.tensor(self.weights[real_idx], dtype=torch.float32)
        return embedding, label, weight


# ==============================================================================
# Author-Grouped K-Fold Splits
# ==============================================================================

def create_author_kfold_splits(weights_csv_path, n_splits=5, seed=None):
    if seed is None:
        seed = random.randint(0, 2**31)

    df      = pd.read_csv(weights_csv_path, usecols=[LABEL_COL, AUTHOR_COL])
    authors = df[AUTHOR_COL].values

    # Authors that have ANY MS sequences
    ms_author_set   = set(df.loc[df[LABEL_COL] == MS_LABEL, AUTHOR_COL].unique())
    all_unique      = df[AUTHOR_COL].unique()
    ms_authors      = [a for a in all_unique if a in ms_author_set]
    other_authors   = [a for a in all_unique if a not in ms_author_set]

    rng = np.random.RandomState(seed)
    rng.shuffle(ms_authors)
    rng.shuffle(other_authors)

    fold_authors = defaultdict(list)
    for i, author in enumerate(ms_authors):
        fold_authors[i % n_splits].append(author)

    other_author_counts  = {a: (df[AUTHOR_COL] == a).sum() for a in other_authors}
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
        val_authors   = set(fold_authors[val_fold])
        train_authors = set()
        for fi in range(n_splits):
            if fi != val_fold:
                train_authors.update(fold_authors[fi])

        val_indices   = [i for a in val_authors   for i in author_to_indices[a]]
        train_indices = [i for a in train_authors for i in author_to_indices[a]]
        splits.append((np.array(sorted(train_indices)), np.array(sorted(val_indices))))

    return splits


# ==============================================================================
# Flip Rate Schedule
# ==============================================================================

def compute_flip_rate(step, flip_rate_init, flip_rate_final, flip_rate_delta):
    """
    Linear ramp from flip_rate_init toward flip_rate_final at flip_rate_delta
    per step, clamped appropriately.
    """
    raw = flip_rate_init + flip_rate_delta * step
    if flip_rate_delta >= 0:
        return min(raw, flip_rate_final)
    else:
        return max(raw, flip_rate_final)


# ==============================================================================
# Label Exclusion
# ==============================================================================

def apply_label_exclusion(outputs, labels, case_weights, other_id, ms_id, flip_rate):
    """
    Hard exclusion: among MS-labeled samples the model predicts as Other,
    drop the top flip_rate fraction (most confident) from the loss.
    """
    device     = outputs.device
    batch_size = outputs.shape[0]

    if batch_size == 0 or flip_rate <= 0.0:
        return outputs, labels, case_weights, {
            "n_conflicting": 0, "n_excluded": 0, "n_remaining": batch_size
        }

    with torch.no_grad():
        preds      = torch.argmax(outputs, dim=1)
        conflicting = (preds == other_id) & (labels == ms_id)
        n_conflicting = conflicting.sum().item()

    if n_conflicting <= 1:
        return outputs, labels, case_weights, {
            "n_conflicting": n_conflicting, "n_excluded": 0, "n_remaining": batch_size
        }

    with torch.no_grad():
        other_probs        = torch.softmax(outputs, dim=1)[:, other_id]
        conflicting_indices = torch.where(conflicting)[0]
        sorted_order       = torch.argsort(other_probs[conflicting_indices], descending=True)
        n_to_drop          = min(math.ceil(n_conflicting * flip_rate), n_conflicting)
        drop_indices       = conflicting_indices[sorted_order[:n_to_drop]]
        exclude_mask       = torch.zeros(batch_size, dtype=torch.bool, device=device)
        exclude_mask[drop_indices] = True

    keep_mask  = ~exclude_mask
    n_excluded = exclude_mask.sum().item()

    if keep_mask.sum().item() == 0:
        # Safety: never drop the entire batch
        return outputs, labels, case_weights, {
            "n_conflicting": n_conflicting, "n_excluded": 0, "n_remaining": batch_size
        }

    return (
        outputs[keep_mask],
        labels[keep_mask],
        case_weights[keep_mask],
        {"n_conflicting": n_conflicting, "n_excluded": n_excluded,
         "n_remaining": keep_mask.sum().item()},
    )


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_model(model, dataset, ms_id, device, max_samples=None):
    model.eval()
    if max_samples is not None and len(dataset) > max_samples:
        eval_dataset = Subset(dataset, random.sample(range(len(dataset)), k=max_samples))
    else:
        eval_dataset = dataset

    loader = DataLoader(eval_dataset, batch_size=INFERENCE_BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    all_probs, all_labels, all_weights = [], [], []
    with torch.no_grad():
        for embeddings, labels, weights in loader:
            embeddings = embeddings.to(device)
            probs = torch.softmax(model(embeddings), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            all_weights.append(weights.numpy())

    all_probs   = np.concatenate(all_probs)
    all_labels  = np.concatenate(all_labels)
    all_weights = np.concatenate(all_weights)

    ms_scores = all_probs[:, ms_id]
    ms_binary = (all_labels == ms_id).astype(np.float32)

    gain = compute_gain_curve(ms_scores, ms_binary, all_weights)
    return gain, all_probs


# ==============================================================================
# Full Dataset Inference
# ==============================================================================

def predict_full_dataset(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=INFERENCE_BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    all_probs = []
    with torch.no_grad():
        for embeddings, labels, weights in loader:
            embeddings = embeddings.to(device)
            probs = torch.softmax(model(embeddings), dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


# ==============================================================================
# Stacked Ensemble Inference
# ==============================================================================

def ensemble_predict(checkpoint_paths, hparams, test_dataset, device):
    individual_probs = []
    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"  Loading model {i} from {os.path.basename(ckpt_path)}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        hp   = ckpt.get("hparams", hparams)   # the architecture this checkpoint was trained with

        model = EmbeddingClassifier(
            input_dim=EMBEDDING_DIM,
            hidden_dims=hp["hidden_dims"] or None,
            num_classes=2,
            dropout=hp["dropout"],
            input_dropout=hp["input_dropout"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        probs = predict_full_dataset(model, test_dataset, device)
        individual_probs.append(probs)
        print(f"    val augc={ckpt.get('metric', '?')}, best_step={ckpt.get('step', '?')}")

        del model
        torch.cuda.empty_cache()

    stacked        = np.stack(individual_probs, axis=0)
    ensemble_probs = stacked.mean(axis=0)
    return ensemble_probs, individual_probs


# ==============================================================================
# Generate CV Assignments
# ==============================================================================

def generate_model_assignments(n_models, n_folds, base_seed=42):
    """
    Cycles through resampled K-fold rounds to produce n_models (cv_seed, fold_idx)
    assignments.
    """
    assignments = []
    cv_round    = 0
    while len(assignments) < n_models:
        cv_seed = base_seed + cv_round * 7919
        for fold_idx in range(n_folds):
            if len(assignments) >= n_models:
                break
            assignments.append((cv_seed, fold_idx))
        cv_round += 1
    return assignments


# ==============================================================================
# Train Single Model
# ==============================================================================

def train_single_model(
    hparams, train_dataset, val_dataset, label_encoder, other_id, ms_id,
    device, model_idx, checkpoint_path
):
    model = EmbeddingClassifier(
        input_dim=EMBEDDING_DIM,
        hidden_dims=hparams["hidden_dims"] or None,
        num_classes=2,
        dropout=hparams["dropout"],
        input_dropout=hparams["input_dropout"],
    ).to(device)

    # Class-balanced weighted sampler
    label_counts   = Counter(train_dataset.labels)
    sample_weights = np.array([1.0 / label_counts[l] for l in train_dataset.labels])
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=hparams["batch_size"], sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    criterion = BalancedCrossEntropyLoss(
        other_id=other_id, ms_id=ms_id,
        label_smoothing=hparams["label_smoothing"],
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
        betas=(0.9, 0.99),
    )
    scaler       = GradScaler()
    warmup_steps = hparams["warmup_steps"]
    total_steps  = MAX_STEPS_PER_FOLD
    val_interval = min(VAL_INTERVAL, max(1, total_steps - 1))   # short smoke runs still validate and save

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7),
        ],
        milestones=[warmup_steps],
    )

    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        mode="max",
    )

    best_metric       = float("-inf")   # always save the first evaluated model, even if its AUGC is negative
    best_step         = 0
    step_count        = 0
    running_loss      = 0.0
    running_excluded  = 0
    running_conflicting = 0
    running_batches   = 0

    model.train()
    for epoch in range(999):
        for embeddings, labels, case_weights in train_loader:
            embeddings   = embeddings.to(device)
            labels       = labels.to(device)
            case_weights = case_weights.to(device)

            # Two-stage autocast: forward pass then exclusion then loss
            with autocast():
                outputs = model(embeddings)

            flip_rate = compute_flip_rate(
                step_count,
                hparams["flip_rate_init"],
                hparams["flip_rate_final"],
                hparams["flip_rate_delta"],
            )

            outputs_f, labels_f, weights_f, excl_stats = apply_label_exclusion(
                outputs, labels, case_weights, other_id, ms_id, flip_rate
            )

            running_excluded    += excl_stats["n_excluded"]
            running_conflicting += excl_stats["n_conflicting"]
            running_batches     += 1

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

            if step_count % 200 == 0 and step_count > 0:
                avg_loss = running_loss / max(running_batches, 1)
                avg_excl = running_excluded / max(running_batches, 1)
                avg_conf = running_conflicting / max(running_batches, 1)
                print(f"  [M{model_idx} S{step_count}] loss={avg_loss:.5f} "
                      f"flip_rate={flip_rate:.3f} "
                      f"excl={avg_excl:.0f}/{avg_conf:.0f} per batch")
                running_loss        = 0.0
                running_excluded    = 0
                running_conflicting = 0
                running_batches     = 0

            if step_count % val_interval == 0 and step_count > 0:
                gain_results, _ = evaluate_model(
                    model, val_dataset, ms_id, device, max_samples=VAL_SAMPLES
                )
                # Checkpoint on AUGC (matching the exclusion tuning objective)
                metric   = gain_results["augc"]
                improved = metric > best_metric

                if improved:
                    best_metric = metric
                    best_step   = step_count
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "step":    step_count,
                        "metric":  best_metric,
                        "hparams": hparams,
                        "model_idx": model_idx,
                    }, checkpoint_path)

                marker = " *" if improved else ""
                print(f"  [M{model_idx} S{step_count}] augc={metric:.4f} "
                      f"(best={best_metric:.4f} @ S{best_step}){marker} "
                      f"flip_rate={flip_rate:.3f}")

                if early_stopping(metric):
                    print(f"  [M{model_idx}] Early stopped at step {step_count}")
                    break

                model.train()

            step_count += 1
            if step_count >= total_steps:
                break

        if step_count >= total_steps or early_stopping.early_stop:
            break

    training_info = {
        "model_idx":     model_idx,
        "best_metric":   best_metric,
        "best_step":     best_step,
        "total_steps":   step_count,
        "early_stopped": early_stopping.early_stop,
    }

    del model, optimizer, scaler, scheduler
    torch.cuda.empty_cache()
    return best_metric, training_info


# ==============================================================================
# Main
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training {N_MODELS} models via resampled {N_FOLDS}-fold inner CV")
    print(f"Max {MAX_STEPS_PER_FOLD} steps per model, "
          f"early stopping patience={EARLY_STOPPING_PATIENCE}")
    print(f"Checkpoint metric: AUGC (exclusion variant)")

    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Label encoder ---
    label_encoder = LabelEncoder()
    label_encoder.fit(BINARY_LABELS)
    other_id = label_encoder.transform(["Other"])[0]
    ms_id    = label_encoder.transform(["Multiple sclerosis"])[0]
    print(f"\nBinary labels: {list(label_encoder.classes_)}")
    print(f"Other ID: {other_id}, MS ID: {ms_id}")

    # --- Generate inner CV model assignments ---
    assignments = generate_model_assignments(N_MODELS, N_FOLDS, base_seed=42)
    print(f"\nModel assignments ({N_MODELS} models, cv_seed/fold_idx):")
    for i, (seed, fold) in enumerate(assignments):
        print(f"  Model {i}: seed={seed}, fold={fold}")

    # ===================================================================
    # 2-Fold Outer CV
    # ===================================================================

    outer_folds = [
        {
            "name":             "unflipped",
            "train_csv":        TRAIN_CSV,
            "train_weights_csv": TRAIN_WEIGHTS_CSV,
            "pred_csv":         TEST_CSV,
            "pred_weights_csv": TEST_WEIGHTS_CSV,
            "best_config_json": BEST_CONFIG_FOLD1_JSON,
        },
        {
            "name":             "flipped",
            "train_csv":        TEST_CSV,
            "train_weights_csv": TEST_WEIGHTS_CSV,
            "pred_csv":         TRAIN_CSV,
            "pred_weights_csv": TRAIN_WEIGHTS_CSV,
            "best_config_json": BEST_CONFIG_FOLD2_JSON,
        },
    ]

    for outer_fold in outer_folds:
        fold_name        = outer_fold["name"]
        fold_output_dir  = os.path.join(OUTPUT_DIR, fold_name)
        fold_models_dir  = os.path.join(fold_output_dir, "checkpoints")
        fold_preds_dir   = os.path.join(fold_output_dir, "predictions")

        os.makedirs(fold_models_dir, exist_ok=True)
        os.makedirs(fold_preds_dir,  exist_ok=True)

        print(f"\n{'='*70}")
        print(f"OUTER FOLD: {fold_name.upper()}")
        print(f"  Train on: {os.path.basename(outer_fold['train_csv'])}")
        print(f"  Predict on: {os.path.basename(outer_fold['pred_csv'])}")
        print(f"{'='*70}")

        # --- Load fold-specific best hyperparameters ---
        print(f"\nLoading best config from {outer_fold['best_config_json']}")
        with open(outer_fold["best_config_json"]) as f:
            best_config = json.load(f)

        params      = best_config["params"]
        hidden_dims = best_config.get("hidden_dims", [])
        if isinstance(hidden_dims, str):
            hidden_dims = json.loads(hidden_dims.replace("'", '"')) if hidden_dims != "[]" else []

        hparams = {
            "hidden_dims":      hidden_dims,
            "dropout":          params["dropout"],
            "input_dropout":    params["input_dropout"],
            "batch_size":       params.get("batch_size", 4096),
            "lr":               params["lr"],
            "weight_decay":     params["weight_decay"],
            "label_smoothing":  params.get("label_smoothing", 0.0),
            "warmup_steps":     params.get("warmup_steps", 1000),
            "flip_rate_init":   params["flip_rate_init"],
            "flip_rate_final":  params["flip_rate_final"],
            "flip_rate_delta":  params["flip_rate_delta"],
        }

        print(f"  Architecture: {hidden_dims} ({best_config.get('n_params', '?')} params)")
        print(f"  lr={hparams['lr']:.2e}  wd={hparams['weight_decay']:.4f}  "
              f"bs={hparams['batch_size']}")
        print(f"  dropout={hparams['dropout']:.2f}  "
              f"input_dropout={hparams['input_dropout']:.2f}")
        print(f"  label_smoothing={hparams['label_smoothing']:.3f}  "
              f"warmup_steps={hparams['warmup_steps']}")
        print(f"  flip: {hparams['flip_rate_init']:.3f} -> {hparams['flip_rate_final']:.3f} "
              f"(delta={hparams['flip_rate_delta']:.5f})")

        # --- Preprocess train embeddings ---
        print(f"\n=== Loading Training Data ({fold_name}) ===")
        train_emb, train_labels_path, train_weights_path = get_preprocessed_paths(
            outer_fold["train_csv"], outer_fold["train_weights_csv"], PREPROCESSED_DIR
        )

        labels_raw = np.load(train_labels_path, allow_pickle=True)
        unique, counts = np.unique(labels_raw, return_counts=True)
        print("Training distribution:")
        for u, c in zip(unique, counts):
            print(f"  {u}: {c:,}")
        del labels_raw

        # --- Preprocess prediction embeddings ---
        print(f"\n=== Loading Prediction Data ({fold_name}) ===")
        pred_emb, pred_labels_path, pred_weights_path = get_preprocessed_paths(
            outer_fold["pred_csv"], outer_fold["pred_weights_csv"], PREPROCESSED_DIR
        )

        labels_raw = np.load(pred_labels_path, allow_pickle=True)
        unique, counts = np.unique(labels_raw, return_counts=True)
        print("Prediction set distribution:")
        for u, c in zip(unique, counts):
            print(f"  {u}: {c:,}")
        del labels_raw

        # --- Train ensemble (or reuse the released checkpoints) ---
        existing = [os.path.join(fold_models_dir, f"model_{i:02d}.pt") for i in range(N_MODELS)]
        if not RETRAIN and all(os.path.exists(p) for p in existing):
            print(f"\nUsing the {N_MODELS} existing checkpoints in {fold_models_dir} "
                  "(set BCRM_RETRAIN=1 to retrain)")
            checkpoint_paths = existing
        else:
            # --- Train ensemble ---
            print(f"\n{'='*70}")
            print(f"ENSEMBLE TRAINING ({fold_name})")
            print(f"{'='*70}")

            checkpoint_paths  = []
            training_summaries = []
            start_time        = time.time()

            for model_idx, (cv_seed, fold_idx) in enumerate(assignments):
                print(f"\n{'='*70}")
                print(f"Model {model_idx}/{N_MODELS} "
                      f"(cv_seed={cv_seed}, fold={fold_idx}) [{fold_name}]")
                print(f"{'='*70}")

                model_start = time.time()
                seed_everything(cv_seed + fold_idx)  # weight init, dropout, batch order

                splits = create_author_kfold_splits(
                    outer_fold["train_weights_csv"], n_splits=N_FOLDS, seed=cv_seed
                )
                train_indices, val_indices = splits[fold_idx]

                print(f"  Train: {len(train_indices):,}, Val: {len(val_indices):,}")

                train_dataset = MMapEmbeddingDataset(
                    train_emb, train_labels_path, train_weights_path, label_encoder,
                    ms_label=MS_LABEL, indices=train_indices, quiet=False,
                )
                val_dataset = MMapEmbeddingDataset(
                    train_emb, train_labels_path, train_weights_path, label_encoder,
                    ms_label=MS_LABEL, indices=val_indices, quiet=False,
                )

                ckpt_path = os.path.join(fold_models_dir, f"model_{model_idx:02d}.pt")
                checkpoint_paths.append(ckpt_path)

                best_metric, info = train_single_model(
                    hparams, train_dataset, val_dataset, label_encoder,
                    other_id, ms_id, device, model_idx, ckpt_path,
                )

                elapsed     = time.time() - model_start
                info["elapsed_sec"] = elapsed
                info["cv_seed"]     = cv_seed
                info["fold_idx"]    = fold_idx
                training_summaries.append(info)

                print(f"\n  Model {model_idx} done: best augc={best_metric:.4f} "
                      f"@ step {info['best_step']} ({elapsed/60:.1f} min)")

                del train_dataset, val_dataset
                torch.cuda.empty_cache()

            total_training_time = time.time() - start_time

            # --- Training summary ---
            all_best = [s["best_metric"] for s in training_summaries]
            print(f"\nMean val AUGC: {np.mean(all_best):.4f} ± {np.std(all_best):.4f}")
            print(f"Total training time: {total_training_time/60:.1f} min")

            summary_path = os.path.join(fold_output_dir, "training_summary.json")
            with open(summary_path, "w") as f:
                json.dump({
                    "fold_name": fold_name,
                    "hparams": hparams,
                    "training_summaries": training_summaries,
                    "total_training_time_sec": total_training_time,
                    "mean_val_augc": float(np.mean(all_best)),
                    "std_val_augc":  float(np.std(all_best)),
                }, f, indent=2)
            print(f"Training summary saved to {summary_path}")

        # --- Stacked Ensemble Inference ---
        print(f"\n{'='*70}")
        print(f"ENSEMBLE INFERENCE ({fold_name})")
        print(f"{'='*70}")

        pred_dataset = MMapEmbeddingDataset(
            pred_emb, pred_labels_path, pred_weights_path, label_encoder,
            ms_label=MS_LABEL, indices=None, quiet=False,
        )

        ensemble_probs, individual_probs = ensemble_predict(
            checkpoint_paths, hparams, pred_dataset, device
        )

        pred_labels_enc = pred_dataset.labels
        pred_weights_arr = np.array([pred_dataset.weights[pred_dataset.indices[i]]
                                     for i in range(len(pred_dataset))])
        ms_binary         = (pred_labels_enc == ms_id).astype(np.float32)
        ensemble_ms_scores = ensemble_probs[:, ms_id]

        ensemble_gain = compute_gain_curve(ensemble_ms_scores, ms_binary, pred_weights_arr)
        print(f"\nEnsemble results ({N_MODELS} models):")
        print(f"  AUGC: {ensemble_gain['augc']:.4f}")
        print(f"  Top-20% AUGC: {ensemble_gain['top20_augc']:.4f}")
        for frac in [0.01, 0.05, 0.1, 0.2]:
            print(f"  precision@{frac}: {ensemble_gain.get(f'precision@{frac}', 0):.4f}  "
                  f"lift@{frac}: {ensemble_gain.get(f'lift@{frac}', 0):.2f}x")

        # --- Load Author/Subject for prediction CSV ---
        print(f"\nLoading Author/Subject metadata from {outer_fold['pred_weights_csv']}...")
        meta_df = pd.read_csv(
            outer_fold["pred_weights_csv"],
            usecols=[AUTHOR_COL, SUBJECT_COL, LABEL_COL, WEIGHT_COL]
        )

        # --- Save prediction CSV ---
        pred_csv_path = os.path.join(fold_preds_dir, "ensemble_predictions.csv")
        pred_df = pd.DataFrame({
            "ms_score":        ensemble_ms_scores,
            "predicted_class": label_encoder.classes_[ensemble_probs.argmax(axis=1)],
            "true_label":      pred_dataset.labels_raw_subset,
            "weight":          pred_weights_arr,
            "Author":          meta_df[AUTHOR_COL].values,
            "Subject":         meta_df[SUBJECT_COL].values,
        })
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"Predictions saved to {pred_csv_path} ({len(pred_df):,} rows)")

        # Save results JSON
        results_path = os.path.join(fold_output_dir, "test_results.json")
        individual_gains = []
        for probs in individual_probs:
            g = compute_gain_curve(probs[:, ms_id], ms_binary, pred_weights_arr)
            individual_gains.append(g)

        with open(results_path, "w") as f:
            json.dump({
                "fold_name": fold_name,
                "ensemble":  ensemble_gain,
                "individual": individual_gains,
                "n_models":  N_MODELS,
                "n_pred":    len(pred_dataset),
                "n_pred_ms": int(ms_binary.sum()),
                "hparams":   hparams,
            }, f, indent=2)
        print(f"Results saved to {results_path}")

        del pred_dataset
        torch.cuda.empty_cache()
        checkpoint_paths = []
        training_summaries = []

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  unflipped predictions: {os.path.join(OUTPUT_DIR, 'unflipped/predictions/ensemble_predictions.csv')}")
    print(f"  flipped predictions:   {os.path.join(OUTPUT_DIR, 'flipped/predictions/ensemble_predictions.csv')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
