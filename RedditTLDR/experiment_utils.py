# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions for the context selection experiment.
Shared across Phase 1 (Initialization) and Phase 2 (Experimental Loop).
"""

import pickle
import random
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# Add parent directory for LoRe utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import solve_regularized, learn_multiple_few_shot, eval_multiple


# ==============================================================================
# Configuration
# ==============================================================================

CONFIG = {
    # Dataset Limits
    'total_pool_size': 50,    # The pool we split into Demo/Val
    'test_set_size': 50,      # Held-out Test Set

    # Experimental Variables
    'n_trials': 100,          # Search budget (How many contexts to check)
    'bootstrap_seeds': 5,     # Number of Cross-Validation folds

    # The loops we want to run
    'shot_counts': [2, 4, 8],
    'split_configs': [        # (Demo Size, Val Size)
        (10, 40),             # Split A: Low Resource, High Validation
        (20, 30),             # Split B
        (30, 20),             # Split C
        (40, 10),             # Split D: High Resource, Low Validation
    ],

    # LoRe training parameters
    'basis_rank': 5,          # Number of basis vectors K for LoRe
    'alpha': 0,               # Regularization parameter
    'num_iterations': 1000,   # Training iterations for basis learning
    'learning_rate': 0.5,     # Learning rate for basis learning
    'personalize_iterations': 500,  # Iterations for weight learning
    'personalize_lr': 0.1,    # Learning rate for weight learning

    # Paths
    'train_embeddings_path': 'tldr_embeddings_train.pkl',
    'val_embeddings_path': 'tldr_embeddings_val.pkl',
    'checkpoint_dir': 'checkpoints/context_selection',
    'results_dir': 'results/context_selection',
}


# ==============================================================================
# Data Loading and Processing
# ==============================================================================

def load_embeddings(train_path: str, val_path: str) -> Tuple[Dict, Dict]:
    """Load pre-computed embeddings from pickle files."""
    with open(train_path, 'rb') as f:
        worker_results_train = pickle.load(f)
    with open(val_path, 'rb') as f:
        worker_results_val = pickle.load(f)
    return worker_results_train, worker_results_val


def get_common_workers(worker_results_train: Dict, worker_results_val: Dict,
                       min_entries: int = 2) -> Tuple[Dict, Dict, List[str]]:
    """
    Get workers that appear in both train and val sets with sufficient entries.

    Returns:
        filtered_train: Filtered train results
        filtered_val: Filtered val results
        common_workers: List of common worker IDs
    """
    # Sort and filter by minimum entries (skip first min_entries workers with fewest examples)
    filtered_train = dict(sorted(
        worker_results_train.items(),
        key=lambda item: len(item[1]),
        reverse=False
    )[min_entries:])
    filtered_val = dict(sorted(
        worker_results_val.items(),
        key=lambda item: len(item[1]),
        reverse=False
    )[min_entries:])

    train_workers = set(filtered_train.keys())
    val_workers = set(filtered_val.keys())
    common_workers = list(train_workers & val_workers)

    return filtered_train, filtered_val, common_workers


def split_workers_fixed(common_workers: List[str], seed: int = 0) -> Tuple[List[str], List[str]]:
    """
    Split workers into train and test sets (50/50 split).
    This mirrors the logic in train_basis.py.

    Returns:
        train_workers: Workers for training pool
        test_workers: Workers for held-out test
    """
    random.seed(seed)
    shuffled = common_workers.copy()
    random.shuffle(shuffled)
    split_idx = len(shuffled) // 2
    return shuffled[:split_idx], shuffled[split_idx:]


def extract_feature_diff(data_entry: Dict, device: torch.device) -> torch.Tensor:
    """Extract embedding difference (winning - losing) for a single entry."""
    winning_emb = data_entry['embeddings']['winning'][0]
    losing_emb = data_entry['embeddings']['losing'][0]
    diff = winning_emb - losing_emb
    return torch.tensor(diff, dtype=torch.float32).to(device)


def build_feature_tensor(data_list: List[Dict], device: torch.device) -> torch.Tensor:
    """Build a feature tensor from a list of data entries."""
    features = [extract_feature_diff(entry, device) for entry in data_list]
    return torch.stack(features)


def random_sample_indices(N: int, T: int) -> Tuple[List[int], List[int]]:
    """
    Randomly sample T indices from 0 to N-1 without replacement.
    Returns (sampled_indices, remaining_indices).
    """
    all_indices = list(range(N))
    sampled = random.sample(all_indices, min(T, N))
    remaining = list(set(all_indices) - set(sampled))
    return sampled, remaining


# ==============================================================================
# Weight Learning Functions
# ==============================================================================

def learn_weights_from_context(context_features: torch.Tensor, V: torch.Tensor,
                               num_iterations: int = 500,
                               learning_rate: float = 0.1) -> torch.Tensor:
    """
    Learn weight vector w from a k-shot context using fixed V.

    This is a single-user version of learn_multiple_few_shot.

    Args:
        context_features: K-shot context tensor of shape (K, d)
        V: Fixed basis matrix of shape (d, B)
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for Adam optimizer

    Returns:
        Learned weight vector w of shape (B,) with softmax applied
    """
    device = context_features.device
    num_basis = V.shape[1]

    w_logits = torch.nn.Parameter(torch.randn(num_basis, device=device))
    optimizer = optim.Adam([w_logits], lr=learning_rate)

    for _ in range(num_iterations):
        optimizer.zero_grad()
        w = F.softmax(w_logits, dim=0)
        V_w = V @ w
        logits = (context_features @ V_w) / 100.0
        loss = -torch.log(torch.sigmoid(logits)).mean()
        loss.backward()
        optimizer.step()

    return F.softmax(w_logits, dim=0).detach()


# ==============================================================================
# Evaluation Functions
# ==============================================================================

def compute_accuracy(features: torch.Tensor, V: torch.Tensor, w: torch.Tensor) -> float:
    """
    Compute accuracy on a set of features using V and w.

    Args:
        features: Tensor of shape (N, d) containing feature differences
        V: Basis matrix of shape (d, K)
        w: Weight vector of shape (K,) or (K, 1)

    Returns:
        Accuracy (fraction of positive predictions)
    """
    if w.dim() == 1:
        V_w = V @ w
    else:
        V_w = V @ w.squeeze()

    scores = features @ V_w
    num_positive = (scores > 0).sum().item()
    return num_positive / scores.numel()


def compute_likelihood_score(features: torch.Tensor, V: torch.Tensor,
                             w: torch.Tensor, scale: float = 100.0) -> float:
    """
    Compute log-likelihood score for evaluation on validation features.
    Higher is better (less negative).

    Args:
        features: Validation features tensor of shape (N, d)
        V: Basis matrix of shape (d, K)
        w: Weight vector
        scale: Scaling factor for logits

    Returns:
        Mean log-likelihood score
    """
    if w.dim() == 1:
        V_w = V @ w
    else:
        V_w = V @ w.squeeze()

    logits = (features @ V_w) / scale
    log_likelihood = torch.log(torch.sigmoid(logits))
    return log_likelihood.mean().item()


# ==============================================================================
# Saving and Loading
# ==============================================================================

def save_results(results: Any, path: str):
    """Save results to JSON file."""
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)


def load_results(path: str) -> Any:
    """Load results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_checkpoint(obj: Any, path: str):
    """Save PyTorch checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def load_checkpoint(path: str, device: torch.device = None) -> Any:
    """Load PyTorch checkpoint."""
    if device is not None:
        return torch.load(path, map_location=device)
    return torch.load(path)


# ==============================================================================
# Logging and Printing
# ==============================================================================

def print_experiment_header(split_config: Tuple[int, int], seed: int, k: int):
    """Print header for current experiment configuration."""
    n_demo, n_val = split_config
    print(f"\n{'='*60}")
    print(f"Split: Demo={n_demo}, Val={n_val} | Seed={seed} | K={k}")
    print(f"{'='*60}")


def print_results_summary(results_log: List[Dict]):
    """Print a summary table of all results."""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)

    # Group by split config
    from collections import defaultdict
    by_split = defaultdict(list)
    for r in results_log:
        key = (r['split_demo'], r['split_val'])
        by_split[key].append(r)

    for (n_demo, n_val), records in sorted(by_split.items()):
        print(f"\n>>> Split: Demo={n_demo}, Val={n_val}")
        print("-"*70)
        print(f"{'K':>4} | {'Method 1 (Random@Demo)':>25} | {'Method 2 (Optimized)':>18} | {'Method 3 (Full)':>18}")
        print("-"*70)

        # Group by K and average across seeds
        by_k = defaultdict(list)
        for r in records:
            by_k[r['k']].append(r)

        for k in sorted(by_k.keys()):
            k_records = by_k[k]

            m1_means = [r['method_1_strict_mean_acc'] for r in k_records]
            m1_stds = [r['method_1_strict_std'] for r in k_records]
            m2_accs = [r['method_2_optimized_acc'] for r in k_records]
            m3_means = [r['method_3_full_mean_acc'] for r in k_records]

            m1_str = f"{np.mean(m1_means)*100:.2f}% ± {np.mean(m1_stds)*100:.2f}%"
            m2_str = f"{np.mean(m2_accs)*100:.2f}%"
            m3_str = f"{np.mean(m3_means)*100:.2f}%"

            print(f"{k:>4} | {m1_str:>25} | {m2_str:>18} | {m3_str:>18}")

    print("\n" + "="*80)
