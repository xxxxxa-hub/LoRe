# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Phase 2: The Experimental Loop

This script runs the three comparison methods across different configurations
for EACH UNSEEN USER. Each user has their own:
- D_pool (train_features_unseen[i]): to split into Demo/Val
- D_test (test_features_sparse_unseen[i]): held-out test

Loops:
- Per User: Each unseen user is processed independently
- Outer Loop: Data Availability Splits (10/40, 20/30, 30/20, 40/10 Demo/Val)
- Middle Loop: Bootstrap Seeds (5-fold cross-validation)
- Inner Loop: Context Size (K = 2, 4, 8 shots)

Three Methods:
1. Strict Baseline (Random @ Demo): Random sampling from demo set only
2. Optimized Strategy (Your Method): Use validation to select best context
3. Dem+Val Baseline (Random @ Full): Random sampling from full pool
"""

import random
import os
import sys
import argparse
from typing import List, Dict, Any
from collections import defaultdict

import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from experiment_utils import (
    CONFIG,
    load_checkpoint,
    save_results,
    learn_weights_from_context,
    compute_accuracy,
    compute_likelihood_score,
    print_results_summary,
)


# ==============================================================================
# The Three Experimental Methods
# ==============================================================================

def method_1_strict_baseline(
    demo_features: torch.Tensor,
    test_features: torch.Tensor,
    V: torch.Tensor,
    k: int,
    n_trials: int,
    seed: int,
    num_iterations: int = 500,
    learning_rate: float = 0.1,
) -> Dict[str, Any]:
    """
    Method 1: The Strict Baseline (Random @ Demo)

    Concept: Represents a standard user who only has access to the small demo
    set and has no validation mechanism.

    Procedure:
    1. Randomly sample N_trials different K-shot contexts from D_demo
    2. Evaluate all of them on D_test
    3. Report Mean Accuracy and Std Dev
    """
    rng = random.Random(seed)
    n_demo = demo_features.shape[0]

    if k > n_demo:
        return {'mean_accuracy': None, 'std_accuracy': None, 'all_accuracies': [], 'skipped': True}

    accuracies = []

    for trial in range(n_trials):
        # 1. Sample K from Demo Set
        indices = rng.sample(range(n_demo), k)
        context = demo_features[indices]

        # 2. Learn weights from context
        w = learn_weights_from_context(context, V, num_iterations, learning_rate)

        # 3. Evaluate on Test (blind, no validation)
        acc = compute_accuracy(test_features, V, w)
        accuracies.append(acc)

    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'all_accuracies': accuracies,
        'skipped': False,
    }


def method_2_optimized_selection(
    demo_features: torch.Tensor,
    val_features: torch.Tensor,
    test_features: torch.Tensor,
    V: torch.Tensor,
    k: int,
    n_trials: int,
    seed: int,
    num_iterations: int = 500,
    learning_rate: float = 0.1,
) -> Dict[str, Any]:
    """
    Method 2: The Optimized Strategy (Your Method)

    Concept: Uses the Validation set to filter "bad" contexts and pick the "winner."

    Procedure:
    1. Randomly sample N_trials different K-shot contexts from D_demo
    2. Evaluate them on D_val (NOT Test!) to get Likelihood Score
    3. Select Top-1 Context (C_best) with highest Likelihood
    4. Report Single Accuracy of C_best on D_test
    """
    rng = random.Random(seed)
    n_demo = demo_features.shape[0]

    if k > n_demo:
        return {'test_accuracy': None, 'best_val_score': None, 'best_context_indices': [], 'skipped': True}

    candidates = []  # List of (indices, w, val_score)

    # Search Phase: Try N_trials contexts
    for trial in range(n_trials):
        # 1. Sample K from Demo Set
        indices = rng.sample(range(n_demo), k)
        context = demo_features[indices]

        # 2. Learn weights from context
        w = learn_weights_from_context(context, V, num_iterations, learning_rate)

        # 3. Score on Validation (Likelihood) - NOT TEST!
        val_score = compute_likelihood_score(val_features, V, w)
        candidates.append((indices, w, val_score))

    # Selection Phase: Pick best by validation score
    best_candidate = max(candidates, key=lambda x: x[2])
    best_indices, best_w, best_val_score = best_candidate

    # Final Test Phase: Evaluate winner on Test Data
    test_accuracy = compute_accuracy(test_features, V, best_w)

    return {
        'test_accuracy': test_accuracy,
        'best_val_score': best_val_score,
        'best_context_indices': list(best_indices),
        'skipped': False,
    }


def method_3_full_pool_baseline(
    demo_features: torch.Tensor,
    val_features: torch.Tensor,
    test_features: torch.Tensor,
    V: torch.Tensor,
    k: int,
    n_trials: int,
    seed: int,
    num_iterations: int = 500,
    learning_rate: float = 0.1,
) -> Dict[str, Any]:
    """
    Method 3: The "Dem+Val" Baseline (Full Context)

    Concept: Represents the "Upper Bound" if we use ALL available data
    (Demo + Val) as the context, without any sampling.

    Procedure:
    1. Use the entire full pool (Demo + Val) as context
    2. Learn weights from the full context
    3. Evaluate on D_test and report accuracy

    Note: k and n_trials parameters are unused since we use the full set.
    """
    # Combine Demo and Val into full pool - use ALL of it as context
    full_pool = torch.cat([demo_features, val_features], dim=0)

    # Learn weights from the FULL context (no sampling)
    w = learn_weights_from_context(full_pool, V, num_iterations, learning_rate)

    # Evaluate on Test
    acc = compute_accuracy(test_features, V, w)

    return {
        'mean_accuracy': acc,
        'std_accuracy': 0.0,  # No variance since we use the full set deterministically
        'all_accuracies': [acc],
        'skipped': False,
        'context_size': full_pool.shape[0],
    }


# ==============================================================================
# Per-User Experiment Runner
# ==============================================================================

def run_experiment_for_user(
    user_idx: int,
    pool_features: torch.Tensor,
    test_features: torch.Tensor,
    V: torch.Tensor,
    config: Dict,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run the full experimental loop for a single unseen user.

    Args:
        user_idx: Index of the unseen user
        pool_features: D_pool for this user (N_pool, d)
        test_features: D_test for this user (N_test, d)
        V: Basis matrix (d, B)
        config: Configuration dictionary
        verbose: Whether to print progress

    Returns:
        List of result records for this user
    """
    results_log = []

    n_trials = config['n_trials']
    bootstrap_seeds = config['bootstrap_seeds']
    shot_counts = config['shot_counts']
    split_configs = config['split_configs']
    num_iterations = config['personalize_iterations']
    learning_rate = config['personalize_lr']

    pool_size = pool_features.shape[0]
    pool_indices = list(range(pool_size))

    if verbose:
        print(f"\n  User {user_idx}: pool_size={pool_size}, test_size={test_features.shape[0]}")

    # --------------------------------------------------------------------------
    # Outer Loop: Data Availability Splits (e.g., 0.2/0.8 Demo/Val ratios)
    # --------------------------------------------------------------------------
    for (demo_ratio, val_ratio) in split_configs:
        # Compute actual counts from ratios based on pool size
        n_demo = int(pool_size * demo_ratio)
        n_val = int(pool_size * val_ratio)

        # Ensure at least 1 sample in each set if ratio > 0
        if demo_ratio > 0 and n_demo == 0:
            n_demo = 1
        if val_ratio > 0 and n_val == 0:
            n_val = 1

        # Ensure we don't exceed pool size
        if n_demo + n_val > pool_size:
            n_val = pool_size - n_demo

        if verbose:
            print(f"    Split ratio {demo_ratio:.2f}/{val_ratio:.2f} -> n_demo={n_demo}, n_val={n_val} (pool_size={pool_size})")

        # ----------------------------------------------------------------------
        # Middle Loop: Bootstrap Seeds (Cross-Validation)
        # ----------------------------------------------------------------------
        for seed_idx in range(bootstrap_seeds):
            # Deterministic Data Splitting
            rng = random.Random(seed_idx)
            shuffled_indices = pool_indices.copy()
            rng.shuffle(shuffled_indices)

            # The Critical Split
            demo_indices = shuffled_indices[:n_demo]
            val_indices = shuffled_indices[n_demo:n_demo + n_val]

            demo_features = pool_features[demo_indices]
            val_features = pool_features[val_indices]

            # ------------------------------------------------------------------
            # Inner Loop: Shot Count (K)
            # ------------------------------------------------------------------
            for k in shot_counts:
                if k > n_demo:
                    continue

                # Record for this configuration
                record = {
                    "user_idx": user_idx,
                    "pool_size": pool_size,
                    "test_size": test_features.shape[0],
                    "demo_ratio": demo_ratio,
                    "val_ratio": val_ratio,
                    "split_demo": n_demo,
                    "split_val": n_val,
                    "seed": seed_idx,
                    "k": k,
                    "n_trials": n_trials,
                }

                # Unique seed per config
                method_seed = user_idx * 10000 + seed_idx * 1000 + k

                # ==============================================================
                # METHOD 1: STRICT BASELINE (Random @ Demo)
                # ==============================================================
                m1_results = method_1_strict_baseline(
                    demo_features=demo_features,
                    test_features=test_features,
                    V=V,
                    k=k,
                    n_trials=n_trials,
                    seed=method_seed,
                    num_iterations=num_iterations,
                    learning_rate=learning_rate,
                )
                record['method_1_strict_mean_acc'] = m1_results['mean_accuracy']
                record['method_1_strict_std'] = m1_results['std_accuracy']

                # ==============================================================
                # METHOD 2: OPTIMIZED STRATEGY (Your Method)
                # ==============================================================
                m2_results = method_2_optimized_selection(
                    demo_features=demo_features,
                    val_features=val_features,
                    test_features=test_features,
                    V=V,
                    k=k,
                    n_trials=n_trials,
                    seed=method_seed,
                    num_iterations=num_iterations,
                    learning_rate=learning_rate,
                )
                record['method_2_optimized_acc'] = m2_results['test_accuracy']
                record['method_2_best_val_score'] = m2_results['best_val_score']

                # ==============================================================
                # METHOD 3: FULL DATA BASELINE (Random @ Demo+Val)
                # ==============================================================
                m3_results = method_3_full_pool_baseline(
                    demo_features=demo_features,
                    val_features=val_features,
                    test_features=test_features,
                    V=V,
                    k=k,
                    n_trials=n_trials,
                    seed=method_seed,
                    num_iterations=num_iterations,
                    learning_rate=learning_rate,
                )
                record['method_3_full_mean_acc'] = m3_results['mean_accuracy']
                record['method_3_full_std'] = m3_results['std_accuracy']

                results_log.append(record)

    return results_log


def run_experiment(
    train_features_unseen: List[torch.Tensor],
    test_features_sparse_unseen: List[torch.Tensor],
    V: torch.Tensor,
    config: Dict,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run the full experimental loop for ALL unseen users.

    Args:
        train_features_unseen: List of D_pool tensors, one per unseen user
        test_features_sparse_unseen: List of D_test tensors, one per unseen user
        V: Basis matrix (d, B)
        config: Configuration dictionary
        verbose: Whether to print progress

    Returns:
        List of result records across all users
    """
    all_results = []
    n_unseen = len(train_features_unseen)

    print(f"\nRunning experiment for {n_unseen} unseen users...")
    print(f"Config: n_trials={config['n_trials']}, bootstrap_seeds={config['bootstrap_seeds']}")
    print(f"Shot counts: {config['shot_counts']}")
    print(f"Split configs: {config['split_configs']}")

    for user_idx in range(n_unseen):
        pool_features = train_features_unseen[user_idx]
        test_features = test_features_sparse_unseen[user_idx]

        user_results = run_experiment_for_user(
            user_idx=user_idx,
            pool_features=pool_features,
            test_features=test_features,
            V=V,
            config=config,
            verbose=verbose,
        )
        all_results.extend(user_results)

        if verbose and (user_idx + 1) % 10 == 0:
            print(f"  Completed {user_idx + 1}/{n_unseen} users")

    return all_results


def aggregate_results(results_log: List[Dict]) -> Dict:
    """
    Aggregate results across all users for summary statistics.
    """
    # Group by (demo_ratio, val_ratio, k)
    grouped = defaultdict(list)
    for r in results_log:
        key = (r['demo_ratio'], r['val_ratio'], r['k'])
        grouped[key].append(r)

    summary = {}
    for key, records in grouped.items():
        demo_ratio, val_ratio, k = key

        # Filter out None values (skipped experiments)
        m1_accs = [r['method_1_strict_mean_acc'] for r in records if r['method_1_strict_mean_acc'] is not None]
        m2_accs = [r['method_2_optimized_acc'] for r in records if r['method_2_optimized_acc'] is not None]
        m3_accs = [r['method_3_full_mean_acc'] for r in records if r['method_3_full_mean_acc'] is not None]

        summary[f"demo{demo_ratio:.2f}_val{val_ratio:.2f}_k{k}"] = {
            'demo_ratio': demo_ratio,
            'val_ratio': val_ratio,
            'k': k,
            'n_records': len(records),
            'method_1': {
                'mean': np.mean(m1_accs) if m1_accs else None,
                'std': np.std(m1_accs) if m1_accs else None,
            },
            'method_2': {
                'mean': np.mean(m2_accs) if m2_accs else None,
                'std': np.std(m2_accs) if m2_accs else None,
            },
            'method_3': {
                'mean': np.mean(m3_accs) if m3_accs else None,
                'std': np.std(m3_accs) if m3_accs else None,
            },
        }

    return summary


def print_summary_table(summary: Dict):
    """Print a formatted summary table."""
    print("\n" + "="*90)
    print("AGGREGATED RESULTS (Averaged across all users and bootstrap seeds)")
    print("="*90)

    # Group by split ratio
    by_split = defaultdict(list)
    for key, data in summary.items():
        split_key = (data['demo_ratio'], data['val_ratio'])
        by_split[split_key].append(data)

    for (demo_ratio, val_ratio), entries in sorted(by_split.items()):
        print(f"\n>>> Split Ratio: Demo={demo_ratio:.0%}, Val={val_ratio:.0%}")
        print("-"*80)
        print(f"{'K':>4} | {'Method 1 (Random@Demo)':>25} | {'Method 2 (Optimized)':>22} | {'Method 3 (Full)':>22}")
        print("-"*80)

        for entry in sorted(entries, key=lambda x: x['k']):
            k = entry['k']
            m1 = entry['method_1']
            m2 = entry['method_2']
            m3 = entry['method_3']

            m1_str = f"{m1['mean']*100:.2f}% ± {m1['std']*100:.2f}%" if m1['mean'] else "N/A"
            m2_str = f"{m2['mean']*100:.2f}% ± {m2['std']*100:.2f}%" if m2['mean'] else "N/A"
            m3_str = f"{m3['mean']*100:.2f}% ± {m3['std']*100:.2f}%" if m3['mean'] else "N/A"

            print(f"{k:>4} | {m1_str:>25} | {m2_str:>22} | {m3_str:>22}")

    print("\n" + "="*90)


def run_phase2(args):
    """
    Main Phase 2 execution.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoints from Phase 1
    print("\n" + "="*60)
    print("Loading Phase 1 Checkpoints")
    print("="*60)

    V = load_checkpoint(os.path.join(args.checkpoint_dir, 'V_basis.pt'), device)
    unseen_data = load_checkpoint(os.path.join(args.checkpoint_dir, 'unseen_user_data.pt'), device)

    train_features_unseen = unseen_data['train_features_unseen']  # D_pool per user
    test_features_sparse_unseen = unseen_data['test_features_sparse_unseen']  # D_test per user
    N_unseen = unseen_data['N_unseen']

    # Move to device
    train_features_unseen = [t.to(device) for t in train_features_unseen]
    test_features_sparse_unseen = [t.to(device) for t in test_features_sparse_unseen]

    print(f"V shape: {V.shape}")
    print(f"N_unseen: {N_unseen}")
    print(f"Pool sizes: min={min(t.shape[0] for t in train_features_unseen)}, max={max(t.shape[0] for t in train_features_unseen)}")
    print(f"Test sizes: min={min(t.shape[0] for t in test_features_sparse_unseen)}, max={max(t.shape[0] for t in test_features_sparse_unseen)}")

    # Build config from args
    config = {
        'n_trials': args.n_trials,
        'bootstrap_seeds': args.bootstrap_seeds,
        'shot_counts': [int(x) for x in args.shot_counts.split(',')],
        'split_configs': [
            tuple(map(float, s.split('/'))) for s in args.split_configs.split(',')
        ],
        'personalize_iterations': args.personalize_iterations,
        'personalize_lr': args.personalize_lr,
    }

    print("\n" + "="*60)
    print("Experiment Configuration")
    print("="*60)
    print(f"N_trials (search budget): {config['n_trials']}")
    print(f"Bootstrap seeds: {config['bootstrap_seeds']}")
    print(f"Shot counts (K): {config['shot_counts']}")
    print(f"Split configs: {config['split_configs']}")

    # Run the experiment
    print("\n" + "="*60)
    print("Running Experiment")
    print("="*60)

    results_log = run_experiment(
        train_features_unseen=train_features_unseen,
        test_features_sparse_unseen=test_features_sparse_unseen,
        V=V,
        config=config,
        verbose=args.verbose,
    )

    # Aggregate and print summary
    summary = aggregate_results(results_log)
    print_summary_table(summary)

    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, 'context_selection_results.json')
    save_results(results_log, results_path)
    print(f"\nDetailed results saved to: {results_path}")

    summary_path = os.path.join(args.results_dir, 'context_selection_summary.json')
    save_results(summary, summary_path)
    print(f"Summary saved to: {summary_path}")

    return results_log, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Run context selection experiment")

    # Paths
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints/context_selection',
                        help='Directory containing Phase 1 checkpoints')
    parser.add_argument('--results_dir', type=str,
                        default='results/context_selection',
                        help='Directory to save results')

    # Experiment parameters
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Search budget (number of contexts to try)')
    parser.add_argument('--bootstrap_seeds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--shot_counts', type=str, default='2,4,8',
                        help='Comma-separated list of shot counts')
    parser.add_argument('--split_configs', type=str,
                        default='0.2/0.8,0.4/0.6,0.6/0.4,0.8/0.2',
                        help='Comma-separated Demo/Val ratio splits (e.g., 0.2/0.8 means 20%% demo, 80%% val)')

    # Weight learning parameters
    parser.add_argument('--personalize_iterations', type=int, default=500,
                        help='Iterations for weight learning')
    parser.add_argument('--personalize_lr', type=float, default=0.1,
                        help='Learning rate for weight learning')

    # Other
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')

    args = parser.parse_args()
    run_phase2(args)
