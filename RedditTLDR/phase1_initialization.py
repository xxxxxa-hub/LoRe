# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Phase 1: Initialization (Pre-computation)

This script follows the data flow from vary_fewshot.py:

1. Fix User Splits:
   - Seen Users (train_workers): Used to train basis V and W
   - Unseen Users (test_workers): Used for context selection experiments

2. For Seen Users:
   - train_features: from train pkl (up to 150 samples per user) → train V, W
   - test_features_sparse: from val pkl → not used in this experiment

3. For Unseen Users:
   - train_features_unseen: from train pkl (up to T_unseen samples) → D_pool (to split into Demo/Val)
   - test_features_sparse_unseen: from val pkl → D_test (held-out, never seen during selection)

4. Basis Generation:
   - Train V and W on train_features (seen users) using LoRe
   - This ensures no test/unseen user info leaks into basis
"""

import pickle
import random
import os
import sys
import argparse

import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import solve_regularized
from experiment_utils import (
    CONFIG,
    save_checkpoint,
    save_results,
)


def load_reward_model_V(device: torch.device) -> torch.Tensor:
    """
    Load the final layer weight from the Skywork Reward Model.
    This is V_final (the pretrained reward direction).
    """
    from transformers import AutoModel

    model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    print(f"Loading reward model: {model_name}")

    rm = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )

    # Get the last linear layer
    last_linear_layer = None
    for name, module in rm.named_modules():
        if isinstance(module, torch.nn.Linear):
            last_linear_layer = module

    V_final = last_linear_layer.weight[:, 0].to(device).to(torch.float32).reshape(-1, 1)
    print(f"V_final shape: {V_final.shape}")

    # Clean up
    del rm
    torch.cuda.empty_cache()

    return V_final


def random_sample(N: int, T: int):
    """
    Randomly sample T numbers between 0 and N-1 without replacement.
    Returns (sampled_indices, remaining_indices).
    """
    all_numbers = list(range(N))
    samples = random.sample(all_numbers, min(T, N))
    remaining = list(set(all_numbers) - set(samples))
    return samples, remaining


def process_data_like_vary_fewshot(
    worker_results_train: dict,
    worker_results_test: dict,
    T_seen: int = 150,
    T_unseen: int = 50,
    seed: int = 0,
    device: torch.device = None,
):
    """
    Process data following the exact flow of vary_fewshot.py.

    Args:
        worker_results_train: Training embeddings dict (from train pkl)
        worker_results_test: Test embeddings dict (from val pkl)
        T_seen: Max samples per seen user for training V
        T_unseen: Max samples per unseen user for D_pool
        seed: Random seed
        device: Torch device

    Returns:
        Dictionary containing all processed features and metadata
    """
    random.seed(seed)

    # Sort and filter (skip first 2 workers with fewest entries)
    all_worker_results_train = dict(sorted(
        worker_results_train.items(),
        key=lambda item: len(item[1]),
        reverse=False
    )[2:])
    all_worker_results_test = dict(sorted(
        worker_results_test.items(),
        key=lambda item: len(item[1]),
        reverse=False
    )[2:])

    # Get common workers and split into seen/unseen
    train_worker_set = set(worker_results_train.keys())
    test_worker_set = set(worker_results_test.keys())
    common_workers = list(train_worker_set & test_worker_set)
    random.shuffle(common_workers)

    split = len(common_workers) // 2
    print(f"Total common workers: {len(common_workers)}, Split at: {split}")

    seen_worker_ids = common_workers[:split]   # "train_workers" in original
    unseen_worker_ids = common_workers[split:]  # "test_workers" in original

    # Process features
    train_features = []          # Seen users: for training V
    test_features_sparse = []    # Seen users: held-out (not used in experiment)

    train_features_unseen = []   # Unseen users: D_pool (to split into Demo/Val)
    test_features_sparse_unseen = []  # Unseen users: D_test (held-out)

    N_seen = 0
    N_unseen = 0

    for worker_id, data in all_worker_results_train.items():
        if worker_id in seen_worker_ids:
            # SEEN USER: extract train features for basis learning
            temp = []
            T = min(len(data), T_seen)
            idx, _ = random_sample(len(data), T)
            for i in idx:
                x = data[i]['embeddings']['winning'][0] - data[i]['embeddings']['losing'][0]
                temp.append(torch.tensor(x, dtype=torch.float32).to(device))
            train_features.append(torch.stack(temp))

            # Also get their test features (from val pkl)
            temp2 = []
            for test_data in all_worker_results_test[worker_id]:
                x = test_data['embeddings']['winning'][0] - test_data['embeddings']['losing'][0]
                temp2.append(torch.tensor(x, dtype=torch.float32).to(device))
            test_features_sparse.append(torch.stack(temp2))
            N_seen += 1

        elif worker_id in unseen_worker_ids:
            # UNSEEN USER: extract features for context selection experiment
            temp = []
            T = min(len(data), T_unseen)
            idx, _ = random_sample(len(data), T)
            for i in idx:
                x = data[i]['embeddings']['winning'][0] - data[i]['embeddings']['losing'][0]
                temp.append(torch.tensor(x, dtype=torch.float32).to(device))
            train_features_unseen.append(torch.stack(temp))

            # Test features (from val pkl) - this is D_test
            temp2 = []
            for test_data in all_worker_results_test[worker_id]:
                x = test_data['embeddings']['winning'][0] - test_data['embeddings']['losing'][0]
                temp2.append(torch.tensor(x, dtype=torch.float32).to(device))
            test_features_sparse_unseen.append(torch.stack(temp2))
            N_unseen += 1

    print(f"Seen users (N_seen): {N_seen}")
    print(f"Unseen users (N_unseen): {N_unseen}")
    print(f"train_features: {len(train_features)} users")
    print(f"train_features_unseen (D_pool): {len(train_features_unseen)} users")
    print(f"test_features_sparse_unseen (D_test): {len(test_features_sparse_unseen)} users")

    return {
        # Seen users (for basis training)
        'train_features': train_features,
        'test_features_sparse': test_features_sparse,
        'seen_worker_ids': seen_worker_ids,
        'N_seen': N_seen,

        # Unseen users (for context selection experiment)
        'train_features_unseen': train_features_unseen,  # D_pool per user
        'test_features_sparse_unseen': test_features_sparse_unseen,  # D_test per user
        'unseen_worker_ids': unseen_worker_ids,
        'N_unseen': N_unseen,
    }


def train_basis_on_seen_users(
    train_features: list,
    V_final: torch.Tensor,
    num_basis: int,
    alpha: float,
    num_iterations: int,
    learning_rate: float,
) -> tuple:
    """
    Train LoRe basis vectors V and weights W on SEEN users' data.

    This ensures no unseen user information leaks into the basis.

    Args:
        train_features: List of feature tensors, one per seen user
        V_final: Pretrained reward direction
        num_basis: Number of basis vectors (rank K)
        alpha: Regularization parameter
        num_iterations: Training iterations
        learning_rate: Learning rate

    Returns:
        W: Learned weights for seen users (N_seen, K)
        V: Learned basis matrix (d, K)
    """
    print(f"\nTraining basis on {len(train_features)} seen users")
    print(f"Rank K={num_basis}, alpha={alpha}")

    W, V = solve_regularized(
        V_final, alpha, train_features, num_basis,
        num_iterations=num_iterations,
        learning_rate=learning_rate
    )

    print(f"Learned V shape: {V.shape}")
    print(f"Learned W shape: {W.shape}")

    return W, V


def run_phase1(args):
    """
    Main Phase 1 execution.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Load V_final from reward model
    print("\n" + "="*60)
    print("Step 1: Loading Reward Model V_final")
    print("="*60)
    V_final = load_reward_model_V(device)

    # 2. Load embeddings
    print("\n" + "="*60)
    print("Step 2: Loading Embeddings")
    print("="*60)
    with open(args.train_embeddings, 'rb') as f:
        worker_results_train = pickle.load(f)
    with open(args.val_embeddings, 'rb') as f:
        worker_results_test = pickle.load(f)

    print(f"Loaded {len(worker_results_train)} workers from train pkl")
    print(f"Loaded {len(worker_results_test)} workers from val pkl")

    # 3. Process data following vary_fewshot.py flow
    print("\n" + "="*60)
    print("Step 3: Processing Data (following vary_fewshot.py)")
    print("="*60)
    data = process_data_like_vary_fewshot(
        worker_results_train,
        worker_results_test,
        T_seen=args.T_seen,
        T_unseen=args.T_unseen,
        seed=args.seed,
        device=device,
    )

    # 4. Train basis on SEEN users (no leakage from unseen users)
    print("\n" + "="*60)
    print("Step 4: Training Basis on Seen Users")
    print("="*60)
    W, V = train_basis_on_seen_users(
        data['train_features'],
        V_final,
        num_basis=args.num_basis,
        alpha=args.alpha,
        num_iterations=args.num_iterations,
        learning_rate=args.learning_rate,
    )

    # 5. Save everything
    print("\n" + "="*60)
    print("Step 5: Saving Checkpoints")
    print("="*60)

    # Save V and W (learned from seen users)
    save_checkpoint(V, os.path.join(args.checkpoint_dir, 'V_basis.pt'))
    save_checkpoint(W, os.path.join(args.checkpoint_dir, 'W_seen_users.pt'))
    save_checkpoint(V_final, os.path.join(args.checkpoint_dir, 'V_final.pt'))

    # Save unseen user data for Phase 2 experiment
    # train_features_unseen = D_pool (to be split into Demo/Val)
    # test_features_sparse_unseen = D_test (held-out)
    save_checkpoint({
        'train_features_unseen': data['train_features_unseen'],  # D_pool per user
        'test_features_sparse_unseen': data['test_features_sparse_unseen'],  # D_test per user
        'N_unseen': data['N_unseen'],
    }, os.path.join(args.checkpoint_dir, 'unseen_user_data.pt'))

    # Save metadata
    metadata = {
        'seen_worker_ids': data['seen_worker_ids'],
        'unseen_worker_ids': data['unseen_worker_ids'],
        'N_seen': data['N_seen'],
        'N_unseen': data['N_unseen'],
        'T_seen': args.T_seen,
        'T_unseen': args.T_unseen,
        'seed': args.seed,
        'num_basis': args.num_basis,
        'alpha': args.alpha,
        # Store pool sizes per unseen user
        'pool_sizes': [t.shape[0] for t in data['train_features_unseen']],
        'test_sizes': [t.shape[0] for t in data['test_features_sparse_unseen']],
    }
    save_results(metadata, os.path.join(args.checkpoint_dir, 'metadata.json'))

    print(f"\nPhase 1 Complete!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"  - V_basis.pt: Learned basis matrix (from seen users)")
    print(f"  - W_seen_users.pt: Learned weights for seen users")
    print(f"  - V_final.pt: Pretrained reward direction")
    print(f"  - unseen_user_data.pt: D_pool and D_test for unseen users")
    print(f"  - metadata.json: Configuration and worker IDs")

    print(f"\nUnseen users ready for Phase 2:")
    print(f"  - N_unseen: {data['N_unseen']} users")
    print(f"  - D_pool sizes: min={min(metadata['pool_sizes'])}, max={max(metadata['pool_sizes'])}, mean={np.mean(metadata['pool_sizes']):.1f}")
    print(f"  - D_test sizes: min={min(metadata['test_sizes'])}, max={max(metadata['test_sizes'])}, mean={np.mean(metadata['test_sizes']):.1f}")

    return V, W, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Initialize datasets and train basis")

    # Data paths
    parser.add_argument('--train_embeddings', type=str,
                        default='tldr_embeddings_train.pkl',
                        help='Path to training embeddings pickle')
    parser.add_argument('--val_embeddings', type=str,
                        default='tldr_embeddings_val.pkl',
                        help='Path to validation embeddings pickle')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints/',
                        help='Directory to save checkpoints')

    # Data sampling parameters (following vary_fewshot.py)
    parser.add_argument('--T_seen', type=int, default=150,
                        help='Max samples per seen user for training V')
    parser.add_argument('--T_unseen', type=int, default=100,
                        help='Max samples per unseen user for D_pool')

    # LoRe parameters
    parser.add_argument('--num_basis', type=int, default=5,
                        help='Number of basis vectors (rank K)')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Regularization parameter')
    parser.add_argument('--num_iterations', type=int, default=1000,
                        help='Training iterations for basis learning')
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='Learning rate for basis learning')

    # Other
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')

    args = parser.parse_args()
    run_phase1(args)
