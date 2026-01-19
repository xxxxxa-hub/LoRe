# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main Entry Point for Context Selection Experiment

This script runs both Phase 1 and Phase 2, or allows running them separately.

Data Flow (following vary_fewshot.py):
- Seen Users: train_features (from train pkl) → train V, W
- Unseen Users:
  - train_features_unseen (from train pkl) → D_pool (split into Demo/Val)
  - test_features_sparse_unseen (from val pkl) → D_test (held-out)

Usage:
    # Run full experiment (both phases)
    python run_experiment.py --run_all

    # Run Phase 1 only (initialization)
    python run_experiment.py --phase1_only

    # Run Phase 2 only (requires Phase 1 checkpoints)
    python run_experiment.py --phase2_only

    # Quick test with reduced parameters
    python run_experiment.py --run_all --quick_test
"""

import argparse
import os
import sys
from datetime import datetime

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from experiment_utils import save_results, set_all_seeds


def run_phase1(args):
    """Run Phase 1: Initialization."""
    from phase1_initialization import run_phase1 as phase1_main

    class Phase1Args:
        pass

    p1_args = Phase1Args()
    p1_args.train_embeddings = args.train_embeddings
    p1_args.val_embeddings = args.val_embeddings
    p1_args.checkpoint_dir = args.checkpoint_dir
    p1_args.T_seen = args.T_seen
    p1_args.T_unseen = args.T_unseen
    p1_args.num_basis = args.num_basis
    p1_args.alpha = args.alpha
    p1_args.num_iterations = args.num_iterations
    p1_args.learning_rate = args.learning_rate
    p1_args.seed = args.seed
    p1_args.device = args.device

    return phase1_main(p1_args)


def run_phase2(args):
    """Run Phase 2: Experiment."""
    from phase2_experiment import run_phase2 as phase2_main

    class Phase2Args:
        pass

    p2_args = Phase2Args()
    p2_args.checkpoint_dir = args.checkpoint_dir
    p2_args.results_dir = args.results_dir
    p2_args.n_trials = args.n_trials
    p2_args.bootstrap_seeds = args.bootstrap_seeds
    p2_args.shot_counts = args.shot_counts
    p2_args.split_configs = args.split_configs
    p2_args.personalize_iterations = args.personalize_iterations
    p2_args.personalize_lr = args.personalize_lr
    p2_args.device = args.device
    p2_args.verbose = args.verbose

    return phase2_main(p2_args)


def main():
    parser = argparse.ArgumentParser(
        description="Context Selection Experiment for LoRe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data Flow (following vary_fewshot.py):
  - Seen Users: train_features (train pkl) → train V, W
  - Unseen Users:
    - train_features_unseen (train pkl) → D_pool (Demo/Val)
    - test_features_sparse_unseen (val pkl) → D_test

Examples:
    # Run full experiment
    python run_experiment.py --run_all

    # Quick test run
    python run_experiment.py --run_all --quick_test

    # Run Phase 1 only
    python run_experiment.py --phase1_only

    # Run Phase 2 with custom config
    python run_experiment.py --phase2_only --n_trials 50 --shot_counts 2,4
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--run_all', action='store_true',
                           help='Run both Phase 1 and Phase 2')
    mode_group.add_argument('--phase1_only', action='store_true',
                           help='Run Phase 1 (initialization) only')
    mode_group.add_argument('--phase2_only', action='store_true',
                           help='Run Phase 2 (experiment) only')

    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                       help='Run with reduced parameters for testing')

    # Data paths
    parser.add_argument('--train_embeddings', type=str,
                       default='tldr_embeddings_train.pkl',
                       help='Path to training embeddings pickle')
    parser.add_argument('--val_embeddings', type=str,
                       default='tldr_embeddings_val.pkl',
                       help='Path to validation embeddings pickle')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='checkpoints/',
                       help='Directory for checkpoints')
    parser.add_argument('--results_dir', type=str,
                       default='results/',
                       help='Directory for results')

    # Phase 1: Data sampling parameters (following vary_fewshot.py)
    parser.add_argument('--T_seen', type=int, default=150,
                       help='Max samples per seen user for training V')
    parser.add_argument('--T_unseen', type=int, default=100,
                       help='Max samples per unseen user for D_pool')

    # Phase 1: LoRe parameters
    parser.add_argument('--num_basis', type=int, default=5,
                       help='Number of basis vectors (rank K)')
    parser.add_argument('--alpha', type=float, default=0,
                       help='Regularization parameter')
    parser.add_argument('--num_iterations', type=int, default=1000,
                       help='Training iterations for basis learning')
    parser.add_argument('--learning_rate', type=float, default=0.5,
                       help='Learning rate for basis learning')

    # Phase 2: Experiment parameters
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Search budget (number of contexts to try)')
    parser.add_argument('--bootstrap_seeds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--shot_counts', type=str, default='2,4,8',
                       help='Comma-separated list of shot counts')
    parser.add_argument('--split_configs', type=str,
                       default='0.2/0.8,0.4/0.6,0.6/0.4,0.8/0.2',
                       help='Comma-separated Demo/Val ratio splits (e.g., 0.2/0.8 means 20%% demo, 80%% val)')

    # Phase 2: Weight learning parameters
    parser.add_argument('--personalize_iterations', type=int, default=500,
                       help='Iterations for weight learning')
    parser.add_argument('--personalize_lr', type=float, default=0.1,
                       help='Learning rate for weight learning')

    # Other
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')

    args = parser.parse_args()

    # Apply quick test settings if requested
    if args.quick_test:
        print("\n" + "="*60)
        print("QUICK TEST MODE: Using reduced parameters")
        print("="*60)
        args.T_unseen = 20  # Smaller pool per user
        args.num_basis = 3
        args.num_iterations = 100
        args.n_trials = 10
        args.bootstrap_seeds = 2
        args.shot_counts = '2,4'
        args.split_configs = '0.25/0.75,0.5/0.5'
        args.personalize_iterations = 100
        args.verbose = True

    # Set all random seeds for reproducibility
    set_all_seeds(args.seed)

    # Print configuration
    print("\n" + "="*60)
    print("Context Selection Experiment")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {args.device}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Results dir: {args.results_dir}")

    # Run phases
    if args.run_all or args.phase1_only:
        print("\n" + "#"*60)
        print("# PHASE 1: INITIALIZATION")
        print("#"*60)
        V, W, data = run_phase1(args)

    if args.run_all or args.phase2_only:
        print("\n" + "#"*60)
        print("# PHASE 2: EXPERIMENT")
        print("#"*60)
        results_log, summary = run_phase2(args)

        # Save final summary
        final_summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'T_seen': args.T_seen,
                'T_unseen': args.T_unseen,
                'n_trials': args.n_trials,
                'bootstrap_seeds': args.bootstrap_seeds,
                'shot_counts': args.shot_counts,
                'split_configs': args.split_configs,
            },
            'summary': summary,
        }
        summary_path = os.path.join(args.results_dir, 'experiment_final_summary.json')
        save_results(final_summary, summary_path)
        print(f"\nFinal summary saved to: {summary_path}")

    print("\n" + "="*60)
    print(f"Experiment Complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


if __name__ == "__main__":
    main()
