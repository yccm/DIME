"""
Example usage of DIME for conditional outcome generation.

Scenario:
- 10 continuous input features (X0-X9)
- 1 categorical input feature (A with 2 categories)
- 2 continuous outcome features (Y0, Y1)

Goal: Model P(outcome1, outcome2 | continuous_features, categorical_feature)
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from models.model import DIME
from utils_train import preprocess, TabularDataset


def load_synthetic_data():
    """Load the synthetic dataset."""
    dataname = 'synthetic'

    # Load dataset info
    info_path = f'dataset/{dataname}/info_new.json'
    info = json.load(open(info_path, 'r'))

    outcome_col_idx = info['outcome_col_idx']
    num_col_idx = info['num_col_idx']

    # Preprocess data
    data_dir = f'dataset/{dataname}/'
    X_num, X_cat, categories, _, _, _ = preprocess(
        data_dir,
        task_type=info['task_type'],
        inverse=True,
        concat=False
    )

    # Calculate number of numerical features (excluding outcomes)
    n_num = len(num_col_idx) - len(outcome_col_idx)
    num_outcomes = len(outcome_col_idx)

    # Split train and test
    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    # Separate input features from outcomes
    X_train_num, X_train_outcome = X_train_num[:, :n_num], X_train_num[:, n_num:]
    X_test_num, X_test_outcome = X_test_num[:, :n_num], X_test_num[:, n_num:]

    # Convert to tensors
    X_train_num = torch.tensor(X_train_num).float()
    X_train_cat = torch.tensor(X_train_cat).long()
    X_train_outcome = torch.tensor(X_train_outcome).float()
    X_test_num = torch.tensor(X_test_num).float()
    X_test_cat = torch.tensor(X_test_cat).long()
    X_test_outcome = torch.tensor(X_test_outcome).float()

    # Normalize numerical features
    mean_num, std_num = X_train_num.mean(0), X_train_num.std(0)
    X_train_num = (X_train_num - mean_num) / std_num / 2
    X_test_num = (X_test_num - mean_num) / std_num / 2

    # Normalize outcomes
    mean_outcome, std_outcome = X_train_outcome.mean(0), X_train_outcome.std(0)
    X_train_outcome = (X_train_outcome - mean_outcome) / std_outcome / 2
    X_test_outcome = (X_test_outcome - mean_outcome) / std_outcome / 2

    print(f"Loaded synthetic dataset:")
    print(f"  Training samples: {X_train_num.shape[0]}")
    print(f"  Test samples: {X_test_num.shape[0]}")
    print(f"  Numerical features: {n_num}")
    print(f"  Categorical features: {len(categories)}, categories: {categories}")
    print(f"  Outcome features: {num_outcomes}")

    return (X_train_num, X_train_cat, X_train_outcome,
            X_test_num, X_test_cat, X_test_outcome,
            n_num, categories, num_outcomes)


def example_training():
    """Example training loop for DIME."""

    # Load synthetic data
    (X_train_num, X_train_cat, X_train_outcome,
     X_test_num, X_test_cat, X_test_outcome,
     n_num, categories, n_outcome) = load_synthetic_data()

    # Model configuration
    embed_dim = 128
    buffer_size = 8
    depth = 3
    dropout_rate = 0.0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Initialize model
    model = DIME(
        n_num=n_num,
        categories=categories,
        n_outcome=n_outcome,
        embed_dim=embed_dim,
        buffer_size=buffer_size,
        depth=depth,
        norm_layer=nn.LayerNorm,
        dropout_rate=dropout_rate,
        device=device
    ).to(device)

    # Create dataset and dataloader
    train_dataset = TabularDataset(X_train_num, X_train_cat, X_train_outcome)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    # Training mode
    model.train()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training step (single batch for demo)
    x_num, x_cat, x_outcome = next(iter(train_loader))
    x_num = x_num.to(device)
    x_cat = x_cat.to(device)
    x_outcome = x_outcome.to(device)

    optimizer.zero_grad()
    loss, loss_per_outcome = model(
        x_num=x_num,
        x_cat=x_cat,
        x_outcome=x_outcome
    )

    loss.backward()
    optimizer.step()

    print(f"\nTraining step:")
    print(f"  Batch size: {x_num.shape[0]}")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Loss per outcome: {loss_per_outcome.detach().cpu().numpy()}")

    return model, (X_test_num, X_test_cat, X_test_outcome)


def example_sampling(model=None, test_data=None):
    """Example sampling from DIME."""

    if model is None or test_data is None:
        # Train model and get test data if not provided
        model, test_data = example_training()

    X_test_num, X_test_cat, X_test_outcome = test_data

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.eval()

    # Use first 10 test samples
    batch_size = min(10, X_test_num.shape[0])
    x_num = X_test_num[:batch_size].to(device)
    x_cat = X_test_cat[:batch_size].to(device)
    x_outcome_gt = X_test_outcome[:batch_size].to(device)  # Ground truth for comparison

    print("\n" + "="*60)
    print("Sampling Examples")
    print("="*60)
    print(f"Using {batch_size} test samples")
    print(f"Ground truth outcomes (first 3):\n{x_outcome_gt[:3].cpu().numpy()}")

    print("\nIterative autoregressive sampling:")
    sampled_outcomes = model.sample_iterative(
        x_num=x_num,
        x_cat=x_cat,
        num_steps=50,
        device=device
    )
    print(f"Sampled outcomes shape: {sampled_outcomes.shape}")
    print(f"First 3 samples:\n{sampled_outcomes[:3].cpu().numpy()}")

    return sampled_outcomes

if __name__ == "__main__":
    # Run training example
    print("="*60)
    print("Training Example")
    print("="*60)
    model, test_data = example_training()

    # Run sampling examples
    example_sampling(model, test_data)