import numpy as np
import pandas as pd
import os

def create_synthetic_dataset(n_samples=1000, random_seed=42):
    """
    Create synthetic dataset with the following structure:
    - Columns 0-9 (X0-X9): Gaussian numerical features
    - Columns 10-11 (X10-X11): 5-hot categorical features
    - Column 12 (A): Binary treatment (0/1)
    - Column 13 (Y1_0): Outcome when A=0 (continuous)
    - Column 14 (Y1_1): Outcome when A=1 (continuous)
    - Column 15 (Y2_0): Second outcome when A=0 (continuous)
    - Column 16 (Y2_1): Second outcome when A=1 (continuous)

    Parameters:
    -----------
    n_samples : int
        Total number of samples to generate (default: 1000)
    random_seed : int
        Random seed for reproducibility (default: 42)
    """
    np.random.seed(random_seed)

    # Create 10 Gaussian numerical features (X0-X9)
    X_numerical = np.random.randn(n_samples, 10)

    # Create binary treatment A (0 or 1)
    A = np.random.randint(0, 2, size=(n_samples, 1))

    # Create potential outcomes (continuous values)
    # Y0_0, Y0_1: First set of outcomes
    Y0_0 = np.random.randn(n_samples, 1) * 2 + 5  # Mean=5, std=2
    Y0_1 = np.random.randn(n_samples, 1) * 2 + 7  # Mean=7, std=2

    # Y1_0, Y1_1: Second set of outcomes
    Y1_0 = np.random.randn(n_samples, 1) * 1.5 + 3  # Mean=3, std=1.5
    Y1_1 = np.random.randn(n_samples, 1) * 1.5 + 4  # Mean=4, std=1.5

    # Combine all features into a single array
    data = np.hstack([
        X_numerical,      # X0-X9
        A,                # A
        Y0_0,             # Y0_0
        Y0_1,             # Y0_1
        Y1_0,             # Y1_0
        Y1_1              # Y1_1
    ])

    # Create column names
    columns = (
        [f'X{i}' for i in range(10)] +  # X0-X9
        ['A'] +                          # A
        ['Y0_0', 'Y0_1', 'Y1_0', 'Y1_1'] # Outcomes
    )

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # change A to categorical (int)
    df['A'] = df['A'].astype(np.int64)
    df['target'] = 0 # Placeholder target column

    # Create output directory if it doesn't exist
    output_dir = 'dataset/synthetic'
    os.makedirs(output_dir, exist_ok=True)

    # Save raw data (before preprocessing)
    df.to_csv(f'{output_dir}/data.csv', index=False)

    return df


if __name__ == "__main__":
    # Generate the dataset
    df = create_synthetic_dataset(n_samples=1000)

