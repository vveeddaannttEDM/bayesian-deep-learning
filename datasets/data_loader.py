import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TabularDataset(Dataset):
    def __init__(self, features, targets):
        self.X = torch.tensor(features.values, dtype=torch.float32)
        self.y = torch.tensor(targets.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(csv_path: str, target_column: str, test_size=0.2, batch_size=32):
    """
    Loads and preprocesses tabular data from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        target_column (str): Name of the target column.
        test_size (float): Fraction of data to use for testing.
        batch_size (int): Batch size for the dataloader.

    Returns:
        Tuple of DataLoaders: (train_loader, test_loader)
    """
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Normalize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42
    )

    train_dataset = TabularDataset(X_train, y_train)
    test_dataset = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    train_loader, test_loader = load_dataset(
        csv_path="datasets/sample_data.csv", target_column="target"
    )

    for batch in train_loader:
        inputs, targets = batch
        print("Input batch shape:", inputs.shape)
        print("Target batch shape:", targets.shape)
        break
