from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple regressor on AstroCLIP embeddings for redshift.")
    parser.add_argument("--embeddings", type=Path, required=True, help="HDF5 file produced by embed_dataset.py.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--model", choices=["image", "spectrum", "joint"], default="joint")
    parser.add_argument("--output", type=Path, default=Path("outputs/redshift_regressor.pt"))
    return parser.parse_args()


def load_embeddings(path: Path, modality: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        redshift = f["redshift"][()]
        if modality == "image":
            features = f["image_embeddings"][()]
        elif modality == "spectrum":
            features = f["spectrum_embeddings"][()]
        else:
            features = np.concatenate(
                [f["image_embeddings"][()], f["spectrum_embeddings"][()]], axis=1
            )
    return features, redshift


def run_training(features: np.ndarray, targets: np.ndarray, args: argparse.Namespace) -> None:
    n_samples = features.shape[0]
    split = int(0.8 * n_samples)
    train_X, val_X = features[:split], features[split:]
    train_y, val_y = targets[:split], targets[split:]

    train_dataset = TensorDataset(torch.from_numpy(train_X).float(), torch.from_numpy(train_y).float())
    val_dataset = TensorDataset(torch.from_numpy(val_X).float(), torch.from_numpy(val_y).float())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = MLP(train_X.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * batch_x.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "input_dim": train_X.shape[1]}, args.output)


def main() -> None:
    args = parse_args()
    features, targets = load_embeddings(args.embeddings, args.model)
    run_training(features, targets, args)


if __name__ == "__main__":
    main()
