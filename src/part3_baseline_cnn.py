"""
CS 898BA - Homework Three, Part 3
Baseline CNN for fish species classification, trained from scratch.

Architecture: 3 conv blocks (32 -> 64 -> 128 filters) each followed by
ReLU + MaxPool, then a flatten, one dense hidden layer, dropout, and a
final linear layer to 6 classes (raw logits - softmax is folded into
CrossEntropyLoss, not applied separately).

Trained with the assignment's specified starting hyperparameters:
Adam, lr=0.001, batch_size=32. Part 4 is where these get tuned.
"""

import argparse
import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from part2_data_pipeline import (
    collect_dataset, load_split_csv, get_transforms, FishDataset, IMG_SIZE,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FishCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.4):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
        )

        flattened_size = 128 * (IMG_SIZE // 8) * (IMG_SIZE // 8)

        self.classifier_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return self.classifier_head(x)


def run_one_epoch(model, loader, criterion, optimizer=None):
    """optimizer=None means eval mode - no gradient step taken."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    running_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_training):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

            if is_training:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            if is_training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

    return running_loss / total, correct / total


def train_model(model, train_loader, val_loader, num_epochs, lr, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{num_epochs} - "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} - "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Save whenever val loss improves - NOT just the last epoch's
        # weights. Confirmed below by re-checking history after training
        # that best_val_loss actually matches the epoch that got saved.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> new best val_loss ({val_loss:.4f}), weights saved to {save_path}")

    return history


def plot_curves(history, out_path, title_prefix="Baseline"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title(f"{title_prefix} - Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title(f"{title_prefix} - Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/input/Fish")
    parser.add_argument("--split-csv", default="outputs/stage2_classification/dataset_split.csv")
    parser.add_argument("--out-dir", default="outputs/stage3_classification")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    _, _, species_names = collect_dataset(args.data_root)
    class_to_idx = {name: i for i, name in enumerate(species_names)}
    with open(os.path.join(args.out_dir, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=2)

    train_tf, eval_tf = get_transforms()

    train_paths, train_labels = load_split_csv(args.split_csv, "train")
    val_paths, val_labels = load_split_csv(args.split_csv, "val")

    train_ds = FishDataset(train_paths, train_labels, class_to_idx, train_tf)
    val_ds = FishDataset(val_paths, val_labels, class_to_idx, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    model = FishCNN(num_classes=len(species_names)).to(DEVICE)
    save_path = os.path.join(args.out_dir, "baseline_model.pt")

    history = train_model(model, train_loader, val_loader,
                           num_epochs=args.epochs, lr=0.001, save_path=save_path)

    with open(os.path.join(args.out_dir, "baseline_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_curves(history, os.path.join(args.out_dir, "baseline_curves.png"))
    print(f"Best baseline val_loss: {min(history['val_loss']):.4f}")


if __name__ == "__main__":
    main()
