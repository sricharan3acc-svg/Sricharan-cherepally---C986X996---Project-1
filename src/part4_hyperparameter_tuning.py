"""
CS 898BA - Homework Three, Part 4
Grid search hyperparameter tuning for the fish CNN.

Tuning 3 hyperparameters:
  - learning rate: 0.01, 0.001, 0.0001
  - batch size:    32, 64
  - dropout rate:  0.3, 0.5

That's 3 x 2 x 2 = 12 configurations, which is small enough to just run
an exhaustive grid search rather than needing Optuna/random sampling -
random search earns its keep on larger search spaces, not a 12-run grid.

Each config trains for a reduced epoch count relative to the baseline
(grid search over 12 configs x full epoch budget would take too long),
then the config with the lowest validation loss gets retrained for the
full epoch budget and saved as the "optimized" model.
"""

import argparse
import os
import json
import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from part2_data_pipeline import collect_dataset, load_split_csv, get_transforms, FishDataset
from part3_baseline_cnn import FishCNN, train_model, plot_curves, DEVICE

LEARNING_RATES = [0.01, 0.001, 0.0001]
BATCH_SIZES = [32, 64]
DROPOUT_RATES = [0.3, 0.5]
SEARCH_EPOCHS = 10   # short budget per config, just to rank them
FINAL_EPOCHS = 25    # full budget for retraining the winning config


def run_grid_search(species_names, class_to_idx, train_tf, eval_tf,
                     train_paths, train_labels, val_paths, val_labels, out_dir):
    results = []

    grid = list(itertools.product(LEARNING_RATES, BATCH_SIZES, DROPOUT_RATES))
    print(f"Running grid search over {len(grid)} configurations...")

    for run_idx, (lr, batch_size, dropout_rate) in enumerate(grid, start=1):
        print(f"\n--- Config {run_idx}/{len(grid)}: lr={lr}, batch_size={batch_size}, dropout={dropout_rate} ---")

        train_ds = FishDataset(train_paths, train_labels, class_to_idx, train_tf)
        val_ds = FishDataset(val_paths, val_labels, class_to_idx, eval_tf)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

        model = FishCNN(num_classes=len(species_names), dropout_rate=dropout_rate).to(DEVICE)
        temp_save_path = os.path.join(out_dir, f"_temp_config{run_idx}.pt")

        history = train_model(model, train_loader, val_loader,
                               num_epochs=SEARCH_EPOCHS, lr=lr, save_path=temp_save_path)

        best_val_loss = min(history["val_loss"])
        best_val_acc = history["val_acc"][history["val_loss"].index(best_val_loss)]

        results.append({
            "run": run_idx,
            "learning_rate": lr,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
        })

        # don't keep 12 sets of temp weights around, only the winning
        # config gets retrained and saved for real further down
        os.remove(temp_save_path)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/input/Fish")
    parser.add_argument("--split-csv", default="outputs/stage2_classification/dataset_split.csv")
    parser.add_argument("--out-dir", default="outputs/stage4_classification")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    _, _, species_names = collect_dataset(args.data_root)
    class_to_idx = {name: i for i, name in enumerate(species_names)}

    train_tf, eval_tf = get_transforms()
    train_paths, train_labels = load_split_csv(args.split_csv, "train")
    val_paths, val_labels = load_split_csv(args.split_csv, "val")

    results = run_grid_search(species_names, class_to_idx, train_tf, eval_tf,
                               train_paths, train_labels, val_paths, val_labels, args.out_dir)

    results_sorted = sorted(results, key=lambda r: r["best_val_loss"])
    with open(os.path.join(args.out_dir, "grid_search_results.json"), "w") as f:
        json.dump(results_sorted, f, indent=2)

    print("\n=== Grid search results, best to worst by val_loss ===")
    for r in results_sorted:
        print(r)

    # Sanity check before committing to a "winner" - print the gap between
    # best and second-best so a near-tie doesn't get over-interpreted as a
    # meaningful hyperparameter effect in the README writeup.
    best, runner_up = results_sorted[0], results_sorted[1]
    gap = runner_up["best_val_loss"] - best["best_val_loss"]
    print(f"\nBest vs runner-up val_loss gap: {gap:.4f} "
          f"({'meaningful' if gap > 0.02 else 'small - treat as close to a tie'})")

    best_config = results_sorted[0]
    print(f"\nRetraining winning config for {FINAL_EPOCHS} epochs: {best_config}")

    train_ds = FishDataset(train_paths, train_labels, class_to_idx, train_tf)
    val_ds = FishDataset(val_paths, val_labels, class_to_idx, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=best_config["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=best_config["batch_size"], shuffle=False, num_workers=2)

    final_model = FishCNN(num_classes=len(species_names), dropout_rate=best_config["dropout_rate"]).to(DEVICE)
    final_save_path = os.path.join(args.out_dir, "optimized_model.pt")

    final_history = train_model(final_model, train_loader, val_loader,
                                 num_epochs=FINAL_EPOCHS, lr=best_config["learning_rate"],
                                 save_path=final_save_path)

    with open(os.path.join(args.out_dir, "optimized_history.json"), "w") as f:
        json.dump(final_history, f, indent=2)
    with open(os.path.join(args.out_dir, "best_config.json"), "w") as f:
        json.dump(best_config, f, indent=2)

    plot_curves(final_history, os.path.join(args.out_dir, "optimized_curves.png"), title_prefix="Optimized")
    print(f"Best optimized val_loss: {min(final_history['val_loss']):.4f}")


if __name__ == "__main__":
    main()
