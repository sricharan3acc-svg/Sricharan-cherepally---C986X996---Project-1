"""
CS 898BA - Homework Three, Part 4 (resume)

Use this if part4_hyperparameter_tuning.py's grid search finished
(grid_search_results.json exists) but the final retrain of the winning
config got interrupted before finishing - e.g. system shutdown mid-run.

This skips straight to picking the best config from the saved grid search
results and retrains it from scratch, so there's no partial/corrupted
state left over from the interrupted run.
"""

import argparse
import os
import json

from torch.utils.data import DataLoader

from part2_data_pipeline import collect_dataset, load_split_csv, get_transforms, FishDataset
from part3_baseline_cnn import FishCNN, train_model, plot_curves, DEVICE
from part4_hyperparameter_tuning import FINAL_EPOCHS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/input/Fish")
    parser.add_argument("--split-csv", default="outputs/stage2_classification/dataset_split.csv")
    parser.add_argument("--out-dir", default="outputs/stage4_classification")
    args = parser.parse_args()

    results_path = os.path.join(args.out_dir, "grid_search_results.json")
    with open(results_path) as f:
        results_sorted = json.load(f)

    best_config = results_sorted[0]
    print(f"Resuming from saved grid search - winning config: {best_config}")

    _, _, species_names = collect_dataset(args.data_root)
    class_to_idx = {name: i for i, name in enumerate(species_names)}

    train_tf, eval_tf = get_transforms()
    train_paths, train_labels = load_split_csv(args.split_csv, "train")
    val_paths, val_labels = load_split_csv(args.split_csv, "val")

    train_ds = FishDataset(train_paths, train_labels, class_to_idx, train_tf)
    val_ds = FishDataset(val_paths, val_labels, class_to_idx, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=best_config["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=best_config["batch_size"], shuffle=False, num_workers=2)

    final_model = FishCNN(num_classes=len(species_names), dropout_rate=best_config["dropout_rate"]).to(DEVICE)
    final_save_path = os.path.join(args.out_dir, "optimized_model.pt")

    # this overwrites the partial optimized_model.pt from the interrupted
    # run - that's intentional, we want a clean full run, not whatever
    # epoch happened to be mid-save when the laptop shut down
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
