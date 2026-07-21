"""
CS 898BA - Homework Three, Part 5
Evaluate baseline vs optimized models on the held-out test set.

Produces:
  - classification_report.txt (accuracy, precision, recall, f1 per class,
    for both models)
  - confusion_matrix.png (optimized model, test set)
  - comparison_grid.png (loss/acc curves for both models + confusion matrix,
    side by side, for the README)
"""

import argparse
import os
import json

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from part2_data_pipeline import collect_dataset, load_split_csv, get_transforms, FishDataset
from part3_baseline_cnn import FishCNN, DEVICE


def get_predictions(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())
    return all_labels, all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/input/Fish")
    parser.add_argument("--split-csv", default="outputs/stage2_classification/dataset_split.csv")
    parser.add_argument("--baseline-weights", default="outputs/stage3_classification/baseline_model.pt")
    parser.add_argument("--optimized-weights", default="outputs/stage4_classification/optimized_model.pt")
    parser.add_argument("--optimized-config", default="outputs/stage4_classification/best_config.json")
    parser.add_argument("--baseline-history", default="outputs/stage3_classification/baseline_history.json")
    parser.add_argument("--optimized-history", default="outputs/stage4_classification/optimized_history.json")
    parser.add_argument("--out-dir", default="outputs/stage5_classification")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    _, _, species_names = collect_dataset(args.data_root)
    class_to_idx = {name: i for i, name in enumerate(species_names)}
    # explicit label order passed to sklearn everywhere below - this is the
    # exact bug the HW2 review flagged: don't let a library infer an order
    # and assume it lines up with what gets printed.
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    ordered_class_names = [idx_to_class[i] for i in range(len(species_names))]

    _, eval_tf = get_transforms()
    test_paths, test_labels = load_split_csv(args.split_csv, "test")
    test_ds = FishDataset(test_paths, test_labels, class_to_idx, eval_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    with open(args.optimized_config) as f:
        best_config = json.load(f)

    baseline_model = FishCNN(num_classes=len(species_names)).to(DEVICE)
    baseline_model.load_state_dict(torch.load(args.baseline_weights, map_location=DEVICE))

    optimized_model = FishCNN(num_classes=len(species_names),
                               dropout_rate=best_config["dropout_rate"]).to(DEVICE)
    optimized_model.load_state_dict(torch.load(args.optimized_weights, map_location=DEVICE))

    baseline_labels, baseline_preds = get_predictions(baseline_model, test_loader)
    optimized_labels, optimized_preds = get_predictions(optimized_model, test_loader)

    baseline_report = classification_report(
        baseline_labels, baseline_preds,
        labels=list(range(len(species_names))), target_names=ordered_class_names,
    )
    optimized_report = classification_report(
        optimized_labels, optimized_preds,
        labels=list(range(len(species_names))), target_names=ordered_class_names,
    )

    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write("=== Baseline model - test set ===\n")
        f.write(baseline_report)
        f.write("\n\n=== Optimized model - test set ===\n")
        f.write(optimized_report)
        f.write(f"\n\nWinning hyperparameter config: {best_config}\n")

    print(baseline_report)
    print(optimized_report)

    # quick sanity spot-check: print 5 individual predictions next to their
    # true labels before trusting the aggregate numbers above
    print("\nSpot-check - 5 optimized-model predictions vs ground truth:")
    for i in range(min(5, len(optimized_labels))):
        true_name = ordered_class_names[optimized_labels[i]]
        pred_name = ordered_class_names[optimized_preds[i]]
        flag = "" if true_name == pred_name else "  <-- MISS"
        print(f"  true={true_name:10s} pred={pred_name:10s}{flag}")

    cm = confusion_matrix(optimized_labels, optimized_preds, labels=list(range(len(species_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ordered_class_names)

    fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax_cm, xticks_rotation=45, cmap="Blues", colorbar=False)
    ax_cm.set_title("Optimized model - confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=120)
    plt.close(fig_cm)

    with open(args.baseline_history) as f:
        baseline_history = json.load(f)
    with open(args.optimized_history) as f:
        optimized_history = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(baseline_history["train_loss"], label="train")
    axes[0, 0].plot(baseline_history["val_loss"], label="val")
    axes[0, 0].set_title("Baseline - Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(baseline_history["train_acc"], label="train")
    axes[0, 1].plot(baseline_history["val_acc"], label="val")
    axes[0, 1].set_title("Baseline - Accuracy")
    axes[0, 1].legend()

    axes[1, 0].plot(optimized_history["train_loss"], label="train")
    axes[1, 0].plot(optimized_history["val_loss"], label="val")
    axes[1, 0].set_title("Optimized - Loss")
    axes[1, 0].legend()

    axes[1, 1].plot(optimized_history["train_acc"], label="train")
    axes[1, 1].plot(optimized_history["val_acc"], label="val")
    axes[1, 1].set_title("Optimized - Accuracy")
    axes[1, 1].legend()

    disp.plot(ax=axes[0, 2], xticks_rotation=45, cmap="Blues", colorbar=False)
    axes[0, 2].set_title("Optimized - Confusion Matrix")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "comparison_grid.png"), dpi=120)
    plt.close(fig)

    print(f"\nAll outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
