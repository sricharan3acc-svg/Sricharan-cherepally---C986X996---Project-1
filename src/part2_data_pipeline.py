"""
CS 898BA - Homework Three, Part 2
Data preprocessing and augmentation for the fish classification dataset.

This script does three things:
  1. Walks data/input/Fish/<species>/*.jpg and builds a stratified 70/15/15
     train/val/test split.
  2. Saves that split to a CSV so the same exact split gets reused by
     part3, part4, and part5 (splitting fresh in every script would risk
     silently training on what should be test data).
  3. Builds the torchvision transform pipelines - resize + normalize for
     all three splits, plus extra augmentation only on the training set.

Run this first. It doesn't train anything, it just prepares the split file.
"""

import argparse
import os
import csv
from collections import Counter

from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset

RANDOM_SEED = 42
IMG_SIZE = 128  # keeping this smaller than 224 on purpose - only ~1000
                # images total across 6 species, and a from-scratch CNN
                # at 224x224 would overfit even faster than it already does


def collect_dataset(root_dir):
    """Walk the species folders and return parallel lists of filepaths and labels."""
    filepaths, labels = [], []
    species_names = sorted(os.listdir(root_dir))
    for species in species_names:
        species_dir = os.path.join(root_dir, species)
        if not os.path.isdir(species_dir):
            continue
        for fname in sorted(os.listdir(species_dir)):
            filepaths.append(os.path.join(species_dir, fname))
            labels.append(species)
    return filepaths, labels, species_names


def make_stratified_split(filepaths, labels, train_frac=0.70, val_frac=0.15):
    """
    70/15/15 stratified split, done as two sequential splits since
    train_test_split only cuts one way at a time.
    """
    test_frac = 1.0 - train_frac - val_frac

    train_paths, rest_paths, train_labels, rest_labels = train_test_split(
        filepaths, labels,
        test_size=(1.0 - train_frac),
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    # rest_paths now holds val+test together - split it down the middle
    val_share_of_rest = val_frac / (val_frac + test_frac)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        rest_paths, rest_labels,
        test_size=(1.0 - val_share_of_rest),
        stratify=rest_labels,
        random_state=RANDOM_SEED,
    )

    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


def save_split_csv(split_dict, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label", "split"])
        for split_name, (paths, labels) in split_dict.items():
            for p, lbl in zip(paths, labels):
                writer.writerow([p, lbl, split_name])


def load_split_csv(csv_path, split_name):
    """Read the saved split file back and filter down to one split."""
    paths, labels = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split_name:
                paths.append(row["filepath"])
                labels.append(row["label"])
    return paths, labels


class FishDataset(Dataset):
    """
    Thin wrapper around a list of filepaths + string labels.

    class_to_idx is built once (alphabetically, matching os.listdir's sort
    order from collect_dataset) and re-used everywhere downstream, so the
    integer a model predicts always maps back to the same species name in
    part4 and part5. Getting this mapping inconsistent between scripts was
    exactly the kind of silent bug the HW2 review was warning about.
    """

    def __init__(self, filepaths, labels, class_to_idx, transform):
        self.filepaths = filepaths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = Image.open(self.filepaths[idx]).convert("RGB")
        img = self.transform(img)
        label_idx = self.class_to_idx[self.labels[idx]]
        return img, label_idx


def get_transforms():
    """
    Train transform gets augmentation (flip / rotation / brightness jitter).
    Val and test transforms only resize + normalize - augmenting validation
    or test data would just be measuring performance on a distorted problem,
    not the real one.
    """
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tf, eval_tf


def sanity_check_augmentation(split_dict, train_tf, out_path):
    """
    Save a grid comparing a few raw training images against their augmented
    versions. This is here specifically because of the HW2 review comment -
    negated masks looked plausible until actually eyeballed against the
    source image. Same principle here: confirm the augmented fish still
    LOOKS like the same fish and the species label wasn't scrambled by a
    rotation/crop before trusting this pipeline feeds the CNN correctly.
    """
    train_paths, train_labels = split_dict["train"]
    sample_idx = list(range(0, len(train_paths), max(1, len(train_paths) // 4)))[:4]

    fig, axes = plt.subplots(2, len(sample_idx), figsize=(4 * len(sample_idx), 8))
    for col, idx in enumerate(sample_idx):
        raw_img = Image.open(train_paths[idx]).convert("RGB")
        aug_tensor = train_tf(raw_img)
        # undo the normalize just for display purposes
        aug_img = aug_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5

        axes[0, col].imshow(raw_img)
        axes[0, col].set_title(f"raw - {train_labels[idx]}")
        axes[0, col].axis("off")

        axes[1, col].imshow(aug_img)
        axes[1, col].set_title(f"augmented - {train_labels[idx]}")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Sanity-check grid saved to {out_path} - confirm labels still match visually.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/input/Fish")
    parser.add_argument("--out-dir", default="outputs/stage2_classification")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    filepaths, labels, species_names = collect_dataset(args.data_root)
    print(f"Found {len(filepaths)} images across {len(species_names)} species: {species_names}")
    print("Per-species counts:", Counter(labels))

    split_dict = make_stratified_split(filepaths, labels)
    for split_name, (paths, lbls) in split_dict.items():
        print(f"{split_name}: {len(paths)} images - {Counter(lbls)}")

    split_csv_path = os.path.join(args.out_dir, "dataset_split.csv")
    save_split_csv(split_dict, split_csv_path)
    print(f"Split saved to {split_csv_path}")

    train_tf, eval_tf = get_transforms()
    sanity_check_augmentation(split_dict, train_tf, os.path.join(args.out_dir, "augmentation_sanity_check.png"))


if __name__ == "__main__":
    main()
