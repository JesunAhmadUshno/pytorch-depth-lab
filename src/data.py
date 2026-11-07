"""EuroSAT data: frozen stratified split and loader factory.

The split (seed 20261107, 21,600 / 2,700 / 2,700) is regenerated
deterministically from the class folders, so both machines build
the identical split from the identical zip. Split indices are
saved to results/split.json (derived statistic, no image data).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .common import IMAGENET_MEAN, IMAGENET_STD, RECIPE, REPO_ROOT, RESULTS

DATA_DIR = REPO_ROOT / "data"
IMAGE_DIR = DATA_DIR / "EuroSAT_RGB"
ZIP_PATH = DATA_DIR / "EuroSAT_RGB.zip"
SPLIT_SEED = RECIPE["split_seed"]
FRACTIONS = (0.8, 0.1, 0.1)  # train, val, test


def class_files(image_dir: Path = IMAGE_DIR) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for cls_dir in sorted(p for p in image_dir.iterdir() if p.is_dir()):
        out[cls_dir.name] = sorted(f.name for f in cls_dir.glob("*.jpg"))
    return out


def build_split(image_dir: Path = IMAGE_DIR) -> dict:
    """Stratified split as lists of global indices into the sorted
    (class, file) enumeration used by ImageFolder."""
    files = class_files(image_dir)
    classes = sorted(files)
    # Global index space: ImageFolder sorts classes, then files.
    index_of: dict[tuple[str, str], int] = {}
    k = 0
    for cls in classes:
        for name in files[cls]:
            index_of[(cls, name)] = k
            k += 1
    rng = random.Random(SPLIT_SEED)
    split: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    per_class: dict[str, dict[str, int]] = {}
    for cls in classes:
        idx = [index_of[(cls, name)] for name in files[cls]]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * FRACTIONS[0])
        n_val = int(n * FRACTIONS[1])
        parts = {
            "train": idx[:n_train],
            "val": idx[n_train : n_train + n_val],
            "test": idx[n_train + n_val :],
        }
        per_class[cls] = {k2: len(v) for k2, v in parts.items()}
        for k2, v in parts.items():
            split[k2].extend(v)
    for v in split.values():
        v.sort()
    return {"seed": SPLIT_SEED, "classes": classes, "per_class": per_class, "split": split}


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


def make_datasets(image_dir: Path = IMAGE_DIR):
    """Returns (train_ds, val_ds, test_ds) honouring the frozen split."""
    split_path = RESULTS / "split.json"
    if split_path.exists():
        saved = json.loads(split_path.read_text())
        split = saved["split"]
    else:
        split = build_split(image_dir)["split"]
    train_base = ImageFolder(image_dir, transform=train_transform())
    eval_base = ImageFolder(image_dir, transform=eval_transform())
    return (
        Subset(train_base, split["train"]),
        Subset(eval_base, split["val"]),
        Subset(eval_base, split["test"]),
    )


def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int = 0,
    pin: bool = False,
    persistent: bool = False,
    seed: int | None = None,
    sampler=None,
) -> DataLoader:
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=persistent and workers > 0,
        generator=gen,
    )


def verify() -> dict:
    """Recompute dataset facts and write results/dataset.json and
    results/split.json. Run on each machine; outputs must agree."""
    sha = hashlib.sha256()
    with open(ZIP_PATH, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    files = class_files()
    counts = {cls: len(v) for cls, v in files.items()}
    split = build_split()
    payload = {
        "dataset": "EuroSAT RGB",
        "zenodo_record": 7711810,
        "version": "v2",
        "zip_md5_record": "f46e308c4d50d4bf32fedad2d3b",
        "zip_sha256": sha.hexdigest(),
        "total_images": sum(counts.values()),
        "per_class_counts": counts,
        "image_size_checked": list(Image.open(IMAGE_DIR / split["classes"][0] / files[split["classes"][0]][0]).size),
        "split_sizes": {k: len(v) for k, v in split["split"].items()},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dataset.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (RESULTS / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        print(json.dumps(verify(), indent=2))
