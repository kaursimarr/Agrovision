"""Framework-free dataset helpers: discovery, stratified splitting, class weights.

Kept separate from ``data_loader`` (which needs TensorFlow) so they can be
unit-tested and reused without a deep-learning install.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .config import IMAGE_EXTENSIONS


def find_dataset_root(data_dir: str | Path) -> Path:
    """Locate the folder that directly contains the class sub-folders.

    PlantVillage downloads come nested in different ways, e.g.
    ``PlantVillage/PlantVillage/<classes>`` or ``plantvillage dataset/color/<classes>``.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}\n"
            "See README.md ('Get the dataset') for download instructions."
        )
    for _ in range(4):
        if (root / "color").is_dir():          # abdallahalidev/plantvillage-dataset layout
            root = root / "color"
            continue
        subdirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
        has_images = any(f.suffix.lower() in IMAGE_EXTENSIONS for f in root.iterdir() if f.is_file())
        if not has_images and len(subdirs) == 1:  # single wrapper folder -> descend
            root = subdirs[0]
            continue
        break
    return root


def list_images(root: str | Path) -> tuple[list[str], list[int], list[str]]:
    """Return (relative image paths, integer labels, sorted class names)."""
    root = Path(root)
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    paths: list[str] = []
    labels: list[int] = []
    class_names: list[str] = []
    for d in class_dirs:
        files = sorted(f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            continue
        idx = len(class_names)
        class_names.append(d.name)
        for f in files:
            paths.append(f.relative_to(root).as_posix())
            labels.append(idx)
    if len(class_names) < 2:
        raise ValueError(f"Found {len(class_names)} class folder(s) with images under {root}; need at least 2.")
    return paths, labels, class_names


def limit_per_class(paths: list[str], labels: list[int], n: int, seed: int) -> tuple[list[str], list[int]]:
    """Randomly keep at most ``n`` images per class (for quick CPU runs)."""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        by_class[y].append(i)
    keep: list[int] = []
    for y in sorted(by_class):
        idxs = by_class[y]
        keep.extend(sorted(rng.sample(idxs, n)) if len(idxs) > n else idxs)
    keep.sort()
    return [paths[i] for i in keep], [labels[i] for i in keep]


def stratified_split(paths, labels, fractions, seed) -> dict:
    """Stratified train/val/test split that keeps class proportions."""
    train_f, val_f, test_f = fractions
    if abs(train_f + val_f + test_f - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to 1.")
    idx = np.arange(len(paths))
    y = np.asarray(labels)
    idx_train, idx_rest = train_test_split(idx, train_size=train_f, stratify=y, random_state=seed)
    idx_val, idx_test = train_test_split(
        idx_rest, test_size=test_f / (val_f + test_f), stratify=y[idx_rest], random_state=seed
    )
    out = {}
    for name, ids in (("train", idx_train), ("val", idx_val), ("test", idx_test)):
        ids = sorted(int(i) for i in ids)
        out[name] = {"paths": [paths[i] for i in ids], "labels": [int(labels[i]) for i in ids]}
    return out


def get_or_create_split(root, split_path, fractions, seed, max_per_class=None, force=False) -> dict:
    """Load the saved split if it matches the current settings, else create + save it.

    Persisting the split guarantees every model is trained and tested on exactly the
    same images, which is essential for a fair CNN vs transfer-learning comparison.
    """
    root = Path(root)
    split_path = Path(split_path)
    meta = {"root": str(root.resolve()), "seed": seed, "max_per_class": max_per_class,
            "fractions": list(fractions)}
    if split_path.exists() and not force:
        info = json.loads(split_path.read_text())
        if info.get("meta") == meta:
            return info
        print("[data] Existing split.json used different settings -> regenerating it.\n"
              "       Models trained on the old split should be retrained before evaluation.")

    paths, labels, class_names = list_images(root)
    if max_per_class:
        paths, labels = limit_per_class(paths, labels, max_per_class, seed)
    splits = stratified_split(paths, labels, fractions, seed)
    info = {"meta": meta, "class_names": class_names, **splits}
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(info))
    return info


def compute_class_weights(labels, num_classes: int) -> dict[int, float]:
    """'Balanced' class weights (PlantVillage is imbalanced, e.g. 152 vs 5507 images/class)."""
    classes = np.arange(num_classes)
    present = np.unique(labels)
    weights = compute_class_weight("balanced", classes=present, y=np.asarray(labels))
    out = {int(c): 1.0 for c in classes}
    out.update({int(c): float(w) for c, w in zip(present, weights)})
    return out
