"""tf.data input pipeline: decode -> resize -> batch -> prefetch.

Images stay in the raw 0-255 range on purpose: each model contains its own
normalisation layer (see models.py), so the saved models accept raw images and the
Streamlit app needs no model-specific preprocessing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import (CLASS_NAMES_PATH, IMG_SIZE, SEED, SPLIT_FRACTIONS, SPLIT_PATH)
from .data_utils import find_dataset_root, get_or_create_split

AUTOTUNE = tf.data.AUTOTUNE


def _decode(path: tf.Tensor, label: tf.Tensor):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)            # bilinear
    img = tf.cast(img, tf.float32)                  # keep 0-255
    img.set_shape((*IMG_SIZE, 3))
    return img, label


def make_dataset(root, paths, labels, batch_size: int, training: bool) -> tf.data.Dataset:
    full_paths = [str(Path(root) / p) for p in paths]
    ds = tf.data.Dataset.from_tensor_slices((full_paths, np.asarray(labels, dtype=np.int32)))
    if training:
        ds = ds.shuffle(len(full_paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def prepare_data(data_dir, batch_size: int, max_per_class: int | None = None,
                 seed: int = SEED, resplit: bool = False) -> dict:
    """Build train/val/test datasets and persist the split + class names."""
    root = find_dataset_root(data_dir)
    split = get_or_create_split(root, SPLIT_PATH, SPLIT_FRACTIONS, seed, max_per_class, force=resplit)
    class_names = split["class_names"]
    CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2))

    root_used = Path(split["meta"]["root"])
    datasets = {
        name: make_dataset(root_used, split[name]["paths"], split[name]["labels"],
                           batch_size, training=(name == "train"))
        for name in ("train", "val", "test")
    }
    print(f"[data] root={root_used}")
    print(f"[data] {len(class_names)} classes | train={len(split['train']['paths'])} "
          f"val={len(split['val']['paths'])} test={len(split['test']['paths'])}")
    return {"class_names": class_names, "split": split, "root": root_used, **{f"{k}_ds": v for k, v in datasets.items()}}
