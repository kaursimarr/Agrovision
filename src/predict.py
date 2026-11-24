"""Single-image inference, used by the Streamlit app and usable from the command line.

    python -m src.predict --image leaf.jpg --model mobilenetv2 --top-k 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import keras

from . import config as C
from .labels import parse_label
from .models import (AddPositionEmbs,  # noqa: F401  (registers the custom layer for loading)
                    ResNetPreprocess)


def load_class_names(path: Path = C.CLASS_NAMES_PATH) -> list[str]:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - train a model first (python -m src.train).")
    return json.loads(Path(path).read_text())


class Predictor:
    """Loads a saved ``.keras`` model and classifies PIL images."""

    def __init__(self, model_file: str | Path, class_names: list[str] | None = None):
        self.model = keras.models.load_model(model_file, compile=False)
        self.class_names = class_names or load_class_names()
        out_units = self.model.output_shape[-1]
        if out_units != len(self.class_names):
            raise ValueError(f"Model outputs {out_units} classes but class_names.json lists "
                             f"{len(self.class_names)}. Retrain or restore the matching class_names.json.")

    @staticmethod
    def preprocess(image: Image.Image) -> np.ndarray:
        """Same decode path as training: RGB, bilinear resize to 224x224, raw 0-255 floats."""
        image = ImageOps.exif_transpose(image).convert("RGB")
        arr = tf.image.resize(np.asarray(image, dtype=np.float32), C.IMG_SIZE).numpy()
        return arr[None, ...]

    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict]:
        probs = self.model(self.preprocess(image), training=False).numpy()[0]
        order = np.argsort(probs)[::-1][:top_k]
        results = []
        for i in order:
            plant, condition, healthy = parse_label(self.class_names[i])
            results.append({"class_name": self.class_names[i], "plant": plant, "condition": condition,
                            "healthy": healthy, "confidence": float(probs[i])})
        return results


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="mobilenetv2", choices=C.MODEL_NAMES)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args(argv)
    predictor = Predictor(C.model_path(args.model))
    for rank, r in enumerate(predictor.predict(Image.open(args.image), args.top_k), 1):
        print(f"{rank}. {r['plant']} - {r['condition']:<40s} {r['confidence'] * 100:5.1f}%")


if __name__ == "__main__":
    main()
