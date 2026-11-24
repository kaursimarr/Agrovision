"""Generate a small SYNTHETIC leaf dataset for smoke-testing the pipeline offline.

The images are drawings (green ellipse + class-specific spots), NOT real leaves.
Use it only to verify that training / evaluation / the app run end-to-end:

    python scripts/make_demo_dataset.py --out data/PlantVillage --per-class 60
    python -m src.train --model all --no-pretrained --epochs 2 --head-epochs 1 --finetune-epochs 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight", "Potato___healthy"]


def draw_leaf(cls: str, rng: np.random.Generator, size: int = 256) -> Image.Image:
    bg = tuple(int(v) for v in rng.integers(150, 200, 3))
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    green = (int(rng.integers(20, 70)), int(rng.integers(110, 170)), int(rng.integers(20, 70)))
    box = (30 + rng.integers(-8, 8), 50 + rng.integers(-8, 8), size - 30 + rng.integers(-8, 8), size - 50 + rng.integers(-8, 8))
    d.ellipse(box, fill=green)
    d.line((box[0], size // 2, box[2], size // 2), fill=(15, 90, 15), width=3)  # midrib
    cx, cy = size // 2, size // 2
    if cls == "Tomato___Early_blight":        # brown concentric rings
        for _ in range(int(rng.integers(3, 7))):
            x, y = cx + rng.integers(-70, 70), cy + rng.integers(-40, 40)
            for r, col in ((18, (110, 70, 20)), (11, (150, 110, 40)), (5, (90, 50, 10))):
                d.ellipse((x - r, y - r, x + r, y + r), outline=col, width=3)
    elif cls == "Tomato___Late_blight":       # large dark water-soaked blotches
        for _ in range(int(rng.integers(2, 5))):
            x, y = cx + rng.integers(-60, 60), cy + rng.integers(-35, 35)
            r = int(rng.integers(18, 34))
            d.ellipse((x - r, y - r, x + r, y + r), fill=(45, 40, 25))
    elif cls == "Potato___healthy":           # lighter green with pale veins
        for k in range(-3, 4):
            d.line((cx, cy, cx + 90, cy + k * 25), fill=(120, 190, 110), width=2)
            d.line((cx, cy, cx - 90, cy + k * 25), fill=(120, 190, 110), width=2)
    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 1.2))))
    arr = np.asarray(img, dtype=np.float32) + rng.normal(0, 6, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/PlantVillage")
    p.add_argument("--per-class", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    for cls in CLASSES:
        folder = Path(args.out) / cls
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            draw_leaf(cls, rng).save(folder / f"{cls}_{i:04d}.jpg", quality=92)
    print(f"Wrote {len(CLASSES) * args.per_class} synthetic images to {args.out}")


if __name__ == "__main__":
    main()
