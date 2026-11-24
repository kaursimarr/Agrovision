"""Download PlantVillage from Kaggle (needs a free Kaggle account + API token).

    pip install kagglehub
    python scripts/download_dataset.py

Kaggle setup: https://www.kaggle.com/docs/api  (put kaggle.json in ~/.kaggle/ or set
KAGGLE_USERNAME / KAGGLE_KEY).  If this fails, download the zip manually in the browser and
unzip it into data/PlantVillage/ - see README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR  # noqa: E402
from src.data_utils import find_dataset_root, list_images  # noqa: E402

SLUG = "abdallahalidev/plantvillage-dataset"   # 38 classes, colour images


def main() -> None:
    try:
        import kagglehub
    except ImportError:
        sys.exit("Install kagglehub first:  pip install kagglehub")
    downloaded = Path(kagglehub.dataset_download(SLUG))
    root = find_dataset_root(downloaded)
    paths, _, classes = list_images(root)
    print(f"Downloaded to: {root}\nFound {len(paths)} images in {len(classes)} classes.")
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_DIR.exists():
        try:
            DATA_DIR.symlink_to(root, target_is_directory=True)
            print(f"Linked {DATA_DIR} -> {root}")
        except OSError:  # Windows without admin/developer mode cannot create symlinks
            print(f"Could not create a link. Train with:  python -m src.train --data-dir \"{root}\"")
    else:
        print(f"{DATA_DIR} already exists; train with:  python -m src.train --data-dir \"{root}\"")


if __name__ == "__main__":
    main()
