"""Central configuration for the Plant Disease Detection project."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "PlantVillage"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SPLIT_PATH = OUTPUTS_DIR / "split.json"
CLASS_NAMES_PATH = OUTPUTS_DIR / "class_names.json"
COMPARISON_DIR = OUTPUTS_DIR / "comparison"

# ---------------------------------------------------------------- data
IMG_SIZE = (224, 224)
IMG_SHAPE = (*IMG_SIZE, 3)
BATCH_SIZE = 32
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# train / validation / test fractions (must sum to 1)
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)

# ---------------------------------------------------------------- training
CNN_EPOCHS = 20
HEAD_EPOCHS = 5           # transfer learning phase 1: frozen backbone
FINE_TUNE_EPOCHS = 8      # transfer learning phase 2: unfreeze top layers
FINE_TUNE_LAYERS = 30     # how many backbone layers to unfreeze in phase 2
LR_CNN = 1e-3
LR_HEAD = 1e-3
LR_FINE_TUNE = 1e-5

# ViT (Vision Transformer) parameters
VIT_PATCH_SIZE = 16
VIT_PROJECTION_DIM = 128
VIT_NUM_HEADS = 4
VIT_TRANSFORMER_LAYERS = 4
VIT_MLP_DIM = 256
VIT_DROPOUT = 0.1
VIT_EPOCHS = 20
LR_VIT = 1e-3

# ---------------------------------------------------------------- models
MODEL_NAMES = ("cnn", "mobilenetv2", "resnet50", "vit")
DISPLAY_NAMES = {
    "cnn": "Custom CNN (from scratch)",
    "mobilenetv2": "MobileNetV2 (transfer learning)",
    "resnet50": "ResNet50 (transfer learning)",
    "vit": "Vision Transformer (ViT from scratch)",
}


def model_path(name: str) -> Path:
    return MODELS_DIR / f"{name}.keras"
