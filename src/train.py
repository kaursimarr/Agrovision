"""Train the custom CNN and/or the transfer-learning models.

Examples
--------
    python -m src.train --model all
    python -m src.train --model mobilenetv2 --max-per-class 300      # quick CPU run
    python -m src.train --model cnn --epochs 30
"""
from __future__ import annotations

import argparse
import json
import time

import keras

from . import config as C
from .data_loader import prepare_data
from .data_utils import compute_class_weights
from .models import build_model, compile_model, unfreeze_top_layers
from .plots import plot_history


def _callbacks(patience: int) -> list:
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=max(1, patience // 2), min_lr=1e-7, verbose=1),
    ]


def _merge(histories) -> dict:
    merged: dict[str, list[float]] = {}
    for h in histories:
        for key, values in h.history.items():
            merged.setdefault(key, []).extend(float(v) for v in values)
    return merged


def train_one(name: str, data: dict, args) -> dict:
    print(f"\n{'=' * 70}\n Training: {C.DISPLAY_NAMES[name]}\n{'=' * 70}")
    keras.utils.set_random_seed(C.SEED)
    num_classes = len(data["class_names"])
    model, base = build_model(name, num_classes, pretrained=not args.no_pretrained)
    model.summary(line_length=100)

    class_weight = None
    if not args.no_class_weight:
        class_weight = compute_class_weights(data["split"]["train"]["labels"], num_classes)

    fit_kwargs = dict(validation_data=data["val_ds"], class_weight=class_weight, verbose=2)
    histories, boundary = [], None
    start = time.time()

    if name in ("cnn", "vit"):
        lr = C.LR_VIT if name == "vit" else C.LR_CNN
        epochs = args.epochs or (C.VIT_EPOCHS if name == "vit" else C.CNN_EPOCHS)
        compile_model(model, lr)
        histories.append(model.fit(data["train_ds"], epochs=epochs,
                                   callbacks=_callbacks(patience=5), **fit_kwargs))
    else:
        # Phase 1: train only the new classification head on the frozen backbone.
        compile_model(model, C.LR_HEAD)
        h1 = model.fit(data["train_ds"], epochs=args.head_epochs, callbacks=_callbacks(patience=3), **fit_kwargs)
        histories.append(h1)
        boundary = len(h1.history["loss"])
        # Phase 2: unfreeze the top backbone layers and fine-tune with a tiny learning rate.
        unfreeze_top_layers(base, args.fine_tune_layers)
        compile_model(model, C.LR_FINE_TUNE)
        histories.append(model.fit(data["train_ds"], epochs=args.finetune_epochs,
                                   callbacks=_callbacks(patience=4), **fit_kwargs))

    train_seconds = time.time() - start
    history = _merge(histories)
    history.update(phase_boundary=boundary, train_time_sec=train_seconds, epochs_run=len(history["loss"]))

    out_dir = C.OUTPUTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(C.model_path(name))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, out_dir / "training_curves.png", C.DISPLAY_NAMES[name])
    print(f"[train] saved {C.model_path(name)} | {history['epochs_run']} epochs | {train_seconds / 60:.1f} min")
    return history


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="all", choices=[*C.MODEL_NAMES, "all"])
    p.add_argument("--data-dir", default=str(C.DATA_DIR))
    p.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=None, help="Custom CNN epochs (default from config)")
    p.add_argument("--head-epochs", type=int, default=C.HEAD_EPOCHS, help="Transfer learning phase 1 epochs")
    p.add_argument("--finetune-epochs", type=int, default=C.FINE_TUNE_EPOCHS, help="Transfer learning phase 2 epochs")
    p.add_argument("--fine-tune-layers", type=int, default=C.FINE_TUNE_LAYERS)
    p.add_argument("--max-per-class", type=int, default=None, help="Use at most N images per class (fast runs)")
    p.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet weights (offline smoke test)")
    p.add_argument("--no-class-weight", action="store_true", help="Disable balanced class weights")
    p.add_argument("--resplit", action="store_true", help="Force a new train/val/test split")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    data = prepare_data(args.data_dir, args.batch_size, max_per_class=args.max_per_class, resplit=args.resplit)
    names = C.MODEL_NAMES if args.model == "all" else (args.model,)
    for name in names:
        train_one(name, data, args)
    print("\nDone. Next: python -m src.evaluate")


if __name__ == "__main__":
    main()
