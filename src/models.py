"""Model definitions: a custom CNN and two transfer-learning models.

All models take RAW images (float 0-255, shape 224x224x3). Data augmentation and the
model-specific normalisation are layers *inside* the model:
  * augmentation layers are only active during training,
  * normalisation matches what the ImageNet backbone expects.
"""
from __future__ import annotations

import tensorflow as tf
import keras
from keras import layers

from .config import (IMG_SHAPE, VIT_DROPOUT, VIT_MLP_DIM, VIT_NUM_HEADS,
                    VIT_PATCH_SIZE, VIT_PROJECTION_DIM, VIT_TRANSFORMER_LAYERS)


@keras.saving.register_keras_serializable(package="plantdisease")
class AddPositionEmbs(layers.Layer):
    """Learnable 1D positional embedding added to patch sequences.

    Registered as a serializable layer so .keras files reload cleanly.
    """

    def __init__(self, num_patches: int, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = int(num_patches)
        self.projection_dim = int(projection_dim)

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.num_patches, self.projection_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_embedding

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return cfg


@keras.saving.register_keras_serializable(package="plantdisease")
class ResNetPreprocess(layers.Layer):
    """Equivalent of ``resnet50.preprocess_input``: RGB->BGR and ImageNet mean subtraction.

    Implemented as a registered layer so ``.keras`` files reload without Lambda/unsafe code.
    """

    _MEAN = (103.939, 116.779, 123.68)

    def call(self, x):
        x = tf.reverse(x, axis=[-1])
        return x - tf.constant(self._MEAN, dtype=x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape


def build_augmentation(name: str = "augmentation") -> keras.Sequential:
    """Random flips, rotation, zoom, shift and contrast (training only)."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15, fill_mode="reflect"),
            layers.RandomZoom(0.15, fill_mode="reflect"),
            layers.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
            layers.RandomContrast(0.15),
        ],
        name=name,
    )


def build_custom_cnn(num_classes: int, img_shape=IMG_SHAPE) -> keras.Model:
    """Four conv blocks (32-64-128-256) with BatchNorm, pooling and dropout."""
    inputs = keras.Input(shape=img_shape, name="image")
    x = build_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255, name="rescale")(x)
    for filters in (32, 64, 128, 256):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="custom_cnn")


def build_transfer_model(name: str, num_classes: int, pretrained: bool = True,
                         img_shape=IMG_SHAPE, dropout: float = 0.3):
    """ImageNet backbone (frozen) + global pooling + dropout + softmax head.

    Returns ``(model, backbone)`` so the trainer can unfreeze the backbone later.
    """
    weights = "imagenet" if pretrained else None
    if name == "mobilenetv2":
        base = keras.applications.MobileNetV2(include_top=False, weights=weights, input_shape=img_shape)
        preprocess = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")  # -> [-1, 1]
    elif name == "resnet50":
        base = keras.applications.ResNet50(include_top=False, weights=weights, input_shape=img_shape)
        preprocess = ResNetPreprocess(name="resnet_preprocess")
    else:
        raise ValueError(f"Unknown transfer model: {name}")

    base.trainable = False
    inputs = keras.Input(shape=img_shape, name="image")
    x = build_augmentation()(inputs)
    x = preprocess(x)
    x = base(x, training=False)          # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=name), base


def build_vit(
    num_classes: int,
    img_shape=IMG_SHAPE,
    patch_size: int = VIT_PATCH_SIZE,
    projection_dim: int = VIT_PROJECTION_DIM,
    num_heads: int = VIT_NUM_HEADS,
    transformer_layers: int = VIT_TRANSFORMER_LAYERS,
    mlp_dim: int = VIT_MLP_DIM,
    dropout: float = VIT_DROPOUT,
) -> keras.Model:
    """Vision Transformer (ViT) architecture based on Dosovitskiy et al. (2020).

    Splits images into non-overlapping patches, linearly projects them, adds
    learnable positional embeddings, processes them through a sequence of
    Transformer Encoder blocks (Multi-Head Self-Attention + MLP with GELU),
    and classifies via sequence pooling and an MLP prediction head.
    """
    num_patches = (img_shape[0] // patch_size) * (img_shape[1] // patch_size)

    inputs = keras.Input(shape=img_shape, name="image")
    # Data augmentation active during training
    x = build_augmentation(name="vit_augmentation")(inputs)
    # In-model normalization to [0, 1]
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

    # Patch extraction & linear projection via non-overlapping Conv2D
    x = layers.Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="patch_conv",
    )(x)
    x = layers.Reshape((num_patches, projection_dim), name="patch_flatten")(x)

    # Positional embedding + dropout
    x = AddPositionEmbs(num_patches=num_patches, projection_dim=projection_dim, name="pos_emb")(x)
    x = layers.Dropout(dropout, name="pos_drop")(x)

    # Transformer Encoder blocks
    for i in range(transformer_layers):
        # Multi-Head Attention block with residual
        norm1 = layers.LayerNormalization(epsilon=1e-6, name=f"ln1_{i}")(x)
        attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=dropout,
            name=f"mha_{i}",
        )(norm1, norm1)
        x = layers.Add(name=f"attn_add_{i}")([x, attn])

        # MLP block with residual
        norm2 = layers.LayerNormalization(epsilon=1e-6, name=f"ln2_{i}")(x)
        mlp = layers.Dense(mlp_dim, activation="gelu", name=f"mlp1_{i}")(norm2)
        mlp = layers.Dropout(dropout, name=f"mlp_drop_{i}")(mlp)
        mlp = layers.Dense(projection_dim, name=f"mlp2_{i}")(mlp)
        mlp = layers.Dropout(dropout, name=f"mlp_out_drop_{i}")(mlp)
        x = layers.Add(name=f"mlp_add_{i}")([x, mlp])

    # Representation head: sequence normalization + pooling
    x = layers.LayerNormalization(epsilon=1e-6, name="head_ln")(x)
    x = layers.GlobalAveragePooling1D(name="seq_pooling")(x)
    x = layers.Dropout(0.2, name="head_dropout")(x)

    # Classification MLP head
    x = layers.Dense(mlp_dim // 2, activation="gelu", name="head_mlp")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="vit")


def build_model(name: str, num_classes: int, pretrained: bool = True):
    """Factory. Returns ``(model, backbone_or_None)``."""
    if name == "cnn":
        return build_custom_cnn(num_classes), None
    if name == "vit":
        return build_vit(num_classes), None
    return build_transfer_model(name, num_classes, pretrained=pretrained)


def unfreeze_top_layers(base: keras.Model, n_layers: int) -> None:
    """Fine-tuning: make the last ``n_layers`` of the backbone trainable (BatchNorm stays frozen)."""
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
