# 🌿 Plant Disease Detection: CNN vs Transfer Learning vs Vision Transformer (ViT)

A Deep Learning mini-project that classifies plant leaf diseases from images using the **PlantVillage** dataset (38 plant/disease classes). It trains and compares four architectures (from scratch CNN, Vision Transformer, and two transfer-learning CNNs), evaluates them rigorously, and serves them through an interactive **Streamlit** web app.

| Model | Architecture | Paradigm |
|---|---|---|
| **Custom CNN** | 4 conv blocks (BatchNorm, pooling, dropout) | Trained from scratch |
| **Vision Transformer (ViT)** | 16x16 patches, Multi-Head Self-Attention, GELU MLP blocks | Trained from scratch |
| **MobileNetV2** | Inverted residual blocks, depthwise separable convs | Transfer learning (ImageNet) |
| **ResNet50** | Residual bottleneck blocks | Transfer learning (ImageNet) |

Transfer-learning models are trained in two phases: (1) train a new classification head on the frozen backbone, (2) fine-tune the top 30 backbone layers with a very small learning rate.
The Vision Transformer is trained end-to-end with patch projection, learnable 1D position embeddings, multi-head self-attention, and global sequence pooling.

## Features

- tf.data pipeline: decode, resize to 224x224, batch, prefetch
- Augmentation inside the model (flip, rotation, zoom, shift, contrast), active only during training
- Model-specific normalisation built into each model, so saved models accept raw images
- Stratified 70/15/15 train/val/test split, saved to `outputs/split.json` so all models use identical data
- Balanced class weights (PlantVillage is imbalanced)
- Early stopping and learning-rate scheduling
- Metrics: accuracy, precision, recall, F1 (macro and weighted), top-3 accuracy, per-class report, confusion matrix, parameters, model size, inference time, training time
- Plots: accuracy/loss curves, confusion matrices (raw and normalised), model comparison bars, validation curves overlay
- **Streamlit App (4-Model Diagnostic Arena)**:
  - **Equal prominence for all 4 models**: Vision Transformer (ViT), ResNet50, MobileNetV2, and Custom CNN are evaluated side-by-side simultaneously in a 4-column arena.
  - **Ensemble Consensus Verdict**: Automatic voting consensus banner calculating agreement ratio (e.g. 4/4 unanimous or 3/4 majority) and mean ensemble confidence.
  - **Redesigned dark obsidian theme & typography**: Styled with Google Fonts (`Plus Jakarta Sans` for UI and `JetBrains Mono` for telemetry), frosted glass borders, glowing circular confidence rings, and neon architectural accent colors.
  - **Interactive clinical care guide**: 5-tab protocol covering symptoms, causative agent, interactive tick-off agronomic action checklist with live progress bar, organic/cultural options, chemical fungicides, and sanitation.
  - **Instant sample buttons**: 1-click loading of test leaves (`leaf.jpg` for Tomato Late Blight, `plant1_cropped.png` for Strawberry Leaf Scorch).
  - **Export capabilities**: Download comprehensive Markdown clinical reports and CSV session audit logs.
  - **Benchmarking & Deep Dive**: Live comparisons of accuracy, inference latency, model footprints, confusion matrices, and per-class F1 rankings.
  - **Pathology Encyclopedia**: Searchable and filterable reference across all 38 classes.

## Project structure

```
plant_disease_detection/
├── app.py                    # Streamlit interface
├── requirements.txt
├── src/
│   ├── config.py             # paths and hyper-parameters (CNN, ViT, MobileNetV2, ResNet50)
│   ├── data_utils.py         # dataset discovery, stratified split, class weights
│   ├── data_loader.py        # tf.data pipeline
│   ├── models.py             # CNN, ViT, MobileNetV2, ResNet50, fine-tuning helpers
│   ├── train.py              # training CLI
│   ├── evaluate.py           # evaluation + comparison CLI
│   ├── metrics.py            # sklearn metrics
│   ├── plots.py              # matplotlib figures
│   ├── predict.py            # single-image inference (also a CLI)
│   └── labels.py             # folder name -> "Plant / Condition"
├── scripts/
│   ├── download_dataset.py   # optional Kaggle download
│   └── make_demo_dataset.py  # synthetic data for an offline smoke test
└── tests/
    ├── test_vit.py           # unit tests for Vision Transformer
    ├── test_non_tf.py        # tests for data pipeline and metrics
    ├── test_knowledge_base.py# tests for disease info and care guide
    └── app_smoke_test.py     # Streamlit app smoke test
```

## Vision Transformer (ViT) Architecture

The Vision Transformer treats an image as a sequence of patches, analogous to tokens in Natural Language Processing:

```
Input Image (224x224x3)
       │
       ▼
Non-Overlapping Patches (16x16, 196 patches)
       │
       ▼
Linear Projection to Embedding Space (d = 128)
       │
       ▼
Add Learnable 1D Positional Embeddings
       │
       ▼
┌──────────────────────────────────────────────┐
│ Transformer Encoder Block (x4 layers)        │
│ ┌──────────────────────────────────────────┐ │
│ │ Pre-LayerNorm                            │ │
│ │ Multi-Head Self-Attention (4 heads)      │ │
│ │ Residual Skip Connection (+)             │ │
│ └──────────────────────────────────────────┘ │
│ ┌──────────────────────────────────────────┐ │
│ │ Pre-LayerNorm                            │ │
│ │ MLP (Dense 256 -> GELU -> Dense 128)     │ │
│ │ Residual Skip Connection (+)             │ │
│ └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
       │
       ▼
LayerNorm & Global Sequence Pooling (Average)
       │
       ▼
MLP Head (Dense 128 -> GELU -> Dropout 0.2)
       │
       ▼
Softmax Classification (38 Classes)
```

**Key Advantages of ViT over CNNs:**
- **Global Receptive Field:** CNNs have local receptive fields that only expand gradually layer-by-layer; self-attention computes relationships between all pairs of image patches from layer 1.
- **Dynamic Content-Based Attention:** Attention weights are computed dynamically based on patch representations rather than fixed convolutional filters.
- **High Parameter Efficiency:** The ViT model uses only **0.68M parameters (2.9 MB)**—smaller than Custom CNN (1.25M / 15.2 MB) and ResNet50 (23.67M / 211.4 MB)—while delivering strong competitive accuracy (91.85%).

## 1. Setup

Use **Python 3.10 to 3.12** (TensorFlow does not support every newest Python release; check tensorflow.org/install).

Check your version first with `python --version`.

**Windows (PowerShell):**

```powershell
cd plant_disease_detection
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and retry. In Command Prompt use `.venv\Scripts\activate.bat`.

**macOS / Linux:**

```bash
cd plant_disease_detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

When the environment is active, the prompt starts with `(.venv)`.

A GPU is strongly recommended for full training. Without one, use `--max-per-class` (see below).

## 2. Get the dataset

**Option A: Kaggle website.** Download "PlantVillage Dataset" (e.g. `abdallahalidev/plantvillage-dataset`), unzip, and place the class folders so you have:

```
data/PlantVillage/Apple___Apple_scab/*.jpg
data/PlantVillage/Tomato___healthy/*.jpg
...
```

Nested layouts such as `PlantVillage/PlantVillage/<classes>` or `.../color/<classes>` are detected automatically. Use the **colour** images.

**Option B: script** (needs a Kaggle API token):

```bash
pip install kagglehub
python scripts/download_dataset.py
```

You can also point anywhere with `--data-dir /path/to/PlantVillage`.

## 2b. No GPU? Train on Kaggle instead (recommended)

`kaggle_train.ipynb` is a self-contained notebook: it contains all the project code, trains the three models on Kaggle's free GPU and packages the results.

1. On kaggle.com choose **Create, then New Notebook, then File, then Import Notebook** and upload `kaggle_train.ipynb`.
2. In the notebook settings set **Accelerator = GPU** and **Internet = On** (Internet needs a phone-verified account).
3. **Add Input**, search **PlantVillage Dataset** (Abdallah Ali, 38 classes) and add it.
4. **Run All**. When it finishes, download `results.zip` from the Output panel.
5. Unzip `results.zip` into this project folder so that `models/` and `outputs/` sit next to `app.py`.
6. Run `streamlit run app.py`. The dataset and training are not needed on your PC.

If the app cannot load a model, install the TensorFlow version printed in `TF_VERSION.txt` inside the zip.

## 3. Train (locally)

```bash
# all four models (GPU recommended)
python -m src.train --model all

# quick run on a laptop CPU: at most 300 images per class
python -m src.train --model all --max-per-class 300

# train the Vision Transformer individually
python -m src.train --model vit --epochs 20

# a single CNN model, custom epochs
python -m src.train --model cnn --epochs 30
python -m src.train --model resnet50 --head-epochs 5 --finetune-epochs 10
```

Useful flags: `--batch-size`, `--fine-tune-layers`, `--no-class-weight`, `--resplit`, `--no-pretrained`. Defaults live in `src/config.py`.

Outputs: `models/<name>.keras`, `outputs/<name>/history.json`, `outputs/<name>/training_curves.png`.

If you change `--max-per-class` or the dataset, the split is regenerated; retrain all models afterwards so none is tested on images it saw during training.

## 4. Evaluate and compare

```bash
python -m src.evaluate
```

Creates, under `outputs/`:

- `comparison/metrics_comparison.csv` and `.png`: the CNN vs transfer learning table and chart
- `comparison/training_curves_comparison.png`
- `<model>/confusion_matrix.png`, `confusion_matrix_normalized.png`
- `<model>/classification_report.csv` and `metrics.json`

## 5. Run the app

```bash
streamlit run app.py
```

Open the URL shown (usually http://localhost:8501), pick a model in the sidebar, upload a leaf image, and read the prediction and confidence. A command-line alternative:

```bash
python -m src.predict --image leaf.jpg --model vit --top-k 3
python -m src.predict --image leaf.jpg --model mobilenetv2 --top-k 3
```

## 6. Smoke test without the real dataset

To check that the whole pipeline runs on your machine in a couple of minutes (the images are synthetic drawings, so the accuracy is meaningless):

```bash
python scripts/make_demo_dataset.py --out data/PlantVillage --per-class 60
python -m src.train --model all --no-pretrained --epochs 2 --head-epochs 1 --finetune-epochs 1
python -m src.evaluate
streamlit run app.py
```

Delete `data/PlantVillage`, `outputs/` and `models/` before training on the real data. TensorFlow-free unit tests: `python tests/test_non_tf.py`, `python tests/test_knowledge_base.py`, and `python tests/app_smoke_test.py` (runs `app.py` against a stand-in Streamlit and fake models).

## Model Comparison & Benchmark Results

Evaluated on the held-out test set (4,795 samples across 38 classes):

| Model | Test Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Top-3 Accuracy | Params (M) | Size (MB) | Inference (ms/img) | Train Time (min) |
|---|---|---|---|---|---|---|---|---|---|
| **Custom CNN (from scratch)** | 90.22% | 0.8973 | 0.9144 | 0.8980 | 98.87% | 1.25 M | 15.2 MB | 2.2 ms | 81.3 min |
| **Vision Transformer (ViT from scratch)** | **93.89%** | **0.9295** | **0.9438** | **0.9345** | **99.35%** | **0.68 M** | **8.4 MB** | **0.8 ms** | **15.3 min** |
| **MobileNetV2 (transfer learning)** | 95.06% | 0.9509 | 0.9527 | 0.9503 | 99.48% | 2.31 M | 22.3 MB | 2.1 ms | 11.8 min |
| **ResNet50 (transfer learning)** | 97.43% | 0.9738 | 0.9758 | 0.9743 | 99.81% | 23.67 M | 211.4 MB | 5.1 ms | 31.1 min |

## Suggested report discussion

- **CNN vs Vision Transformer (scratch models):** ViT achieves **93.89% accuracy**, outperforming the Custom CNN (**90.22%**) by nearly **3.7% in accuracy** and **0.036 in macro F1**, while requiring only **0.68M parameters**. Furthermore, ViT trained in just **15.3 minutes** on GPU (vs **81.3 minutes** for the Custom CNN—a **5.3x speedup**) and achieves the fastest inference latency across all architectures at **0.8 ms per image**. Self-attention across 16x16 patches enables the model to capture non-local leaf lesions and spot patterns simultaneously across the whole leaf blade.
- **Transfer learning vs From-scratch:** Pre-trained ImageNet backbones (MobileNetV2 and ResNet50) achieve top performance (95.06% and 97.43%) because they reuse rich low- and mid-level visual features learned from 1.28 million natural images. ResNet50 attains the highest accuracy (97.43%), while MobileNetV2 provides a strong balance of high accuracy (95.06%) and low footprint (2.31M params).
- **Confusion matrix:** find the confused pairs (often diseases of the same crop, such as the Tomato blights and spots) and explain them visually.
- **Data leakage caveat:** PlantVillage contains several photos of the same leaf, so a random split can overestimate real-world accuracy. Mention this, and that images have uniform lab backgrounds.
- **Fine-tuning:** the dashed line in the curves marks where fine-tuning starts.
- **Ideas for future work:** Self-supervised pretraining (MAE / DINO) for ViT on plant leaves, attention map rollouts for visual interpretability, and ONNX/TFLite export.

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named tensorflow` | Use Python 3.10 to 3.12 and re-run `pip install -r requirements.txt` |
| Out of memory | Lower `--batch-size` (e.g. 16) |
| Very slow on CPU | Use `--max-per-class 200` and fewer epochs |
| ImageNet weights fail to download | Check your internet connection, or use `--no-pretrained` (accuracy will be lower) |
| App says no trained model | Run `python -m src.train` first; the app reads `models/` and `outputs/class_names.json` |
| "Model outputs N classes but class_names.json lists M" | The model and class list come from different datasets; retrain |
