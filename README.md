# 🌿 AGROVISION — AI Crop Diagnostic Console

> **Plant Disease Detection & Classification | Multi-Architecture Deep Learning (ViT, ResNet50, MobileNetV2, Custom CNN)**

AGROVISION is a production-grade AI agricultural diagnostic console powered by deep learning. It evaluates leaf images across **4 distinct neural architectures simultaneously**, calculates an **Ensemble Consensus Verdict**, provides real-time **Image Quality Telemetry** (Laplacian focus & exposure), and dynamically prescribes **Agronomic Treatment Protocols** across 38 crop disease classes.

---

## ⚡ Quick Start for Teammates (No Training Required!)

You **do not need to train the models** or download full datasets to run the application on your computer.

### 1. Clone the repository
```bash
git clone https://github.com/kaursimarr/Agrovision.git
cd Agrovision
```

### 2. Set up the virtual environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Add the Trained Model Weights (`models/`)
> [!IMPORTANT]
> GitHub enforces a strict **100 MB per-file limit**. Because `resnet50.keras` is ~201 MB, the `.keras` model weights are excluded from Git via `.gitignore`. 
> All benchmark metrics, history curves, and class mappings (`outputs/`) are already included in this repository.

To enable inference, copy the 4 pre-trained model files into the `models/` directory:
- `models/vit.keras` (~8.4 MB)
- `models/mobilenetv2.keras` (~22.3 MB)
- `models/cnn.keras` (~15.2 MB)
- `models/resnet50.keras` (~201.4 MB)

*(You can obtain these from your team's shared Google Drive or extract them from the Kaggle `results.zip` output).*

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🧠 Multi-Architecture Overview

| Model | Architecture | Parameter Count | Latency | Paradigm |
| :--- | :--- | :--- | :--- | :--- |
| **Vision Transformer (ViT)** | 16×16 Patches, 4-Head Attention, Pre-LN | **0.68 M** (8.4 MB) | **0.8 ms** | Trained from scratch |
| **ResNet50** | 50-Layer Deep Residual Bottleneck | 23.67 M (201.4 MB) | 5.1 ms | Transfer Learning (ImageNet) |
| **MobileNetV2** | Inverted Residuals & Linear Bottlenecks | 2.31 M (22.3 MB) | 2.1 ms | Transfer Learning (ImageNet) |
| **Custom CNN** | 4-Stage ConvNet with BatchNorm & Dropout | 1.25 M (15.2 MB) | 2.2 ms | Trained from scratch |

---

## 🚀 Key System Features

- **4-Architecture Diagnostic Arena**: Evaluates leaf photos simultaneously across ViT, ResNet50, MobileNetV2, and Custom CNN.
- **Ensemble Consensus Engine**: Majority voting consensus algorithm determining diagnostic agreement and mean ensemble confidence.
- **Computer Vision Quality Telemetry**:
  - **Resolution**: Flags images under $150\text{px}$.
  - **Exposure & Glare**: Photometric luminance analysis.
  - **Focus Index**: Discrete 2D Laplacian operator variance ($\nabla^2 I$) to detect blurry lesions.
- **Agronomic Treatment Protocol**:
  - **Healthy Crops**: Canopy health monitoring and vigor maintenance schedules.
  - **Diseased Crops**: Action checklists, biological controls, chemical fungicides, and prevention protocols.
- **Interactive Multi-Tab Console**:
  - `🎯 Diagnostic Console`: Image scanner, camera input, and consensus diagnosis.
  - `📊 Multi-Architecture Benchmarks`: Comparative ROC, accuracy, loss, and latency metrics.
  - `📚 Pathology Encyclopedia`: Searchable reference covering all 38 PlantVillage classes.
  - `⚙️ System Architecture`: Deep dive into layer topologies and pipeline specifications.
- **Instant Field Samples**: 1-click test leaves for instant demonstration (`leaf.jpg` for Tomato Late Blight, `plant1_cropped.png` for Strawberry Leaf Scorch).

---

## 🔬 Vision Transformer (ViT) Architecture Details

The Vision Transformer divides the $224 \times 224$ input image into non-overlapping $16 \times 16$ patches ($N = 196$ tokens):

```
Input Leaf Image (224×224×3)
       │
       ▼
Non-Overlapping Patches (16×16, 196 patches)
       │
       ▼
Linear Projection to Embedding Space (d = 64)
       │
       ▼
Add Learnable 1D Positional Embeddings
       │
       ▼
┌──────────────────────────────────────────────┐
│ Transformer Encoder Blocks (×4 Layers)       │
│  ├─ Layer Normalization                      │
│  ├─ Multi-Head Self-Attention (4 Heads)      │
│  ├─ Residual Skip Connection (+)             │
│  ├─ Layer Normalization                      │
│  ├─ MLP Block (GELU Activations)             │
│  └─ Residual Skip Connection (+)             │
└──────────────────────────────────────────────┘
       │
       ▼
Layer Normalization & Global Sequence Pooling
       │
       ▼
MLP Classification Head -> Softmax (38 Classes)
```

---

## 📊 Benchmark Evaluation Results

Evaluated on the held-out test split (4,795 samples across 38 classes):

| Model | Test Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Top-3 Accuracy | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet50** | **97.43%** | **0.9738** | **0.9758** | **0.9743** | **99.81%** | 5.1 ms |
| **MobileNetV2** | **95.06%** | **0.9509** | **0.9527** | **0.9503** | **99.48%** | 2.1 ms |
| **Vision Transformer (ViT)** | **93.89%** | **0.9295** | **0.9438** | **0.9345** | **99.35%** | **0.8 ms** |
| **Custom CNN** | **90.22%** | **0.8973** | **0.9144** | **0.8980** | **98.87%** | 2.2 ms |

---

## 🛠️ Optional: Training & Evaluation (For Developers)

If you wish to retrain the models from scratch:

```bash
# Train all 4 models locally (GPU recommended)
python -m src.train --model all

# Train individual architectures
python -m src.train --model vit --epochs 20
python -m src.train --model resnet50 --head-epochs 5 --finetune-epochs 10

# Generate full benchmark metrics and confusion matrices
python -m src.evaluate
```

Or run the self-contained training notebook on Kaggle GPU: [`kaggle_train.ipynb`](file:///C:/deep_learning/plant_disease_detection/kaggle_train.ipynb).

---

## 🧪 Testing

To run the automated smoke tests and TensorFlow-free test suites:
```powershell
python tests/app_smoke_test.py
python tests/test_non_tf.py
python tests/test_knowledge_base.py
python tests/test_vit.py
```

---

## 📂 Project Directory Structure

```
Agrovision/
├── app.py                     # Streamlit Application & Multi-Architecture Dashboard
├── requirements.txt           # Python dependencies
├── README.md                  # System documentation
├── .streamlit/
│   └── config.toml           # Theme configuration
├── src/
│   ├── ui.py                  # Obsidian UI Design System & Component Library
│   ├── models.py              # ViT, ResNet50, MobileNetV2 & Custom CNN definitions
│   ├── disease_info.py        # Clinical agronomic knowledge base & protocols
│   ├── config.py              # Hyperparameters & path definitions
│   ├── data_loader.py         # tf.data preprocessing & augmentation pipeline
│   ├── data_utils.py          # Dataset discovery & stratified split
│   ├── evaluate.py            # Comprehensive evaluation pipeline
│   ├── metrics.py             # Classification metrics & confusion matrices
│   ├── plots.py               # Visual reporting & comparison figures
│   └── predict.py             # Inference utility CLI
├── models/                    # Trained .keras weights (placed here)
│   └── .gitkeep
├── outputs/                   # Benchmark metrics, class_names.json & training plots
│   ├── class_names.json       # 38 PlantVillage class mappings
│   └── comparison/            # Multi-model comparison graphs & metrics CSV
├── scripts/
│   ├── download_dataset.py    # Dataset downloader
│   └── make_demo_dataset.py   # Synthetic dataset generator
└── tests/                     # Unit & smoke test suite
```
