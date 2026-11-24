"""AgroVision AI: Production Multi-Architecture Plant Pathology Diagnostic System.

Evaluates leaf imagery across 4 deep learning architectures with equal prominence:
1. Vision Transformer (ViT)
2. ResNet50
3. MobileNetV2
4. Custom 4-Block CNN
"""
from __future__ import annotations

import base64
import hashlib
import importlib
import io
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from src import ui
importlib.reload(ui)  # Guarantees live hot-reload picks up UI changes

from src.config import (CLASS_NAMES_PATH, COMPARISON_DIR, DISPLAY_NAMES, MODEL_NAMES, MODELS_DIR,
                        OUTPUTS_DIR, model_path)
from src.disease_info import CHEMICAL_NOTE, DISCLAIMER, DISEASE_TYPES, get_info
from src.labels import parse_label, pretty_label

# ------------------------------------------------------------------ System Configuration & Stylesheet
st.set_page_config(
    page_title="AGROVISION // AI Diagnostic Console",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(ui.CSS, unsafe_allow_html=True)

ss = st.session_state
if "history" not in ss:
    ss["history"] = []
    ss["history_keys"] = set()
if "sample_image" not in ss:
    ss["sample_image"] = None

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
CANONICAL_MODELS = ("vit", "resnet50", "mobilenetv2", "cnn")


# ------------------------------------------------------------------ Predictor & Data Helpers
def available_models() -> list[str]:
    found = [m for m in CANONICAL_MODELS if model_path(m).exists()]
    extras = sorted(p.stem for p in MODELS_DIR.glob("*.keras") if p.stem not in found)
    return found + extras


def load_class_names() -> list[str]:
    if CLASS_NAMES_PATH.exists():
        try:
            return json.loads(CLASS_NAMES_PATH.read_text())
        except Exception:
            return []
    return []


@st.cache_resource(show_spinner=False)
def get_predictor(name: str):
    from src.predict import Predictor
    return Predictor(model_path(name))


@st.cache_data(show_spinner=False, max_entries=512)
def predict_cached(model_name: str, image_bytes: bytes) -> list[dict]:
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    return get_predictor(model_name).predict(image, top_k=5)


def compute_consensus(all_preds: dict[str, list[dict]]) -> tuple[dict, int, float]:
    """Finds the majority consensus winner across all 4 architectures."""
    if not all_preds:
        return {}, 0, 0.0

    top_classes = [preds[0]["class_name"] for preds in all_preds.values()]
    counts = Counter(top_classes)
    winner_class, count = counts.most_common(1)[0]

    winner_confs = []
    winner_dict = None
    for preds in all_preds.values():
        if preds[0]["class_name"] == winner_class:
            winner_confs.append(preds[0]["confidence"])
            if winner_dict is None:
                winner_dict = preds[0]

    if winner_dict is None:
        first_m = next(iter(all_preds))
        winner_dict = all_preds[first_m][0]
        winner_confs = [winner_dict["confidence"]]

    avg_conf = (sum(winner_confs) / len(winner_confs)) * 100.0
    return winner_dict, count, avg_conf


def label_of(p: dict) -> str:
    return f"{p['plant']} - {p['condition']}"


def record_history(item: dict, top_pred: dict, model_label: str = "4-Model Ensemble") -> None:
    key = item["id"]
    if key in ss["history_keys"]:
        return
    ss["history_keys"].add(key)
    ss["history"].append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Image": item["name"],
        "Model": model_label,
        "Plant": top_pred["plant"],
        "Result": top_pred["condition"],
        "Confidence %": round(top_pred["confidence"] * 100, 1),
    })


def bullets(items: list[str]) -> None:
    if items:
        st.markdown("\n".join(f"- {t}" for t in items))


def img_to_data_uri(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()


# ------------------------------------------------------------------ Image Decoding Helper
def build_items(files) -> list[dict]:
    items = []
    for f in files:
        data = f.getvalue() if hasattr(f, "getvalue") else f
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except (UnidentifiedImageError, OSError, Exception):
            st.warning(f"Skipped '{getattr(f, 'name', 'unnamed file')}': unable to decode image.")
            continue
        items.append({
            "id": hashlib.md5(data).hexdigest()[:10],
            "name": getattr(f, "name", None) or "leaf_photo.png",
            "bytes": data,
            "image": ImageOps.exif_transpose(img).convert("RGB"),
        })
    return items


# ------------------------------------------------------------------ Sidebar Navigation & Controls
models = available_models()
class_names = load_class_names()

with st.sidebar:
    st.markdown(ui.render_sidebar_header(), unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section-title'>⚙️ Diagnostic Controls</div>", unsafe_allow_html=True)
    threshold = st.slider("Diagnostic Warning Threshold (%)", 30, 95, 65)
    top_k = st.slider("Predictions breakdown per model", 1, 5, 3)

    st.markdown("<div class='sidebar-section-title'>🌿 Instant Field Samples</div>", unsafe_allow_html=True)
    st.caption("Load verified benchmark leaf samples:")
    s1, s2 = st.columns(2)
    with s1:
        if st.button("🌿 Sample 1", use_container_width=True):
            p = Path("leaf.jpg")
            if p.exists():
                ss["sample_image"] = {"name": "leaf.jpg", "bytes": p.read_bytes()}
                st.rerun()
    with s2:
        if st.button("🍃 Sample 2", use_container_width=True):
            p = Path("plant1_cropped.png")
            if p.exists():
                ss["sample_image"] = {"name": "plant1_cropped.png", "bytes": p.read_bytes()}
                st.rerun()

    st.markdown("<div class='sidebar-section-title'>📷 Photography Best Practices</div>", unsafe_allow_html=True)
    st.markdown(
        """<ul class="tips-bullet-list">
            <li><b>Single leaf frame:</b> Fill ~70% of frame with the infected leaf blade.</li>
            <li><b>Exclude fingers &amp; hands:</b> Avoid occlusions in the image.</li>
            <li><b>Even lighting:</b> Use natural diffused light (preferably shade).</li>
            <li><b>Sharp focus:</b> Keep the leaf edges and lesion areas clear.</li>
        </ul>""",
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption(f"Session history: **{len(ss['history'])} diagnoses completed**")


# ------------------------------------------------------------------ Top Header Bar
st.markdown(ui.render_top_header(), unsafe_allow_html=True)


# ------------------------------------------------------------------ Main Navigation Tabs
tab_console, tab_benchmark, tab_encyclopedia, tab_specs = st.tabs(
    ["🎯 Diagnostic Console", "📊 Multi-Architecture Benchmarks", "📚 Pathology Encyclopedia", "⚙️ System Architecture"]
)

# ==================================================================
# TAB 1: DIAGNOSTIC CONSOLE
# ==================================================================
with tab_console:
    if not models or not class_names:
        st.info("Initialize the diagnostic engine by placing trained models in `models/*.keras` and `outputs/class_names.json`.")
    else:
        # Check instant sample trigger
        instant_sample_file = None
        if ss["sample_image"] is not None:
            class DummyFile:
                def __init__(self, data: bytes, name: str):
                    self._d, self.name = data, name
                def getvalue(self):
                    return self._d
            instant_sample_file = DummyFile(ss["sample_image"]["bytes"], ss["sample_image"]["name"])

        # Top row: 2 Balanced Columns
        col_scanner, col_result = st.columns([1, 1], gap="large")

        # ---------------- Left Column – Leaf Scanner Input
        with col_scanner:
            with st.container(key="scanner_card", border=True):
                st.markdown(
                    """<div class="ws-card-header">
                        <h3><span style="color:#06b6d4;">📤</span> Leaf Scanner Input</h3>
                    </div>
                    <p class="ws-card-subtitle">Upload a leaf image or capture live from field for diagnosis</p>""",
                    unsafe_allow_html=True,
                )

                input_mode = st.radio("Input Mode", ["Upload leaf photo", "Live field camera"], horizontal=True, label_visibility="collapsed")

                uploaded_files = []
                if input_mode == "Upload leaf photo":
                    uploaded = st.file_uploader("Upload leaf photo (JPG, PNG, BMP ≤ 200 MB)", type=IMAGE_TYPES, accept_multiple_files=True, label_visibility="collapsed")
                    if isinstance(uploaded, list):
                        uploaded_files = uploaded
                    elif uploaded is not None:
                        uploaded_files = [uploaded]
                else:
                    shot = st.camera_input("Capture live leaf photo", label_visibility="collapsed")
                    if shot is not None:
                        uploaded_files = [shot]

                files = uploaded_files if uploaded_files else ([instant_sample_file] if instant_sample_file else [])

                batch_slot = st.empty()
                preview_slot = st.empty()

        # ---------------- Inference Execution
        if not files:
            with col_result:
                st.markdown(ui.render_standby_banner(), unsafe_allow_html=True)
            active_item = None
            top_pred = None
            info = None
        else:
            items = build_items(files)
            if not items:
                with col_result:
                    st.markdown(ui.render_standby_banner(), unsafe_allow_html=True)
                active_item = None
                top_pred = None
                info = None
            else:
                with st.spinner("Executing simultaneous inference across 4 deep learning architectures..."):
                    for it in items:
                        it["all_preds"] = {}
                        for m in models:
                            try:
                                it["all_preds"][m] = predict_cached(m, it["bytes"])
                            except Exception as exc:
                                st.error(f"Inference error on model '{m}': {exc}")
                                st.stop()

                        consensus_top, agree_cnt, avg_conf = compute_consensus(it["all_preds"])
                        it["consensus_top"] = consensus_top
                        it["agree_cnt"] = agree_cnt
                        it["avg_conf"] = avg_conf

                        record_history(it, consensus_top, model_label=f"4-Model Ensemble ({agree_cnt}/{len(models)} agree)")

                if len(items) > 1:
                    with batch_slot.container():
                        st.caption(f"**Batch Overview ({len(items)} images)**")
                        pick_idx = st.selectbox("Select image to view:", range(len(items)),
                                                format_func=lambda i: f"{i + 1}. {items[i]['name']}")
                        active_item = items[pick_idx]
                else:
                    active_item = items[0]

                top_pred = active_item["consensus_top"]
                info = get_info(top_pred["class_name"])

                # Render Leaf Scanner Preview & Telemetry inside the Scanner Card
                with preview_slot.container():
                    w, h = active_item["image"].size
                    notes = ui.quality_notes(active_item["image"])
                    status_illum = "Diffused" if not any("dark" in n or "glare" in n for n in notes) else "Sub-optimal"
                    status_focus = "Crisp" if not any("blurry" in n for n in notes) else "Blurry"

                    thumb_data_uri = img_to_data_uri(active_item["image"])

                    st.markdown(
                        f"""<div class="scanner-split-preview">
                            <div class="scanner-thumb-box">
                                <img src="{thumb_data_uri}" alt="Leaf Preview" />
                            </div>
                            <div class="scanner-meta-col">
                                <div class="scanner-filename">📄 {ui.esc(active_item['name'])}</div>
                                <div class="telemetry-tag-row">
                                    <span class="tele-tag">⛶ {w} × {h} px</span>
                                    <span class="tele-tag">☀️ {status_illum}</span>
                                    <span class="tele-tag">🎯 {status_focus}</span>
                                </div>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                    # Clear/Remove Image Action Button
                    if st.button("🗑️ Remove Image / Re-upload", key="btn_remove_leaf", use_container_width=True):
                        ss["sample_image"] = None
                        st.rerun()

                # Render Right Column Diagnosis Result Card
                with col_result:
                    st.markdown(
                        ui.consensus_banner_html(top_pred, active_item["agree_cnt"], len(models), active_item["avg_conf"], info),
                        unsafe_allow_html=True,
                    )

        # ---------------- Section 2: 4-Architecture Model Matrix (Single Horizontal Row)
        if files and active_item and top_pred:
            st.markdown(ui.render_four_model_matrix(active_item["all_preds"], top_pred), unsafe_allow_html=True)

            # ---------------- Section 3: Agronomic Treatment Protocol
            leaf_thumb_data = img_to_data_uri(active_item["image"])
            st.markdown(ui.render_agronomic_protocol_card(top_pred, info, leaf_thumb_uri=leaf_thumb_data), unsafe_allow_html=True)

            # ---------------- Actions: Report & Audit Log Export
            c1, c2 = st.columns([1.2, 1])
            with c1:
                report_md = ui.report_markdown(
                    active_item["name"],
                    "4-Model Multi-Architecture Ensemble",
                    [top_pred],
                    info,
                    all_model_preds=active_item["all_preds"],
                )
                st.download_button(
                    "⬇️ Export Clinical Diagnosis Report (Markdown)",
                    key=f"dl_rep_{active_item['id']}",
                    data=report_md,
                    file_name=f"plant_pathology_report_{active_item['id']}.md",
                    mime="text/markdown",
                )
            with c2:
                st.caption("Protocol generated in accordance with agricultural extension plant disease guidelines.")

    # Session Diagnosis Audit Log
    if ss["history"]:
        st.divider()
        with st.expander(f"🕑 Session Clinical Audit Log ({len(ss['history'])} diagnoses)"):
            hist_df = pd.DataFrame(ss["history"])
            st.dataframe(hist_df, hide_index=True)
            h1, h2 = st.columns(2)
            h1.download_button("⬇️ Export Audit Log (CSV)", data=hist_df.to_csv(index=False),
                               file_name="diagnosis_history.csv", mime="text/csv")
            if h2.button("Clear Audit Log"):
                ss["history"] = []
                ss["history_keys"] = set()
                st.rerun()


# ==================================================================
# TAB 2: BENCHMARKS & MODEL PERFORMANCE
# ==================================================================
with tab_benchmark:
    table_path = COMPARISON_DIR / "metrics_comparison.csv"
    if not table_path.exists():
        st.info("No comparative metrics found. Run `python -m src.evaluate` to generate benchmark table.")
    else:
        df = pd.read_csv(table_path)
        m1, m2, m3, m4 = st.columns(4)
        top_acc_row = df.loc[df["Test accuracy"].idxmax()]
        fast_row = df.loc[df["Inference (ms/img)"].idxmin()]
        comp_row = df.loc[df["Size (MB)"].idxmin()]
        vit_row = df[df["Model"].str.contains("Vision Transformer", case=False)]

        m1.metric("Highest Test Accuracy", top_acc_row["Model"].split(" (")[0], f"{top_acc_row['Test accuracy'] * 100:.2f}%")
        m2.metric("Lowest Latency", fast_row["Model"].split(" (")[0], f"{fast_row['Inference (ms/img)']:.1f} ms/img")
        m3.metric("Smallest Footprint", comp_row["Model"].split(" (")[0], f"{comp_row['Size (MB)']:.1f} MB")
        if not vit_row.empty:
            m4.metric("ViT Benchmark", "Self-Attention", f"{vit_row.iloc[0]['Test accuracy'] * 100:.2f}%")
        else:
            m4.metric("Models Evaluated", f"{len(df)} Architectures", "Equal Weight")

        st.dataframe(df, hide_index=True)

        num_cols = list(df.select_dtypes("number").columns)
        metric_choice = st.selectbox("Compare models across metric:", num_cols, index=0)
        st.bar_chart(df.set_index("Model")[metric_choice])

        col_p1, col_p2 = st.columns(2)
        m_img = COMPARISON_DIR / "metrics_comparison.png"
        tc_img = COMPARISON_DIR / "training_curves_comparison.png"
        if m_img.exists():
            col_p1.image(str(m_img), caption="Comparative Test Metrics")
        if tc_img.exists():
            col_p2.image(str(tc_img), caption="Validation Curves Comparison")

        trained = [n for n in MODEL_NAMES if (OUTPUTS_DIR / n / "metrics.json").exists()]
        if trained:
            st.divider()
            st.subheader("🔬 Architecture-Specific Deep Dive")
            pick_m = st.selectbox("Inspect Architecture Details", trained,
                                  format_func=lambda n: DISPLAY_NAMES.get(n, n), key="perf_model")
            cm_kind = st.radio("Confusion Matrix Mode", ["Row-normalised", "Raw Counts"], horizontal=True)
            col_crv, col_cm = st.columns(2)
            crv_path = OUTPUTS_DIR / pick_m / "training_curves.png"
            cm_path = OUTPUTS_DIR / pick_m / ("confusion_matrix_normalized.png" if cm_kind == "Row-normalised" else "confusion_matrix.png")
            if crv_path.exists():
                col_crv.image(str(crv_path), caption=f"{DISPLAY_NAMES.get(pick_m, pick_m)} Training Curves")
            if cm_path.exists():
                col_cm.image(str(cm_path), caption=f"{DISPLAY_NAMES.get(pick_m, pick_m)} Confusion Matrix")

            rep_path = OUTPUTS_DIR / pick_m / "classification_report.csv"
            if rep_path.exists():
                rep = pd.read_csv(rep_path, index_col=0)
                rep = rep.loc[rep.index.isin(class_names)] if class_names else rep
                n_classes = st.slider("Show weakest N classes by F1-Score:", 3, max(len(rep), 3), min(8, max(len(rep), 3)))
                weakest = rep["f1-score"].sort_values().head(n_classes)
                weakest.index = [pretty_label(i) for i in weakest.index]
                st.bar_chart(weakest)
                with st.expander("Full Per-Class Classification Report Table"):
                    st.dataframe(rep)


# ==================================================================
# TAB 3: DISEASE KNOWLEDGE BASE
# ==================================================================
with tab_encyclopedia:
    if not class_names:
        st.info("The disease encyclopedia requires `outputs/class_names.json`.")
    else:
        entries = []
        for cls in class_names:
            plant, cond, _ = parse_label(cls)
            entries.append((cls, plant, cond, get_info(cls)))
        c1, c2, c3 = st.columns([2, 1.5, 1.5])
        query = c1.text_input("🔎 Search Pathology Database", placeholder="e.g. tomato, blight, rust, scab, healthy")
        plant_sel = c2.multiselect("Filter by Crop Species", sorted({e[1] for e in entries}))
        type_sel = c3.multiselect("Filter by Pathogen Classification", list(DISEASE_TYPES))
        q = query.strip().lower()
        shown = [e for e in entries
                 if (not q or q in f"{e[1]} {e[2]} {e[3]['name']} {e[3]['type']} {e[3]['summary']}".lower())
                 and (not plant_sel or e[1] in plant_sel) and (not type_sel or e[3]["type"] in type_sel)]
        st.caption(f"Displaying **{len(shown)} of {len(entries)}** plant disease conditions.")
        for cls, plant, cond, info_e in shown:
            with st.expander(f"{'✅' if info_e['healthy'] else '🦠'} {plant} — {cond}"):
                st.markdown(ui.type_chip(info_e["type"]) + ("" if info_e["healthy"] else ui.severity_chip(info_e["severity"])), unsafe_allow_html=True)
                st.write(info_e["summary"])
                if info_e.get("agent"):
                    st.markdown(ui.kv("Pathogen", info_e["agent"]), unsafe_allow_html=True)
                if info_e.get("symptoms"):
                    st.markdown("**Field Symptoms**")
                    bullets(info_e["symptoms"])
                st.markdown("**Recommended Management**" if not info_e["healthy"] else "**Crop Health Maintenance**")
                bullets(info_e["immediate"])
                if info_e.get("prevention"):
                    st.markdown("**Preventative Sanitation**")
                    bullets(info_e["prevention"])


# ==================================================================
# TAB 4: SYSTEM ARCHITECTURE & ABOUT
# ==================================================================
with tab_specs:
    st.markdown(
        """
### AgroVision Multi-Architecture Engineering Specification

This system integrates **four distinct neural paradigms** to achieve robust consensus visual crop diagnosis:

1. **Vision Transformer (ViT from scratch)**
   - *Paradigm*: Self-Attention Transformer without convolutional inductive bias.
   - *Pipeline*: 16×16 non-overlapping patches -> Linear Projection ($d=128$) -> 1D Learnable Positional Embeddings -> 4-layer Multi-Head Attention Encoder ($H=4$, MLP $d=256$) -> Global Sequence Average Pooling.
   - *Key Advantage*: Captures global leaf texture patterns and long-distance spatial dependencies immediately from layer 1. Extremely lightweight (8.4 MB) with 0.8 ms inference latency.

2. **ResNet-50 (Transfer Learning)**
   - *Paradigm*: Deep Residual Network with Identity Skip Connections.
   - *Pipeline*: Pretrained ImageNet weights -> 2-phase transfer training (frozen backbone head warmup + fine-tuning top 30 layers).
   - *Key Advantage*: Deep hierarchical representations (23.7M parameters) with top test accuracy (97.43%).

3. **MobileNet-V2 (Transfer Learning)**
   - *Paradigm*: Inverted Residuals with Linear Bottlenecks & Depthwise Separable Convolutions.
   - *Pipeline*: Lightweight edge backbone with 2.3M parameters.
   - *Key Advantage*: Minimal computational memory footprint (22.3 MB) and fast 2.1 ms inference, optimized for edge and mobile field deployment.

4. **Custom 4-Block CNN (Trained from Scratch)**
   - *Paradigm*: Classical Deep Convolutional Baseline.
   - *Pipeline*: 4 Conv2D-BatchNorm-ReLU-MaxPool stages (32 -> 64 -> 128 -> 256 filters) with Dropout (0.25, 0.4).
   - *Key Advantage*: Pure convolutional benchmark trained without external ImageNet pretraining (90.22% test accuracy).
"""
    )
    st.warning(DISCLAIMER)
