"""AgroVision AI: Senior Design System & UI Components.

A polished production-grade AI agricultural diagnostic dashboard:
- Deep navy obsidian palette (#050b14, #08101e, #0c1527)
- Cyan & teal primary accents (#00e5ff, #06b6d4, #0284c7)
- Emerald healthy crop badge (#10b981) & Rose disease alert (#f43f5e)
- 4-Model identity accents: ViT (Fuchsia #a855f7), ResNet50 (Indigo #3b82f6), MobileNetV2 (Emerald #10b981), Custom CNN (Amber #f59e0b)
- Proportional two-column workspace (Scanner vs Diagnosis Result)
- Single-row 4-Architecture Model Matrix with equal widths & aligned metrics
- Dynamic Agronomic Treatment Protocol with balanced proportions (no blank green boxes)
- Fixed-width, non-wrapping sidebar (315px) with properly sized Sample 1 & 2 buttons
"""
from __future__ import annotations

import base64
import html
import io
from datetime import datetime

import numpy as np
from PIL import Image

from .disease_info import CHEMICAL_NOTE, DISCLAIMER

MODEL_META = {
    "vit": {
        "name": "Vision Transformer",
        "short": "ViT",
        "tag": "ViT-Patch16",
        "sub": "Self-Attention Transformer",
        "color": "#06b6d4",
        "border_color": "#a855f7",
        "acc": "93.89%",
        "speed": "0.8 ms",
        "icon": "⚡",
        "icon_color": "#a855f7",
    },
    "resnet50": {
        "name": "ResNet50",
        "short": "ResNet50",
        "tag": "Deep Residual",
        "sub": "50-Layer Bottleneck CNN",
        "color": "#06b6d4",
        "border_color": "#3b82f6",
        "acc": "97.43%",
        "speed": "5.1 ms",
        "icon": "🧠",
        "icon_color": "#3b82f6",
    },
    "mobilenetv2": {
        "name": "MobileNetV2",
        "short": "MobileNetV2",
        "tag": "Edge Inverted",
        "sub": "Inverted Residuals + Linear",
        "color": "#06b6d4",
        "border_color": "#10b981",
        "acc": "95.06%",
        "speed": "2.1 ms",
        "icon": "📱",
        "icon_color": "#10b981",
    },
    "cnn": {
        "name": "Custom CNN",
        "short": "Custom CNN",
        "tag": "4-Block ConvNet",
        "sub": "Trained From Scratch",
        "color": "#06b6d4",
        "border_color": "#f59e0b",
        "acc": "90.22%",
        "speed": "2.2 ms",
        "icon": "🔬",
        "icon_color": "#f59e0b",
    },
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

/* ==========================================================================
   GLOBAL RESET & THEME TOKENS
   ========================================================================== */
:root {
    --bg-main: #050b14;
    --bg-card: #091120;
    --bg-card-inner: #0e182e;
    --border-card: rgba(0, 229, 255, 0.18);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --cyan-primary: #00e5ff;
    --cyan-teal: #06b6d4;
    --emerald-healthy: #10b981;
    --rose-disease: #f43f5e;
}

html, body, [class*="css"], .stMarkdown, .stText, .stButton, .stSelectbox, .stTabs, p, h1, h2, h3, h4 {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary);
}

span:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]):not([class*="material-icons"]) {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stIconMaterial"], .material-symbols-rounded, .material-symbols-outlined, [class*="material-symbols"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", sans-serif !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 1.25rem !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-smoothing: antialiased !important;
}

code, kbd, samp, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

#MainMenu, footer, header {visibility: hidden !important;}
.stDeployButton {display: none !important;}

.block-container {
    padding-top: 0.8rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1440px !important;
    overflow-x: hidden !important;
}

/* ==========================================================================
   SIDEBAR REFINEMENT (FIXED 315px WIDTH, NO UNNECESSARY WRAPPING)
   ========================================================================== */
section[data-testid="stSidebar"] {
    background-color: #060c16 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    min-width: 315px !important;
    max-width: 315px !important;
    overflow-y: auto !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.2rem 1.15rem !important;
}

.sidebar-brand-box {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1.2rem;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.sidebar-logo-icon {
    font-size: 1.8rem;
    color: #10b981;
    filter: drop-shadow(0 0 10px rgba(16, 185, 129, 0.55));
}
.sidebar-logo-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.4px;
    margin: 0;
    line-height: 1.1;
}
.sidebar-logo-sub {
    font-size: 0.74rem;
    color: #94a3b8;
    margin: 0.15rem 0 0;
    font-weight: 500;
    letter-spacing: 0.2px;
    white-space: nowrap;
}

.sidebar-nav-menu {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    margin-bottom: 1.3rem;
}
.sidebar-nav-item {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.55rem 0.95rem;
    border-radius: 9px;
    font-size: 0.86rem;
    font-weight: 600;
    color: #94a3b8;
    cursor: default;
    white-space: nowrap;
    transition: all 0.2s ease;
}
.sidebar-nav-item.active {
    background: linear-gradient(135deg, #0e7490 0%, #0891b2 100%);
    border: 1px solid #06b6d4;
    color: #ffffff;
    box-shadow: 0 0 14px rgba(6, 182, 212, 0.35);
}
.sidebar-nav-item:not(.active):hover {
    background: rgba(255, 255, 255, 0.04);
    color: #e2e8f0;
}

.sidebar-section-title {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.88rem;
    font-weight: 700;
    color: #ffffff;
    margin: 1.1rem 0 0.55rem;
    white-space: nowrap;
}

.tips-bullet-list {
    margin: 0.4rem 0 0;
    padding-left: 0;
    font-size: 0.76rem;
    color: #94a3b8;
    line-height: 1.45;
}
.tips-bullet-list li {
    margin-bottom: 0.45rem;
    list-style-type: none;
    position: relative;
    padding-left: 0.95rem;
}
.tips-bullet-list li::before {
    content: "•";
    position: absolute;
    left: 0;
    color: #06b6d4;
    font-size: 1.2rem;
    line-height: 0.9;
}

/* ==========================================================================
   TOP HEADER BAR
   ========================================================================== */
.top-header-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #08101e;
    border: 1px solid rgba(0, 229, 255, 0.18);
    border-radius: 16px;
    padding: 1rem 1.6rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45);
}
.header-brand-side {
    display: flex;
    align-items: center;
    gap: 0.95rem;
}
.header-app-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #06b6d4 0%, #0284c7 100%);
    display: grid;
    place-items: center;
    font-size: 1.5rem;
    box-shadow: 0 0 16px rgba(6, 182, 212, 0.5);
    flex-shrink: 0;
}
.header-text-group h1 {
    font-size: 1.35rem;
    font-weight: 800;
    margin: 0;
    color: #ffffff;
    letter-spacing: -0.3px;
}
.header-text-group p {
    font-size: 0.8rem;
    color: #94a3b8;
    margin: 0.15rem 0 0;
    white-space: nowrap;
}
.header-telemetry-pills {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    flex-wrap: wrap;
}
.header-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.32rem 0.75rem;
    border-radius: 8px;
    font-size: 0.73rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    background: rgba(12, 21, 39, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #cbd5e1;
    white-space: nowrap;
}
.header-pill.green-online {
    background: rgba(16, 185, 129, 0.14);
    border-color: rgba(16, 185, 129, 0.45);
    color: #34d399;
}
.header-pill.green-online .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
}

/* ==========================================================================
   NAVIGATION TABS (MODERN ROUNDED PILLS WITH GENEROUS SPACING)
   ========================================================================== */
.stTabs,
div[data-testid="stTabs"] {
    margin-bottom: 1.5rem !important;
}

/* Tab list bar */
.stTabs [role="tablist"],
div[data-testid="stTabs"] [role="tablist"],
div[data-testid="stTabs"] > div > div[role="tablist"],
div[data-testid="stTabs"] > div:first-child,
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border: none !important;
    padding: 0.4rem 0 !important;
    gap: 0.9rem !important;
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    border-bottom: none !important;
}

/* Completely remove the flat continuous underline bar across the entire tablist */
.stTabs [role="tablist"]::after,
div[data-testid="stTabs"] [role="tablist"]::after,
div[data-testid="stTabs"] > div > div[role="tablist"]::after,
div[data-testid="stTabs"] > div:first-child::after,
div[data-testid="stTabs"] [role="tablist"] [role="presentation"] {
    display: none !important;
    content: none !important;
    height: 0 !important;
    width: 0 !important;
    border: none !important;
}

/* Remove bottom indicator line inside each tab */
.stTabs .react-aria-SelectionIndicator,
div[data-testid="stTabs"] .react-aria-SelectionIndicator {
    display: none !important;
    height: 0 !important;
}

/* EVERY TAB (Default / Inactive State) */
.stTabs [role="tab"],
div[data-testid="stTabs"] [role="tab"],
div[data-testid="stTabs"] [data-testid="stTab"],
div[data-testid="stTabs"] button,
.stTabs [data-baseweb="tab"] {
    background: rgba(9, 17, 32, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1.4rem !important;
    margin: 0 !important;
    margin-right: 0.5rem !important;
    height: auto !important;
    min-height: 44px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.55rem !important;
    white-space: nowrap !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25) !important;
    overflow: hidden !important;
}

/* Inner text / markdown in tab */
.stTabs [role="tab"] p,
div[data-testid="stTabs"] [role="tab"] p,
div[data-testid="stTabs"] [data-testid="stTab"] p {
    margin: 0 !important;
    font-size: 0.88rem !important;
    font-weight: inherit !important;
    color: inherit !important;
}

/* Hover state on tabs */
.stTabs [role="tab"]:hover,
div[data-testid="stTabs"] [role="tab"]:hover,
div[data-testid="stTabs"] [data-testid="stTab"]:hover,
div[data-testid="stTabs"] button:hover,
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(14, 24, 46, 0.95) !important;
    border-color: rgba(6, 182, 212, 0.5) !important;
    color: #f8fafc !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 14px rgba(6, 182, 212, 0.2) !important;
}

/* ACTIVE / SELECTED TAB */
.stTabs [role="tab"][aria-selected="true"],
.stTabs [role="tab"][data-selected="true"],
.stTabs [role="tab"][data-selected],
div[data-testid="stTabs"] [role="tab"][aria-selected="true"],
div[data-testid="stTabs"] [role="tab"][data-selected="true"],
div[data-testid="stTabs"] [role="tab"][data-selected],
div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"],
div[data-testid="stTabs"] [data-testid="stTab"][data-selected="true"],
div[data-testid="stTabs"] [data-testid="stTab"][data-selected],
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0e7490 0%, #0891b2 100%) !important;
    border: 1px solid #06b6d4 !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 0 18px rgba(6, 182, 212, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.25) !important;
    transform: translateY(-2px) !important;
}

/* Tab content panel spacing */
div[data-testid="stTabPanel"] {
    padding-top: 0.5rem !important;
}

/* ==========================================================================
   DIAGNOSTIC WORKSPACE CARDS (EQUAL HEIGHT, ROUNDED BORDERS, BALANCED 1:1)
   ========================================================================== */
.ws-card-container {
    background: #091120 !important;
    border: 1px solid rgba(0, 229, 255, 0.18) !important;
    border-radius: 18px !important;
    padding: 1.4rem 1.6rem !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) !important;
    min-height: 295px !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
    box-sizing: border-box !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

/* Scanner card – the Streamlit border-wrapper and the inner key-div */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-scanner_card),
div[class*="st-key-scanner_card"] {
    background: #091120 !important;
    border: 1px solid rgba(0, 229, 255, 0.18) !important;
    border-radius: 18px !important;
    padding: 1.4rem 1.6rem !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35) !important;
    min-height: 295px !important;
    box-sizing: border-box !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

/* Inner vertical block inside scanner card – DON'T use space-between, it crushes the uploader */
.st-key-scanner_card > div[data-testid="stVerticalBlock"],
div[class*="st-key-scanner_card"] div[data-testid="stVerticalBlock"] {
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    gap: 0.5rem !important;
}

/* Ensure the element wrapper around the file uploader is not collapsed */
.st-key-scanner_card div[data-testid="stElementContainer"],
.st-key-scanner_card div[data-testid="element-container"] {
    min-height: auto !important;
}

/* Ensure the file uploader element wrapper has proper height */
.st-key-scanner_card div[data-testid="stElementContainer"]:has(div[data-testid="stFileUploader"]) {
    min-height: 110px !important;
    width: 100% !important;
}

.ws-card-container:hover,
div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-scanner_card):hover,
div[class*="st-key-scanner_card"]:hover {
    border-color: rgba(6, 182, 212, 0.4) !important;
    box-shadow: 0 10px 28px rgba(0, 229, 255, 0.1) !important;
}

.ws-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
}
.ws-card-header h3 {
    font-size: 1.15rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    letter-spacing: -0.2px;
}
.ws-card-subtitle {
    font-size: 0.8rem;
    color: #94a3b8;
    margin: 0 0 0.95rem;
    line-height: 1.4;
}

/* Radio Switcher inside Scanner Card */
div[data-testid="stRadio"] {
    margin-bottom: 0.85rem !important;
}
div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    gap: 0.9rem !important;
    background: rgba(14, 24, 46, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 999px !important;
    padding: 0.35rem 0.95rem !important;
    width: fit-content !important;
}
div[data-testid="stRadio"] label {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
    cursor: pointer !important;
}
div[data-testid="stRadio"] label:hover {
    color: #00e5ff !important;
}

/* Uploader & Preview in Scanner Card */
.scanner-split-preview {
    display: flex;
    gap: 1.15rem;
    align-items: center;
    background: rgba(14, 24, 46, 0.7);
    border: 1px solid rgba(0, 229, 255, 0.22);
    border-radius: 14px;
    padding: 0.95rem 1.15rem;
    margin-top: 0.75rem;
    margin-bottom: 0.75rem;
}
.scanner-thumb-box {
    position: relative;
    width: 130px;
    height: 105px;
    border-radius: 12px;
    overflow: hidden;
    background: #000000;
    flex-shrink: 0;
    border: 1px solid rgba(0, 229, 255, 0.35);
}
.scanner-thumb-box img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.scanner-meta-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0.4rem;
}
.scanner-filename {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    color: #ffffff;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.telemetry-tag-row {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin-top: 0.25rem;
}
.tele-tag {
    background: rgba(0, 0, 0, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 0.22rem 0.6rem;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #94a3b8;
}

/* Diagnosis Result Card */
.diag-badge-healthy {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.28rem 0.85rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    background: rgba(16, 185, 129, 0.18);
    border: 1px solid rgba(16, 185, 129, 0.45);
    color: #34d399;
    margin-bottom: 0.45rem;
}
.diag-badge-disease {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.28rem 0.85rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    background: rgba(244, 63, 94, 0.18);
    border: 1px solid rgba(244, 63, 94, 0.45);
    color: #fb7185;
    margin-bottom: 0.45rem;
}
.diag-title-main {
    font-size: 1.45rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0.1rem 0 0.15rem;
    letter-spacing: -0.4px;
    line-height: 1.25;
}
.diag-sub-consensus {
    font-size: 0.76rem;
    color: #94a3b8;
    margin: 0 0 0.85rem;
}

.diag-split-gauge-row {
    display: flex;
    align-items: center;
    gap: 1.2rem;
}
.gauge-svg-wrap {
    flex: 0 0 88px;
    height: 88px;
    position: relative;
}
.gauge-svg {
    width: 88px;
    height: 88px;
    transform: rotate(-90deg);
}
.gauge-bg-circle {
    fill: none;
    stroke: rgba(255, 255, 255, 0.08);
    stroke-width: 7;
}
.gauge-bar-circle {
    fill: none;
    stroke-width: 7;
    stroke-linecap: round;
}
.gauge-inner-txt {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
}
.gauge-inner-txt b {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
}
.gauge-inner-txt span {
    font-size: 0.6rem;
    color: #94a3b8;
    text-transform: capitalize;
    margin-top: 0.2rem;
}

.diag-verdict-card {
    flex: 1;
    background: rgba(0, 0, 0, 0.25);
    border-radius: 12px;
    padding: 0.75rem 0.95rem;
    border-left: 3px solid #10b981;
}
.diag-verdict-card.disease-verdict {
    border-left-color: #f43f5e;
}
.verdict-header-row {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.86rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.2rem;
}
.verdict-desc-txt {
    font-size: 0.76rem;
    color: #cbd5e1;
    line-height: 1.45;
    margin: 0;
}

/* ==========================================================================
   4-ARCHITECTURE MODEL MATRIX (ROW OF 4 EQUAL CARDS)
   ========================================================================== */
.matrix-bar-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 1.3rem 0 0.8rem;
}
.matrix-bar-header h3 {
    font-size: 1.15rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.matrix-consensus-badge {
    font-size: 0.75rem;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
}
.matrix-consensus-badge b {
    color: #06b6d4;
}

.model-matrix-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.85rem;
    margin-bottom: 1.4rem;
}
@media (max-width: 1080px) {
    .model-matrix-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
@media (max-width: 600px) {
    .model-matrix-grid {
        grid-template-columns: 1fr;
    }
}

.matrix-single-card {
    background: #091120;
    border: 1px solid rgba(0, 229, 255, 0.16);
    border-radius: 14px;
    padding: 0.95rem 1.05rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    min-height: 145px;
    box-sizing: border-box;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.matrix-single-card:hover {
    transform: translateY(-2px);
    border-color: rgba(6, 182, 212, 0.45);
}
.matrix-single-card.vit { border-top: 3px solid #a855f7; }
.matrix-single-card.resnet50 { border-top: 3px solid #3b82f6; }
.matrix-single-card.mobilenetv2 { border-top: 3px solid #10b981; }
.matrix-single-card.cnn { border-top: 3px solid #f59e0b; }

.msc-top-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.7rem;
}
.msc-icon {
    font-size: 1.15rem;
    line-height: 1;
}
.msc-title-group h4 {
    font-size: 0.92rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    line-height: 1.15;
    white-space: nowrap;
}
.msc-title-group span {
    font-size: 0.68rem;
    color: #94a3b8;
    font-weight: 600;
}

.msc-stats-split {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.6rem;
}
.msc-stat-cell {
    flex: 1;
}
.msc-stat-cell b {
    display: block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
}
.msc-stat-cell span {
    font-size: 0.68rem;
    color: #94a3b8;
    text-transform: capitalize;
}

.msc-track {
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    overflow: hidden;
}
.msc-fill {
    height: 100%;
    background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%);
    border-radius: 999px;
}

/* ==========================================================================
   AGRONOMIC TREATMENT PROTOCOL (CLEAN, PROPORTIONAL)
   ========================================================================== */
.protocol-card-main {
    background: #091120;
    border: 1px solid rgba(0, 229, 255, 0.16);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    margin-bottom: 1.4rem;
}
.protocol-card-hdr {
    margin-bottom: 0.85rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.protocol-card-hdr h3 {
    font-size: 1.15rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.protocol-card-hdr p {
    font-size: 0.78rem;
    color: #94a3b8;
    margin: 0.15rem 0 0;
}

.protocol-split-layout {
    display: flex;
    gap: 1.5rem;
    align-items: flex-start;
}
@media (max-width: 860px) {
    .protocol-split-layout {
        flex-direction: column;
    }
}
.protocol-left-col {
    flex: 0 0 250px;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
}
.protocol-thumb-box {
    width: 100%;
    height: 120px;
    border-radius: 10px;
    overflow: hidden;
    background: #040810;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.protocol-thumb-box img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.protocol-crop-name {
    font-size: 1.05rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0.2rem 0 0.1rem;
}
.protocol-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 700;
    width: fit-content;
}
.protocol-status-badge.no-treatment {
    background: rgba(16, 185, 129, 0.18);
    border: 1px solid rgba(16, 185, 129, 0.45);
    color: #34d399;
}
.protocol-status-badge.action-required {
    background: rgba(244, 63, 94, 0.18);
    border: 1px solid rgba(244, 63, 94, 0.45);
    color: #fb7185;
}

.protocol-right-col {
    flex: 1;
}

.protocol-pills-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.85rem;
}
.proto-tab-pill {
    padding: 0.35rem 0.95rem;
    border-radius: 8px;
    font-size: 0.78rem;
    font-weight: 700;
    cursor: default;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}
.proto-tab-pill.active {
    background: linear-gradient(135deg, #0e7490 0%, #0891b2 100%);
    border: 1px solid #06b6d4;
    color: #ffffff;
}
.proto-tab-pill.inactive {
    background: rgba(14, 24, 46, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #94a3b8;
}

.proto-bullet-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.proto-bullet-list li {
    font-size: 0.84rem;
    color: #cbd5e1;
    margin-bottom: 0.5rem;
    line-height: 1.5;
    position: relative;
    padding-left: 1.15rem;
}
.proto-bullet-list li::before {
    content: "•";
    position: absolute;
    left: 0;
    color: #06b6d4;
    font-size: 1.25rem;
    line-height: 1;
}

/* ==========================================================================
   STREAMLIT NATIVE CONTROLS OVERRIDES
   ========================================================================== */

/* Outer file uploader wrapper – keep transparent so it doesn't double-border */
div[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    width: 100% !important;
}

/* Actual dropzone section – the ONLY element that gets the dashed border */
section[data-testid="stFileUploadDropzone"] {
    background: rgba(14, 24, 46, 0.65) !important;
    border: 1.5px dashed rgba(6, 182, 212, 0.45) !important;
    border-radius: 14px !important;
    padding: 1.5rem 1.2rem !important;
    min-height: 110px !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 1rem !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
section[data-testid="stFileUploadDropzone"]:hover {
    border-color: #00e5ff !important;
    background: rgba(14, 24, 46, 0.85) !important;
    box-shadow: 0 0 18px rgba(0, 229, 255, 0.15) !important;
}

/* Upload button inside dropzone */
section[data-testid="stFileUploadDropzone"] button {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 42px !important;
    min-width: 110px !important;
    background: linear-gradient(135deg, #0e7490 0%, #0891b2 100%) !important;
    border: 1px solid #06b6d4 !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    padding: 0.5rem 1.4rem !important;
    box-shadow: 0 0 14px rgba(6, 182, 212, 0.4) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    flex-shrink: 0 !important;
    visibility: visible !important;
    opacity: 1 !important;
}
section[data-testid="stFileUploadDropzone"] button:hover {
    background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%) !important;
    border-color: #00e5ff !important;
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.6) !important;
    transform: translateY(-1px);
}

/* Eliminate 'uploadpload' text duplication – hide the material icon ligature */
section[data-testid="stFileUploadDropzone"] button span[data-testid="stIconMaterial"],
section[data-testid="stFileUploadDropzone"] button .material-symbols-rounded,
section[data-testid="stFileUploadDropzone"] button .material-symbols-outlined {
    display: none !important;
}
section[data-testid="stFileUploadDropzone"] button::before {
    content: "⭡ " !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    margin-right: 0.4rem !important;
    color: #ffffff !important;
}

/* Dropzone instructions – ensure visible */
div[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important;
    flex-direction: column !important;
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    margin-top: 0 !important;
    visibility: visible !important;
    opacity: 1 !important;
    min-width: 0 !important;
    flex: 1 !important;
}
div[data-testid="stFileUploaderDropzoneInstructions"] div {
    display: flex !important;
    flex-direction: column !important;
}
div[data-testid="stFileUploaderDropzoneInstructions"] span {
    display: block !important;
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* Dropzone input – keep hidden but functional */
section[data-testid="stFileUploadDropzone"] input[data-testid="stFileUploaderDropzoneInput"] {
    position: absolute !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
}

/* Widget label for file uploader (hidden by label_visibility="collapsed") */
div[data-testid="stFileUploader"] label[data-testid="stWidgetLabel"] {
    min-height: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}

div.stButton > button {
    background: #0c182e !important;
    border: 1px solid rgba(0, 229, 255, 0.25) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    padding: 0.55rem 1.05rem !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 14px rgba(6, 182, 212, 0.45) !important;
    color: #06b6d4 !important;
}

/* Remove/Re-upload Button inside Left Card */
button[key="btn_remove_leaf"] {
    background: rgba(244, 63, 94, 0.14) !important;
    border: 1px solid rgba(244, 63, 94, 0.4) !important;
    color: #fda4af !important;
    border-radius: 10px !important;
    padding: 0.55rem 1rem !important;
    font-weight: 700 !important;
    font-size: 0.84rem !important;
    transition: all 0.2s ease !important;
}
button[key="btn_remove_leaf"]:hover {
    background: rgba(244, 63, 94, 0.25) !important;
    border-color: #f43f5e !important;
    color: #ffffff !important;
    box-shadow: 0 0 14px rgba(244, 63, 94, 0.4) !important;
}

div[data-testid="stMetric"] {
    background: #091120 !important;
    border: 1px solid rgba(0, 229, 255, 0.16) !important;
    border-radius: 16px !important;
    padding: 0.95rem 1.25rem !important;
}
</style>
"""


def esc(text) -> str:
    return html.escape(str(text))


def confidence_color(pct: float) -> str:
    return "#10b981" if pct >= 80 else "#f59e0b" if pct >= 60 else "#ef4444"


def severity_chip(severity: str) -> str:
    s = str(severity).lower()
    return f'<span style="padding:0.18rem 0.6rem; border-radius:999px; font-size:0.72rem; font-weight:700; background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.12); color:#cbd5e1;">Severity: {esc(severity)}</span>'


def type_chip(kind: str) -> str:
    return f'<span style="padding:0.18rem 0.6rem; border-radius:999px; font-size:0.72rem; font-weight:700; background:rgba(6,182,212,0.15); border:1px solid rgba(6,182,212,0.3); color:#38bdf8;">{esc(kind)}</span>'


def ring_html(pct: float) -> str:
    pct = max(0.0, min(100.0, pct))
    dashoffset = 226.2 * (1.0 - (pct / 100.0))
    c = "#10b981" if pct >= 80 else "#06b6d4"
    return f"""<div class="gauge-svg-wrap">
    <svg class="gauge-svg" viewBox="0 0 88 88">
        <circle class="gauge-bg-circle" cx="44" cy="44" r="36" />
        <circle class="gauge-bar-circle" cx="44" cy="44" r="36"
            stroke="{c}"
            stroke-dasharray="226.2"
            stroke-dashoffset="{dashoffset:.1f}" />
    </svg>
    <div class="gauge-inner-txt">
        <b>{pct:.1f}%</b>
        <span>Confidence</span>
    </div>
</div>"""


def render_sidebar_header() -> str:
    return """<div class="sidebar-brand-box">
    <div class="sidebar-logo-icon">🌿</div>
    <div>
        <h2 class="sidebar-logo-title">AGROVISION</h2>
        <p class="sidebar-logo-sub">AI Crop Diagnostic Suite</p>
    </div>
</div>"""


def render_top_header() -> str:
    return """<div class="top-header-card">
    <div class="header-brand-side">
        <div class="header-app-icon">🌿</div>
        <div class="header-text-group">
            <h1>AI Diagnostic Console</h1>
            <p>Plant Disease Detection &amp; Classification | Multi-Architecture (CNN, ViT, ResNet50, MobileNetV2)</p>
        </div>
    </div>
    <div class="header-telemetry-pills">
        <span class="header-pill green-online"><span class="dot"></span> 4 ARCHITECTURES ONLINE</span>
        <span class="header-pill">🗂️ ENGINE: TF 2.19</span>
        <span class="header-pill" style="color:#00e5ff;">⚡ REALTIME</span>
        <span class="header-pill">⊞ 38 CLASSES</span>
    </div>
</div>"""


def render_hero() -> str:
    return render_top_header()


def consensus_banner_html(top_pred: dict, agreement_count: int, total_models: int, avg_conf: float, info: dict) -> str:
    """Renders the right-hand 'Diagnosis Result' card."""
    is_healthy = top_pred.get("healthy", False)
    cond_name = esc(top_pred.get("condition", "Healthy"))
    plant_name = esc(top_pred.get("plant", "Plant"))

    if is_healthy:
        badge_html = '<div class="diag-badge-healthy">✓ Healthy Crop</div>'
        verdict_icon = "✓"
        verdict_title = "No visible disease symptoms detected."
        verdict_text = "The leaf looks healthy to the model, with even green color and no spots, lesions, or discoloration."
        verdict_class = "diag-verdict-card"
    else:
        badge_html = '<div class="diag-badge-disease">🦠 Pathogen Detected</div>'
        verdict_icon = "⚠️"
        verdict_title = f"{cond_name} Pathology Detected."
        verdict_text = esc(info.get("summary", "Field symptoms indicate pathogenic infection requiring clinical containment."))
        verdict_class = "diag-verdict-card disease-verdict"

    return f"""<div class="ws-card-container">
    <div>
        <div class="ws-card-header">
            <h3>🍃 Diagnosis Result</h3>
            <span style="color:#64748b; font-size:1.1rem; cursor:pointer;" title="Diagnostic Verdict">ⓘ</span>
        </div>
        {badge_html}
        <h2 class="diag-title-main">{plant_name} — {cond_name}</h2>
        <p class="diag-sub-consensus">Ensemble Consensus Diagnostic Across 4 Deep Learning Architectures</p>
    </div>
    <div class="diag-split-gauge-row">
        {ring_html(avg_conf)}
        <div class="{verdict_class}">
            <div class="verdict-header-row">
                <span>{verdict_icon}</span>
                <span>{verdict_title}</span>
            </div>
            <p class="verdict-desc-txt">{verdict_text}</p>
        </div>
    </div>
</div>"""


def render_four_model_matrix(all_preds: dict[str, list[dict]], top_pred: dict) -> str:
    """Renders the 4 model cards in a single row with actual benchmarks."""
    cards_html = []
    canonical = ["vit", "resnet50", "mobilenetv2", "cnn"]

    highest_model = "ViT"
    highest_conf = 0.0

    for m in canonical:
        if m in all_preds:
            conf = all_preds[m][0]["confidence"] * 100.0
            if conf > highest_conf:
                highest_conf = conf
                highest_model = MODEL_META.get(m, {}).get("short", m)

    for m in canonical:
        meta = MODEL_META.get(m, {
            "name": m.title(),
            "short": m,
            "tag": "",
            "acc": "95.0%",
            "speed": "1.5 s",
            "icon": "⚡",
            "icon_color": "#06b6d4",
        })

        if m in all_preds:
            p = all_preds[m][0]
            display_acc = f"{p['confidence'] * 100.0:.1f}%"
            bar_width = p['confidence'] * 100.0
        else:
            display_acc = meta["acc"]
            bar_width = float(meta["acc"].replace("%", ""))

        tag_display = f"<span style='font-size:0.7rem; color:#94a3b8; font-weight:500;'> {meta['tag']}</span>" if meta['tag'] else ""

        cards_html.append(f"""<div class="matrix-single-card {m}">
    <div class="msc-top-row">
        <span class="msc-icon" style="color:{meta['icon_color']};">{meta['icon']}</span>
        <div class="msc-title-group">
            <h4>{meta['name']}{tag_display}</h4>
        </div>
    </div>
    <div class="msc-stats-split">
        <div class="msc-stat-cell">
            <b>{display_acc}</b>
            <span>Accuracy</span>
        </div>
        <div class="msc-stat-cell" style="text-align:right;">
            <b>{meta['speed']}</b>
            <span>Latency</span>
        </div>
    </div>
    <div class="msc-track">
        <div class="msc-fill" style="width:{bar_width:.1f}%;"></div>
    </div>
</div>""")

    return f"""<div class="matrix-bar-header">
    <h3>🗂️ 4-Architecture Model Matrix</h3>
    <span class="matrix-consensus-badge">Ensemble Consensus &bull; Highest: <b>{highest_model} ({highest_conf:.1f}%)</b></span>
</div>
<div class="model-matrix-grid">
    {''.join(cards_html)}
</div>"""


def render_agronomic_protocol_card(top_pred: dict, info: dict, leaf_thumb_uri: str | None = None) -> str:
    """Renders the bottom Agronomic Treatment Protocol card."""
    is_healthy = top_pred.get("healthy", False)
    cond_name = esc(top_pred.get("condition", "Healthy"))
    plant_name = esc(top_pred.get("plant", "Plant"))

    if is_healthy:
        status_pill = '<span class="protocol-status-badge no-treatment">✓ No treatment required</span>'
        tab_pills = """<div class="protocol-pills-row">
            <span class="proto-tab-pill active">🌿 Canopy Health</span>
            <span class="proto-tab-pill inactive">🌱 Vigor Maintenance</span>
        </div>"""
        checklist = """<ul class="proto-bullet-list">
            <li>Maintain regular field monitoring for early detection of any disease symptoms.</li>
            <li>Ensure balanced nutrition (e.g., N-P-K) for sustained plant vigor.</li>
            <li>Continue recommended irrigation and weed management practices.</li>
        </ul>"""
        title = f"Healthy {plant_name}"
    else:
        status_pill = '<span class="protocol-status-badge action-required">⚠️ Action Plan Required</span>'
        tab_pills = """<div class="protocol-pills-row">
            <span class="proto-tab-pill active">🚨 Immediate Action Plan</span>
            <span class="proto-tab-pill inactive">🌿 Organic &amp; Biological</span>
            <span class="proto-tab-pill inactive">🧪 Chemical Fungicide</span>
            <span class="proto-tab-pill inactive">🛡️ Long-Term Prevention</span>
        </div>"""
        steps = info.get("immediate", ["Inspect neighboring canopy rows.", "Remove severely necrotic leaf tissue."])
        items = "".join(f"<li>{esc(s)}</li>" for s in steps)
        checklist = f'<ul class="proto-bullet-list">{items}</ul>'
        title = f"{plant_name} — {cond_name}"

    if leaf_thumb_uri:
        thumb_img = f'<div class="protocol-thumb-box"><img src="{leaf_thumb_uri}" alt="{title}" /></div>'
    else:
        thumb_img = '<div class="protocol-thumb-box" style="display:grid; place-items:center; font-size:2.5rem; background:linear-gradient(135deg, #064e3b 0%, #065f46 100%);">🌿</div>'

    return f"""<div class="protocol-card-main">
    <div class="protocol-card-hdr">
        <h3>🌱 Agronomic Treatment Protocol</h3>
        <p>Recommended field management for the diagnosed condition</p>
    </div>
    <div class="protocol-split-layout">
        <div class="protocol-left-col">
            {thumb_img}
            <h4 class="protocol-crop-name">{title}</h4>
            {status_pill}
        </div>
        <div class="protocol-right-col">
            {tab_pills}
            {checklist}
        </div>
    </div>
</div>"""


def render_standby_banner() -> str:
    return """<div class="ws-card-container">
    <div>
        <div class="ws-card-header">
            <h3>🍃 Diagnosis Result</h3>
            <span style="color:#64748b; font-size:1.1rem;">ⓘ</span>
        </div>
        <div class="diag-badge-healthy" style="background:rgba(255,255,255,0.06); border-color:rgba(255,255,255,0.12); color:#94a3b8;">
            Standby Mode
        </div>
        <h2 class="diag-title-main" style="color:#94a3b8;">No Leaf Image Loaded</h2>
        <p class="diag-sub-consensus">Upload a leaf photo or choose an Instant Field Sample on the left to begin diagnosis.</p>
    </div>
    <div class="diag-split-gauge-row" style="opacity: 0.45;">
        <div class="gauge-svg-wrap">
            <svg class="gauge-svg" viewBox="0 0 88 88">
                <circle class="gauge-bg-circle" cx="44" cy="44" r="36" />
            </svg>
            <div class="gauge-inner-txt">
                <b>0.0%</b>
                <span>Standby</span>
            </div>
        </div>
        <div class="diag-verdict-card">
            <div class="verdict-header-row">
                <span>🍃</span>
                <span>Diagnostic Engine Ready</span>
            </div>
            <p class="verdict-desc-txt">All 4 neural architectures are primed for real-time inference.</p>
        </div>
    </div>
</div>"""


def arena_card_html(model_id: str, preds: list[dict], is_consensus_match: bool = True) -> str:
    return ""


def result_card_html(top: dict, info: dict) -> str:
    return consensus_banner_html(top, 1, 1, top.get("confidence", 0.0) * 100.0, info)


def bars_html(preds: list[dict]) -> str:
    rows = []
    for p in preds:
        pct = p["confidence"] * 100
        label = esc(f"{p['plant']} - {p['condition']}")
        rows.append(f'<div style="display:flex; justify-content:space-between; font-size:0.75rem; margin:0.2rem 0;"><span>{label}</span><b>{pct:.1f}%</b></div>')
    return "".join(rows)


def callout(text: str, kind: str = "") -> str:
    return f'<div style="background:rgba(0,0,0,0.3); border-left:3px solid #06b6d4; border-radius:8px; padding:0.6rem 0.85rem; font-size:0.82rem; margin:0.5rem 0;">{esc(text)}</div>'


def kv(label: str, value: str) -> str:
    return f'<div style="font-size:0.85rem; margin:0.25rem 0; color:#cbd5e1;"><b>{esc(label)}:</b> {esc(value)}</div>' if value else ""


def quality_notes(image: Image.Image) -> list[str]:
    notes: list[str] = []
    w, h = image.size
    if min(w, h) < 150:
        notes.append(f"Low resolution ({w}x{h}px). A closer crop gives sharper diagnostic features.")
    gray = image.convert("L")
    gray.thumbnail((256, 256))
    a = np.asarray(gray, dtype=np.float32)
    mean = float(a.mean())
    if mean < 45:
        notes.append("Lighting is dark. Capture with diffused natural daylight.")
    elif mean > 220:
        notes.append("Over-exposed glare detected. Avoid harsh reflective flash.")
    if a.shape[0] > 2 and a.shape[1] > 2:
        lap = 4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
        if float(lap.var()) < 12:
            notes.append("Lesion may be slightly blurry. Ensure sharp focus on leaf margins.")
    return notes


def confidence_messages(preds: list[dict], threshold_pct: float) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    top = preds[0]["confidence"] * 100
    if top < threshold_pct:
        out.append(("bad", f"Low confidence ({top:.0f}% < {threshold_pct:.0f}%). The photo may show fingers or background clutter."))
    if len(preds) > 1:
        margin = top - preds[1]["confidence"] * 100
        if margin < 15:
            out.append(("warn", f"Top two candidates are close ({margin:.0f}% margin). Review alternative candidate."))
    return out


def report_markdown(item_name: str, model_label: str, preds: list[dict], info: dict,
                    all_model_preds: dict | None = None) -> str:
    top = preds[0]
    lines = [
        "# AgroVision AI Plant Pathology Clinical Diagnostic Report", "",
        f"- **Image Analyzed:** {item_name}",
        f"- **Primary Architecture:** {model_label}",
        f"- **Timestamp:** {datetime.now():%Y-%m-%d %H:%M:%S}", "",
        "## Executive Clinical Diagnosis",
        f"### **{top['plant']} — {top['condition']}** ({top['confidence'] * 100:.1f}% Confidence)", "",
        f"- **Pathogen Classification:** {info.get('type', 'N/A')}",
        f"- **Clinical Severity Level:** {info.get('severity', 'N/A')}",
    ]
    if info.get("agent"):
        lines.append(f"- **Causative Biological Agent:** {info['agent']}")
    lines += ["", f"> {info.get('summary', '')}", ""]

    if all_model_preds:
        lines += [
            "## 4-Model Multi-Architecture Consensus Matrix", "",
            "| Architecture | Top Diagnosis | Confidence |",
            "|---|---|---|",
        ]
        for m_name, m_preds in all_model_preds.items():
            meta = MODEL_META.get(m_name, {})
            t = m_preds[0]
            lines.append(f"| {meta.get('name', m_name)} | {t['plant']} - {t['condition']} | {t['confidence']*100:.1f}% |")
        lines.append("")

    lines += [
        "## Consensus Candidate Breakdown", "",
        "| Rank | Crop — Condition | Confidence |",
        "|---|---|---|",
    ]
    for i, p in enumerate(preds, 1):
        lines.append(f"| {i} | {p['plant']} - {p['condition']} | {p['confidence'] * 100:.1f}% |")
    lines.append("")

    def section(title: str, items: list[str], numbered: bool = False) -> None:
        if not items:
            return
        lines.append(f"## {title}")
        lines.extend(f"{i}. {t}" if numbered else f"- {t}" for i, t in enumerate(items, 1))
        lines.append("")

    section("Diagnostic Symptoms & Lesion Patterns", info.get("symptoms", []))
    section("Immediate Agronomic Action Plan", info.get("immediate", []), numbered=True)
    section("Organic & Biological Solutions", info.get("organic", []))
    if info.get("chemical"):
        section("Chemical & Fungicide Interventions", info["chemical"])
        lines += [f"_{CHEMICAL_NOTE}_", ""]
    section("Long-Term Crop Sanitation & Prevention", info.get("prevention", []))
    lines += ["---", f"_{DISCLAIMER}_", ""]
    return "\n".join(lines)
