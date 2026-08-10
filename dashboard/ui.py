"""Shared UI helpers for the clinically grounded CBM dashboard.

Pure presentation layer: CSS, small HTML fragments and image compositing
helpers. No model/inference logic lives here (that stays in ``app.py``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import streamlit as st
from matplotlib import cm

PRIMARY = "#2563eb"
PURPLE = "#6d28d9"
NAVY = "#0f172a"
SLATE = "#64748b"

APP_CSS = """
<style>
:root {
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --navy: #0f172a;
    --slate: #475569;
    --blue: #2563eb;
    --purple: #6d28d9;
}

.stApp { background: var(--bg); }
.stApp, .stApp * {
    font-family: 'Segoe UI', system-ui, -apple-system, Roboto,
        'Helvetica Neue', Arial, sans-serif;
}

/* Constrain layout to a consistent grid; no full-width stretch */
div[data-testid="stMainBlockContainer"] {
    max-width: 1140px;
    margin: 0 auto;
    padding: 1rem 1.5rem 2.5rem;
}

/* Uniform, polished image rendering */
div[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid var(--border);
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
    display: block;
    margin: 0 auto;
}

/* Cards: white, thin border, subtle shadow, even padding */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.4rem;
}

/* Primary (blue) action buttons */
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    border: none;
    border-radius: 10px;
    color: #ffffff;
    font-weight: 600;
}
[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #1d4ed8, #4338ca);
    border: none;
    color: #ffffff;
}
[data-testid="stBaseButton-secondary"] {
    background: #ffffff;
    color: #1e293b;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    font-weight: 600;
}
[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--blue);
    color: var(--blue);
    background: #f8fafc;
}

/* Top-level navigation tabs */
div[data-testid="stTabs"] { margin-bottom: 0.6rem; }
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-weight: 700;
    color: var(--slate);
    padding: 0.55rem 1.15rem;
    border-radius: 10px 10px 0 0;
}
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--blue);
    border-bottom: 3px solid var(--blue);
}

/* Drag & drop uploader */
[data-testid="stFileUploaderDropzone"] {
    background: #f8fafc;
    border: 1.5px dashed #94a3b8;
    border-radius: 12px;
    padding: 1.4rem 1rem;
}
[data-testid="stFileUploaderDropzone"] button {
    border: 1px solid #cbd5e1;
    background: #ffffff;
    color: #1e293b;
    border-radius: 8px;
    font-weight: 600;
}

/* Header */
.cbm-header-band { display: flex; align-items: center; gap: 0.8rem; }
.logo-tile {
    width: 44px; height: 44px; flex: 0 0 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--blue), var(--purple));
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; color: #ffffff;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.35);
}
.cbm-title { font-size: 1.3rem; font-weight: 800; color: var(--navy); line-height: 1.15; margin: 0; }
.cbm-subtitle { font-size: 0.88rem; color: var(--slate); margin: 0; }

.proto-badge {
    background: #fefce8; border: 1px solid #fde68a; color: #854d0e;
    border-radius: 10px; padding: 0.45rem 0.75rem;
    font-size: 0.72rem; line-height: 1.45; text-align: center;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}

.card-title {
    font-size: 1.02rem; font-weight: 800; color: var(--navy);
    margin: 0 0 0.2rem 0; overflow-wrap: anywhere; line-height: 1.3;
}
.card-title-accent { color: var(--blue); }
.card-title-purple { color: var(--purple); }

.pill {
    display: inline-block; padding: 0.14rem 0.55rem; border-radius: 999px;
    font-weight: 700; font-size: 0.78rem; line-height: 1.4; white-space: nowrap;
}
.pill-green { background: #dcfce7; color: #166534; }
.pill-amber { background: #fef3c7; color: #92400e; }
.pill-red   { background: #fee2e2; color: #991b1b; }
.pill-gray  { background: #e2e8f0; color: #334155; }

.metric-card {
    background: #ffffff; border: 1px solid var(--border); border-radius: 12px;
    padding: 0.7rem 0.55rem; text-align: center;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    min-height: 4.2rem;
}
.metric-label { font-size: 0.72rem; color: var(--slate); font-weight: 600; margin-bottom: 0.2rem; }
.metric-value { font-size: 1.05rem; font-weight: 800; color: var(--navy); }

.prediction-big {
    font-size: 1.4rem; font-weight: 800; color: var(--purple);
    margin: 0.3rem 0 0.15rem 0; overflow-wrap: anywhere; line-height: 1.25;
}
.confidence {
    font-size: 0.95rem; font-weight: 600; color: var(--navy);
    margin: 0 0 0.3rem 0;
}

.legend-wrap { text-align: center; margin-top: 0.2rem; }
.legend-labels {
    display: flex; justify-content: space-between;
    font-size: 0.75rem; color: var(--slate); margin-top: 0.1rem;
}

.researcher-banner {
    background: linear-gradient(135deg, #0f172a, #312e81);
    color: #e0e7ff; border-radius: 14px; padding: 0.85rem 1.15rem;
    font-weight: 700; font-size: 0.95rem; margin-bottom: 0.8rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.15);
    overflow-wrap: anywhere;
}

/* Empty-state placeholders (secondary, but readable) */
.placeholder {
    color: var(--slate); font-size: 0.9rem; min-height: 3.5rem;
    display: flex; align-items: center; justify-content: center;
    text-align: center; padding: 0.6rem 0;
}
.pred-placeholder {
    color: var(--slate); text-align: center; padding: 2.2rem 1rem;
    font-size: 0.95rem; border: 1px dashed #cbd5e1; border-radius: 12px;
    background: #f8fafc;
}
.pred-placeholder .big {
    font-size: 1.15rem; font-weight: 800; color: #94a3b8; margin-bottom: 0.2rem;
}

/* Captions: readable secondary text, never near-invisible */
.stCaptionContainer p, [data-testid="stCaptionContainer"] p { color: var(--slate); }

/* ---- Findings list / rows ---- */
.finding-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.5rem 0.1rem; border-bottom: 1px dashed var(--border);
}
.finding-row:last-child { border-bottom: none; }
.finding-name { font-weight: 700; color: #1e293b; font-size: 0.92rem; }
.finding-prob { color: var(--slate); font-size: 0.85rem; font-weight: 600; }

/* ---- Impression / interpretation box ---- */
.impression-box {
    background: #eff6ff; border-left: 4px solid var(--blue);
    border-radius: 10px; padding: 0.7rem 1rem; margin: 0.35rem 0 0.6rem 0;
}
.impression-box p { margin: 0.2rem 0; color: #1e293b; line-height: 1.55; }
.impression-box .impression-label {
    font-size: 0.72rem; font-weight: 800; letter-spacing: 0.07em;
    text-transform: uppercase; color: var(--blue); margin-bottom: 0.25rem;
}

/* ---- Section label ---- */
.section-label {
    font-size: 0.72rem; font-weight: 800; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--slate); margin: 0.45rem 0 0.2rem 0;
}

/* ---- Normal / abnormal highlight ---- */
.normal-banner {
    background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534;
    border-radius: 10px; padding: 0.6rem 0.9rem; font-weight: 700;
    font-size: 0.9rem; margin: 0.3rem 0;
}
.abnormal-banner {
    background: #fef2f2; border: 1px solid #fecaca; color: #991b1b;
    border-radius: 10px; padding: 0.6rem 0.9rem; font-weight: 700;
    font-size: 0.9rem; margin: 0.3rem 0;
}

/* ---- Viewing-tool hint ---- */
.tool-hint { color: var(--slate); font-size: 0.8rem; line-height: 1.45; }

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: #ffffff;
}

/* Keep long text inside its box */
.stMarkdown p, .stCaptionContainer p, .stMarkdown li { overflow-wrap: anywhere; }

/* ---- Overall Analysis concept rows ---- */
.concept-row {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.42rem 0; border-bottom: 1px dashed #e2e8f0;
}
.concept-row:last-child { border-bottom: none; }
.c-icon { width: 1.4rem; flex: 0 0 1.4rem; text-align: center; font-size: 1.05rem; }
.c-name { flex: 1.25; font-weight: 700; color: #1e293b; font-size: 0.9rem; }
.c-status { flex: 1.15; }
.c-prob { width: 2.7rem; flex: 0 0 2.7rem; text-align: right; font-weight: 800; color: #0f172a; font-size: 0.85rem; }
.c-bar { flex: 1; min-width: 64px; }
.prob-bar { height: 7px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 999px; }

/* ---- Clinical concept cards ---- */
.ccard {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 0.6rem; text-align: center;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}
.ccard.active {
    border: 2px solid #2563eb;
    box-shadow: 0 3px 10px rgba(37, 99, 235, 0.22);
}
.ccard img {
    width: 100%; border-radius: 8px; border: 1px solid #e2e8f0;
    aspect-ratio: 1 / 1; object-fit: cover;
}
.ccard-name { font-weight: 700; color: #1e293b; font-size: 0.82rem; margin-top: 0.4rem; }
.ccard-row { display: flex; align-items: center; justify-content: space-between; gap: 0.3rem; margin-top: 0.35rem; }
.ccard-prob { font-weight: 800; color: #0f172a; font-size: 0.82rem; }

/* ---- Concept detail stats ---- */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.55rem 1rem; margin: 0.45rem 0; }
.stat-item { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.5rem 0.7rem; }
.stat-label { font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }
.stat-value { font-size: 0.95rem; font-weight: 800; color: #0f172a; margin-top: 0.1rem; }
.about-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 0.8rem 1rem; }
.about-box .about-label {
    font-size: 0.72rem; font-weight: 800; letter-spacing: 0.07em;
    text-transform: uppercase; color: #6d28d9; margin-bottom: 0.25rem;
}
.about-box p { margin: 0.15rem 0; color: #334155; line-height: 1.55; font-size: 0.9rem; }

/* ---- Detail image labels ---- */
.detail-label {
    font-size: 0.78rem; font-weight: 800; color: #475569;
    letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 0.3rem;
}
</style>
"""


def inject_css() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)


def card_title(text: str, accent: Optional[str] = None) -> str:
    """Markdown (HTML) fragment for a card heading."""
    cls = ""
    if accent == "blue":
        cls = " card-title-accent"
    elif accent == "purple":
        cls = " card-title-purple"
    return f'<div class="card-title{cls}">{text}</div>'


def logo_tile() -> str:
    return '<div class="logo-tile">🫁</div>'


def header_left_html() -> str:
    return (
        f'<div class="cbm-header-band">{logo_tile()}'
        "<div><div class=\"cbm-title\">Clinically Grounded CBM</div>"
        '<div class="cbm-subtitle">Concept-based Chest X-ray Analysis</div>'
        "</div></div>"
    )


def prototype_badge_html() -> str:
    return (
        '<div class="proto-badge">🔒 Research Prototype<br>'
        "Not for clinical diagnosis<br>or treatment decisions.</div>"
    )


def probability_badge(prob: float, threshold: float = 0.5) -> str:
    """Colored pill showing a model probability (not a clinical standard)."""
    if prob >= 0.6:
        level = "pill-green"
    elif prob >= threshold:
        level = "pill-amber"
    else:
        level = "pill-red"
    return f'<span class="pill {level}">{prob:.0%}</span>'


def status_pill(prob: float, threshold: float = 0.5) -> str:
    """Doctor-facing present/absent status pill from a model probability.

    The label reflects *model confidence* only, never a clinical diagnosis.
    """
    if prob >= 0.7:
        level, label = "pill-red", "Present · high conf."
    elif prob >= threshold:
        level, label = "pill-amber", "Present · low conf."
    else:
        level, label = "pill-gray", "Not detected"
    return (
        f'<span class="pill {level}" title="model probability {prob:.1%}">'
        f"{label}</span>"
    )


def impression_html(label: str, text: str) -> str:
    """Light-blue, clinical-style impression box."""
    safe = text.replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<div class="impression-box"><div class="impression-label">{label}</div>'
        f"<p>{safe}</p></div>"
    )


def status_banner(any_detected: bool) -> str:
    """Full-width normal / abnormal banner for the prediction card."""
    if any_detected:
        return (
            '<div class="abnormal-banner">⚠ Findings detected — see below for '
            "details.</div>"
        )
    return (
        '<div class="normal-banner">✓ No significant abnormality detected at the '
        "model's decision threshold.</div>"
    )


def probability_bar_html(prob: float, threshold: float = 0.5) -> str:
    """Small horizontal bar whose fill reflects a model probability."""
    if prob >= 0.7:
        color = "#ef4444"
    elif prob >= threshold:
        color = "#f59e0b"
    else:
        color = "#94a3b8"
    pct = max(0.0, min(1.0, prob)) * 100
    return (
        f'<div class="prob-bar"><div class="prob-bar-fill" '
        f'style="width:{pct:.1f}%;background:{color}"></div></div>'
    )


def concept_row_html(icon: str, name: str, status_html: str, prob: float,
                     threshold: float = 0.5) -> str:
    """Compact row for the Overall Analysis summary."""
    return (
        '<div class="concept-row">'
        f'<span class="c-icon">{icon}</span>'
        f'<span class="c-name">{name}</span>'
        f'<span class="c-status">{status_html}</span>'
        f'<span class="c-prob">{prob:.0%}</span>'
        f'<span class="c-bar">{probability_bar_html(prob, threshold)}</span>'
        "</div>"
    )


def concept_card_html(name: str, prob: float, status_html: str, data_uri: str,
                      active: bool = False) -> str:
    """Medical-AI concept card: evidence image, name, status and probability."""
    cls = "ccard active" if active else "ccard"
    return (
        f'<div class="{cls}">'
        f'<img src="{data_uri}" alt="{name}"/>'
        f'<div class="ccard-name">{name}</div>'
        f'<div class="ccard-row">{status_html}'
        f'<span class="ccard-prob">{prob:.0%}</span></div>'
        "</div>"
    )


def patient_info_html(age, sex, indication) -> str:
    """Compact patient-context card fed by the doctor."""
    age_txt = f"{int(age)} y" if age else "—"
    sex_txt = sex if sex else "—"
    cells = (
        f'<div><div class="patient-label">Age</div>'
        f'<div class="patient-value">{age_txt}</div></div>'
        f'<div><div class="patient-label">Sex</div>'
        f'<div class="patient-value">{sex_txt}</div></div>'
    )
    grid = f'<div class="patient-grid">{cells}</div>'
    if indication:
        grid += (
            f'<div class="patient-indication"><div class="patient-label">'
            "Clinical indication</div>"
            f'<div class="patient-value">{indication}</div></div>'
        )
    return grid


def metric_card_html(label: str, value: str) -> str:
    return (
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>'
    )


def overlay_image(display: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Composite ``cam`` (normalized 0..1, same size as ``display``) over a
    grayscale X-ray. Returns an RGB uint8 array."""
    heatmap = (cm.jet(np.asarray(cam, dtype=np.float32))[..., :3] * 255.0).astype(np.uint8)
    disp = np.stack([np.asarray(display)] * 3, axis=-1)
    out = (0.42 * disp.astype(np.float32) + 0.58 * heatmap.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


def legend_image(width: int = 260, height: int = 16) -> np.ndarray:
    """Jet gradient used as the Low -> High evidence legend."""
    grad = cm.jet(np.linspace(0.0, 1.0, width))[..., :3]
    img = np.tile(grad, (height, 1, 1))
    return (img * 255.0).astype(np.uint8)
