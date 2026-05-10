# app.py — AI Face Spoof Detection System
# Futuristic Cyberpunk UI | Compatible with Streamlit >= 1.28.0
# Author: Developed by Siva | AI/ML Final Year Project

import streamlit as st
import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Path bootstrap ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Model path ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.absolute()
DEFAULT_MODEL = str(SCRIPT_DIR / "best_model.h5")

# ── Page config (must be FIRST streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Face Spoof Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Cyberpunk / Glassmorphism / Neon
# ─────────────────────────────────────────────────────────────────────────────
CYBER_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

/* ── Root tokens ── */
:root {
  --neon-blue:    #00d4ff;
  --neon-purple:  #a855f7;
  --neon-cyan:    #06ffd4;
  --neon-green:   #39ff14;
  --neon-red:     #ff2d55;
  --bg-deep:      #020408;
  --bg-panel:     rgba(6,14,30,0.82);
  --glass-border: rgba(0,212,255,0.18);
  --text-main:    #e2f0ff;
  --text-dim:     #7a9bbf;
}

/* ── Global reset ── */
html, body, [class*="css"], .stApp {
  background-color: var(--bg-deep) !important;
  font-family: 'Rajdhani', sans-serif !important;
  color: var(--text-main) !important;
}

/* ── Animated particle background ── */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 60% at 20% 10%, rgba(0,212,255,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 80% at 80% 80%, rgba(168,85,247,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 50% 50% at 50% 50%, rgba(6,255,212,0.03) 0%, transparent 70%);
  animation: bgPulse 8s ease-in-out infinite alternate;
  pointer-events: none;
  z-index: 0;
}
@keyframes bgPulse {
  0%   { opacity: 0.6; }
  100% { opacity: 1.0; }
}

/* ── Grid overlay ── */
.stApp::after {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
  background-size: 50px 50px;
  pointer-events: none;
  z-index: 0;
}

/* ── Hero header ── */
.cyber-hero {
  position: relative;
  text-align: center;
  padding: 2.5rem 1.5rem 2rem;
  margin-bottom: 1.8rem;
  background: linear-gradient(135deg,
    rgba(0,212,255,0.08) 0%,
    rgba(168,85,247,0.10) 50%,
    rgba(6,255,212,0.06) 100%);
  border: 1px solid var(--glass-border);
  border-radius: 20px;
  backdrop-filter: blur(12px);
  overflow: hidden;
  animation: heroFadeIn 1s ease forwards;
}
@keyframes heroFadeIn {
  from { opacity: 0; transform: translateY(-20px); }
  to   { opacity: 1; transform: translateY(0); }
}
.cyber-hero::before {
  content: '';
  position: absolute;
  top: 0; left: -100%;
  width: 60%; height: 2px;
  background: linear-gradient(90deg, transparent, var(--neon-blue), transparent);
  animation: scanLine 3s linear infinite;
}
@keyframes scanLine {
  0%   { left: -60%; }
  100% { left: 160%; }
}
.hero-badge {
  display: inline-block;
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.25em;
  color: var(--neon-cyan);
  background: rgba(6,255,212,0.08);
  border: 1px solid rgba(6,255,212,0.3);
  border-radius: 50px;
  padding: 0.22rem 1rem;
  margin-bottom: 0.8rem;
}
.hero-title {
  font-family: 'Orbitron', monospace !important;
  font-size: clamp(1.6rem, 4vw, 3.2rem) !important;
  font-weight: 900 !important;
  line-height: 1.1 !important;
  background: linear-gradient(90deg, var(--neon-blue), var(--neon-purple), var(--neon-cyan));
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
  margin: 0.3rem 0 !important;
  animation: titleGlow 3s ease-in-out infinite alternate;
}
@keyframes titleGlow {
  from { filter: drop-shadow(0 0 8px rgba(0,212,255,0.3)); }
  to   { filter: drop-shadow(0 0 20px rgba(168,85,247,0.6)); }
}
.hero-sub {
  font-size: 1.05rem;
  color: var(--text-dim);
  letter-spacing: 0.12em;
  margin-top: 0.4rem;
  text-transform: uppercase;
}
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 1rem;
  font-family: 'Orbitron', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.2em;
  color: var(--neon-green);
  background: rgba(57,255,20,0.08);
  border: 1px solid rgba(57,255,20,0.3);
  border-radius: 50px;
  padding: 0.3rem 1.2rem;
}
.pulse-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--neon-green);
  box-shadow: 0 0 8px var(--neon-green);
  animation: pulseDot 1.5s ease-in-out infinite;
}
@keyframes pulseDot {
  0%, 100% { transform: scale(1);   opacity: 1;   }
  50%       { transform: scale(1.5); opacity: 0.4; }
}

/* ── Glass card ── */
.glass-card {
  background: var(--bg-panel);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  padding: 1.4rem 1.6rem;
  backdrop-filter: blur(16px);
  position: relative;
  overflow: hidden;
  transition: border-color 0.3s, box-shadow 0.3s;
}
.glass-card:hover {
  border-color: rgba(0,212,255,0.4);
  box-shadow: 0 0 30px rgba(0,212,255,0.12);
}

/* ── Section label ── */
.section-label {
  font-family: 'Orbitron', monospace;
  font-size: 0.7rem;
  letter-spacing: 0.25em;
  color: var(--neon-cyan);
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}
.section-title {
  font-family: 'Orbitron', monospace;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text-main);
  margin-bottom: 1rem;
}

/* ── Metric cards ── */
.metric-row {
  display: flex;
  gap: 0.8rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
}
.metric-card {
  flex: 1;
  min-width: 90px;
  background: rgba(0,212,255,0.05);
  border: 1px solid rgba(0,212,255,0.15);
  border-radius: 12px;
  padding: 0.8rem 0.6rem;
  text-align: center;
  transition: all 0.3s;
}
.metric-card:hover {
  background: rgba(0,212,255,0.1);
  border-color: var(--neon-blue);
  transform: translateY(-2px);
}
.metric-val {
  font-family: 'Orbitron', monospace;
  font-size: 1.5rem;
  font-weight: 800;
  color: var(--neon-blue);
  line-height: 1;
}
.metric-val.green  { color: var(--neon-green); }
.metric-val.red    { color: var(--neon-red);   }
.metric-val.purple { color: var(--neon-purple);}
.metric-lbl {
  font-size: 0.72rem;
  color: var(--text-dim);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-top: 0.3rem;
}

/* ── Result banner ── */
.result-real {
  background: linear-gradient(135deg, rgba(57,255,20,0.08), rgba(0,212,255,0.05));
  border: 1px solid rgba(57,255,20,0.4);
  border-radius: 16px;
  padding: 1.6rem;
  text-align: center;
  animation: resultFade 0.6s ease forwards;
  box-shadow: 0 0 30px rgba(57,255,20,0.12), inset 0 0 30px rgba(57,255,20,0.04);
}
.result-spoof {
  background: linear-gradient(135deg, rgba(255,45,85,0.1), rgba(168,85,247,0.06));
  border: 1px solid rgba(255,45,85,0.4);
  border-radius: 16px;
  padding: 1.6rem;
  text-align: center;
  animation: resultFade 0.6s ease forwards;
  box-shadow: 0 0 30px rgba(255,45,85,0.15), inset 0 0 30px rgba(255,45,85,0.04);
}
.result-noface {
  background: linear-gradient(135deg, rgba(120,120,120,0.08), rgba(60,60,60,0.05));
  border: 1px solid rgba(150,150,150,0.3);
  border-radius: 16px;
  padding: 1.6rem;
  text-align: center;
  animation: resultFade 0.6s ease forwards;
}
@keyframes resultFade {
  from { opacity: 0; transform: scale(0.95); }
  to   { opacity: 1; transform: scale(1);    }
}
.result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.result-label {
  font-family: 'Orbitron', monospace;
  font-size: 1.8rem;
  font-weight: 900;
  letter-spacing: 0.1em;
}
.result-label.green  { color: var(--neon-green); text-shadow: 0 0 20px rgba(57,255,20,0.5); }
.result-label.red    { color: var(--neon-red);   text-shadow: 0 0 20px rgba(255,45,85,0.5); }
.result-label.gray   { color: #888; }
.result-conf {
  font-size: 1rem;
  color: var(--text-dim);
  margin-top: 0.3rem;
  letter-spacing: 0.08em;
}

/* ── Confidence bar ── */
.conf-wrap { margin-top: 1rem; }
.conf-header {
  display: flex;
  justify-content: space-between;
  font-size: 0.78rem;
  color: var(--text-dim);
  margin-bottom: 0.35rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.conf-track {
  height: 10px;
  border-radius: 10px;
  background: rgba(255,255,255,0.07);
  overflow: hidden;
  position: relative;
}
.conf-fill-green {
  height: 100%;
  border-radius: 10px;
  background: linear-gradient(90deg, #0aff6e, #39ff14);
  box-shadow: 0 0 12px rgba(57,255,20,0.5);
  transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
}
.conf-fill-red {
  height: 100%;
  border-radius: 10px;
  background: linear-gradient(90deg, #ff2d55, #ff6b6b);
  box-shadow: 0 0 12px rgba(255,45,85,0.5);
  transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
}
.conf-fill-gray {
  height: 100%;
  border-radius: 10px;
  background: linear-gradient(90deg, #666, #999);
  transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── History table ── */
.history-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}
.history-table th {
  font-family: 'Orbitron', monospace;
  font-size: 0.62rem;
  letter-spacing: 0.2em;
  color: var(--neon-cyan);
  text-transform: uppercase;
  padding: 0.7rem 0.8rem;
  border-bottom: 1px solid rgba(0,212,255,0.15);
  text-align: left;
}
.history-table td {
  padding: 0.6rem 0.8rem;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  color: var(--text-main);
  vertical-align: middle;
}
.history-table tr:hover td {
  background: rgba(0,212,255,0.04);
}
.badge-real {
  display: inline-block;
  padding: 0.18rem 0.7rem;
  border-radius: 50px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  background: rgba(57,255,20,0.12);
  border: 1px solid rgba(57,255,20,0.35);
  color: var(--neon-green);
}
.badge-spoof {
  display: inline-block;
  padding: 0.18rem 0.7rem;
  border-radius: 50px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  background: rgba(255,45,85,0.12);
  border: 1px solid rgba(255,45,85,0.35);
  color: var(--neon-red);
}
.badge-noface {
  display: inline-block;
  padding: 0.18rem 0.7rem;
  border-radius: 50px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  background: rgba(150,150,150,0.12);
  border: 1px solid rgba(150,150,150,0.3);
  color: #aaa;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
  background: rgba(0,212,255,0.03) !important;
  border: 1.5px dashed rgba(0,212,255,0.3) !important;
  border-radius: 14px !important;
  transition: all 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(0,212,255,0.6) !important;
  background: rgba(0,212,255,0.06) !important;
  box-shadow: 0 0 20px rgba(0,212,255,0.1) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,
    rgba(4,10,22,0.98) 0%,
    rgba(6,14,30,0.98) 100%) !important;
  border-right: 1px solid rgba(0,212,255,0.12) !important;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem !important; }

.sidebar-logo {
  text-align: center;
  padding: 1.2rem 0 1rem;
  border-bottom: 1px solid rgba(0,212,255,0.12);
  margin-bottom: 1.2rem;
}
.sidebar-logo-icon {
  font-size: 3rem;
  display: block;
  filter: drop-shadow(0 0 16px rgba(0,212,255,0.6));
  animation: iconPulse 3s ease-in-out infinite;
}
@keyframes iconPulse {
  0%, 100% { filter: drop-shadow(0 0 10px rgba(0,212,255,0.4)); }
  50%       { filter: drop-shadow(0 0 24px rgba(168,85,247,0.7)); }
}
.sidebar-logo-text {
  font-family: 'Orbitron', monospace;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.25em;
  color: var(--neon-blue);
  margin-top: 0.4rem;
  text-transform: uppercase;
}
.sidebar-section {
  font-family: 'Orbitron', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.25em;
  color: var(--neon-purple);
  text-transform: uppercase;
  padding: 0.8rem 0 0.4rem;
  border-top: 1px solid rgba(168,85,247,0.12);
  margin-top: 0.8rem;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
  background: linear-gradient(90deg, var(--neon-blue), var(--neon-purple)) !important;
}

/* ── Buttons ── */
.stButton > button {
  width: 100%;
  font-family: 'Orbitron', monospace !important;
  font-size: 0.78rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  border-radius: 10px !important;
  padding: 0.65rem 1rem !important;
  transition: all 0.3s ease !important;
  border: 1px solid rgba(0,212,255,0.4) !important;
  background: linear-gradient(135deg,
    rgba(0,212,255,0.12) 0%,
    rgba(168,85,247,0.12) 100%) !important;
  color: var(--neon-blue) !important;
  box-shadow: 0 0 12px rgba(0,212,255,0.1) !important;
}
.stButton > button:hover {
  border-color: var(--neon-blue) !important;
  background: linear-gradient(135deg,
    rgba(0,212,255,0.22) 0%,
    rgba(168,85,247,0.22) 100%) !important;
  box-shadow: 0 0 24px rgba(0,212,255,0.25) !important;
  transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #00d4ff, #a855f7) !important;
  color: #000 !important;
  border: none !important;
  box-shadow: 0 0 20px rgba(0,212,255,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 0 36px rgba(0,212,255,0.55) !important;
  transform: translateY(-2px) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--neon-cyan) !important; }

/* ── Success / Error / Warning / Info ── */
[data-testid="stAlert"] {
  border-radius: 12px !important;
  border-left-width: 3px !important;
  backdrop-filter: blur(8px) !important;
}

/* ── Footer ── */
.cyber-footer {
  text-align: center;
  margin-top: 3rem;
  padding: 1.5rem;
  border-top: 1px solid rgba(0,212,255,0.12);
  position: relative;
}
.cyber-footer::before {
  content: '';
  position: absolute;
  top: 0; left: 50%;
  transform: translateX(-50%);
  width: 200px; height: 1px;
  background: linear-gradient(90deg,
    transparent, var(--neon-blue), var(--neon-purple), transparent);
  box-shadow: 0 0 10px var(--neon-blue);
}
.footer-dev {
  font-family: 'Orbitron', monospace;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.2em;
  color: var(--neon-blue);
}
.footer-copy {
  font-size: 0.72rem;
  color: var(--text-dim);
  margin-top: 0.3rem;
  letter-spacing: 0.08em;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.3); }
::-webkit-scrollbar-thumb {
  background: linear-gradient(var(--neon-blue), var(--neon-purple));
  border-radius: 10px;
}

/* ── Misc streamlit overrides ── */
.stMarkdown p { color: var(--text-main) !important; }
[data-testid="metric-container"] {
  background: rgba(0,212,255,0.04) !important;
  border: 1px solid rgba(0,212,255,0.12) !important;
  border-radius: 10px !important;
  padding: 0.6rem 0.8rem !important;
}
[data-testid="stCheckbox"] span { color: var(--text-main) !important; }
[data-testid="stTextInput"] input {
  background: rgba(0,212,255,0.04) !important;
  border: 1px solid rgba(0,212,255,0.2) !important;
  color: var(--text-main) !important;
  border-radius: 8px !important;
}
div[data-testid="column"] { gap: 0 !important; }
</style>
"""

st.markdown(CYBER_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "total_predictions": 0,
        "real_count":  0,
        "spoof_count": 0,
        "detector": None,
        "history": [],          # list of dicts
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# Detector loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_detector(model_path: str, threshold: float, use_face_det: bool):
    """Cache the heavy model load across reruns."""
    from predict_live import LiveSpoofDetector
    return LiveSpoofDetector(
        model_path=model_path,
        threshold=threshold,
        use_face_detection=use_face_det,
        smooth_frames=5,
        camera_id=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: confidence bar HTML
# ─────────────────────────────────────────────────────────────────────────────
def confidence_bar(confidence: float, cls: str) -> str:
    fill_class = (
        "conf-fill-green" if cls == "REAL" else
        "conf-fill-red"   if cls == "SPOOF" else
        "conf-fill-gray"
    )
    return f"""
    <div class="conf-wrap">
      <div class="conf-header">
        <span>Confidence</span>
        <span style="font-family:'Orbitron',monospace;color:var(--neon-blue);">
          {confidence:.1f}%
        </span>
      </div>
      <div class="conf-track">
        <div class="{fill_class}" style="width:{confidence:.1f}%"></div>
      </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# Helper: result panel HTML
# ─────────────────────────────────────────────────────────────────────────────
def result_panel(cls: str, confidence: float, face_conf: float) -> str:
    if cls == "REAL":
        icon, label_cls, card_cls, label = "🛡️", "green", "result-real", "GENUINE"
    elif cls == "SPOOF":
        icon, label_cls, card_cls, label = "⚠️", "red",   "result-spoof", "SPOOF DETECTED"
    else:
        icon, label_cls, card_cls, label = "🔍", "gray",  "result-noface", "NO FACE"

    conf_str = f"{confidence:.1f}%" if cls != "NO_FACE" else "N/A"
    face_str = f"{face_conf:.2f}" if face_conf > 0 else "—"

    bar = confidence_bar(confidence, cls) if cls != "NO_FACE" else ""

    return f"""
    <div class="{card_cls}">
      <div class="result-icon">{icon}</div>
      <div class="result-label {label_cls}">{label}</div>
      <div class="result-conf">Confidence · {conf_str}</div>
      {bar}
      <div style="margin-top:0.9rem;font-size:0.78rem;color:var(--text-dim);
                  letter-spacing:0.08em;text-transform:uppercase;">
        Face Detection Score · {face_str}
      </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# Helper: draw CV2 overlay on result image
# ─────────────────────────────────────────────────────────────────────────────
def draw_overlay(image: np.ndarray, cls: str, confidence: float,
                 bbox, color) -> np.ndarray:
    out = image.copy()

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        clen = 20
        for px, py, dx, dy in [
            (x, y, clen, 0), (x, y, 0, clen),
            (x + w, y, -clen, 0), (x + w, y, 0, clen),
            (x, y + h, clen, 0), (x, y + h, 0, -clen),
            (x + w, y + h, -clen, 0), (x + w, y + h, 0, -clen),
        ]:
            cv2.line(out, (px, py), (px + dx, py + dy), color, 3)

    if cls == "REAL":
        label = f"REAL  {confidence:.1f}%"
        bg = (0, 200, 50)
    elif cls == "SPOOF":
        label = f"SPOOF  {confidence:.1f}%"
        bg = (20, 20, 220)
    else:
        label = "NO FACE"
        bg = (120, 120, 120)

    tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
    cv2.rectangle(out, (8, 8), (tw + 28, 50), (0, 0, 0), -1)
    cv2.rectangle(out, (8, 8), (tw + 28, 50), bg, 2)
    cv2.putText(out, label, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, bg, 2)

    if cls != "NO_FACE":
        bx, by, bw, bh = 8, 60, 260, 14
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        fw = int(bw * confidence / 100)
        cv2.rectangle(out, (bx, by), (bx + fw, by + bh), bg, -1)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Helper: history badge HTML
# ─────────────────────────────────────────────────────────────────────────────
def history_badge(cls: str) -> str:
    if cls == "REAL":
        return '<span class="badge-real">REAL</span>'
    elif cls == "SPOOF":
        return '<span class="badge-spoof">SPOOF</span>'
    return '<span class="badge-noface">NO FACE</span>'


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple:
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="sidebar-logo">
          <span class="sidebar-logo-icon">🛡️</span>
          <div class="sidebar-logo-text">BioSentinel AI</div>
        </div>
        """, unsafe_allow_html=True)

        # Settings
        st.markdown('<div class="sidebar-section">⚙ Configuration</div>',
                    unsafe_allow_html=True)

        model_path = st.text_input("Model Path", value=DEFAULT_MODEL,
                                   help="Absolute path to best_model.h5")
        threshold = st.slider("Detection Threshold", 0.0, 1.0, 0.50, 0.01,
                              help="Lower → more sensitive to SPOOF")
        use_face_det = st.checkbox("Enable Face Detection (MTCNN)", value=True,
                                   help="Detect face region before classifying")

        init_btn = st.button("⚡ Initialize System", use_container_width=True)

        # Model status
        model_exists = os.path.exists(model_path)
        if st.session_state.detector:
            st.markdown("""
            <div style="text-align:center;margin-top:.6rem;padding:.5rem;
                        border-radius:8px;background:rgba(57,255,20,.07);
                        border:1px solid rgba(57,255,20,.3);font-size:.78rem;
                        color:var(--neon-green);font-family:'Orbitron',monospace;
                        letter-spacing:.1em;">
              ✓ SYSTEM READY
            </div>""", unsafe_allow_html=True)
        elif model_exists:
            st.info("Model found — click **Initialize System**")
        else:
            st.error("⚠ Model not found at path above")

        # Stats
        st.markdown('<div class="sidebar-section">📊 Statistics</div>',
                    unsafe_allow_html=True)

        total = st.session_state.total_predictions
        real  = st.session_state.real_count
        spoof = st.session_state.spoof_count
        pct   = f"{(real/total*100):.0f}%" if total else "—"

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="metric-val">{total}</div>
            <div class="metric-lbl">Total</div>
          </div>
          <div class="metric-card">
            <div class="metric-val green">{real}</div>
            <div class="metric-lbl">Real</div>
          </div>
          <div class="metric-card">
            <div class="metric-val red">{spoof}</div>
            <div class="metric-lbl">Spoof</div>
          </div>
          <div class="metric-card">
            <div class="metric-val purple">{pct}</div>
            <div class="metric-lbl">Real %</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("↺ Reset Statistics", use_container_width=True):
            st.session_state.total_predictions = 0
            st.session_state.real_count  = 0
            st.session_state.spoof_count = 0
            st.session_state.history     = []
            st.success("Statistics cleared.")

        # About
        st.markdown('<div class="sidebar-section">ℹ About</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:.82rem;color:var(--text-dim);line-height:1.6;">
          <b style="color:var(--neon-cyan)">Model:</b> MobileNetV3<br>
          <b style="color:var(--neon-cyan)">Detector:</b> MTCNN<br>
          <b style="color:var(--neon-cyan)">Framework:</b> TensorFlow 2.x<br>
          <b style="color:var(--neon-cyan)">UI:</b> Streamlit 1.28+
        </div>
        """, unsafe_allow_html=True)

    return model_path, threshold, use_face_det, init_btn


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
def render_hero():
    st.markdown("""
    <div class="cyber-hero">
      <div class="hero-badge">🔐 ADVANCED LIVENESS DETECTION</div>
      <div class="hero-title">AI Face Spoof Detection System</div>
      <div class="hero-sub">Real-Time Deep Learning Biometric Security</div>
      <div class="status-pill">
        <span class="pulse-dot"></span>
        SYSTEM ACTIVE · BIOSENTINEL v2.0
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def render_history():
    history = st.session_state.history
    if not history:
        st.markdown("""
        <div style="text-align:center;padding:1.5rem;color:var(--text-dim);
                    font-size:.88rem;letter-spacing:.08em;">
          No detections yet — upload an image above.
        </div>
        """, unsafe_allow_html=True)
        return

    rows = ""
    for i, item in enumerate(reversed(history[-20:])):
        badge = history_badge(item["cls"])
        conf  = f"{item['confidence']:.1f}%" if item["cls"] != "NO_FACE" else "—"
        rows += f"""
        <tr>
          <td style="color:var(--text-dim);font-size:.78rem;">
            #{len(history)-i}
          </td>
          <td style="font-size:.78rem;color:var(--text-dim);">
            {item['timestamp']}
          </td>
          <td>{item['filename']}</td>
          <td>{badge}</td>
          <td style="font-family:'Orbitron',monospace;color:var(--neon-blue);">
            {conf}
          </td>
        </tr>
        """

    st.markdown(f"""
    <div class="glass-card" style="overflow-x:auto;">
      <table class="history-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Time</th>
            <th>File</th>
            <th>Result</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    render_hero()

    model_path, threshold, use_face_det, init_btn = render_sidebar()

    # ── Initialize detector ───────────────────────────────────────────────
    if init_btn:
        if not os.path.exists(model_path):
            st.error(f"❌ Model not found: `{model_path}`")
        else:
            with st.spinner("⚡ Booting BioSentinel AI — loading model weights…"):
                try:
                    # Clear cache so new settings take effect
                    load_detector.clear()
                    detector = load_detector(model_path, threshold, use_face_det)
                    st.session_state.detector = detector
                    st.success("✅ Detector initialized — system is operational.")
                except Exception as exc:
                    st.error(f"Initialization failed: {exc}")
                    st.session_state.detector = None

    # ── Gate on detector ──────────────────────────────────────────────────
    if st.session_state.detector is None:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:2.5rem;">
          <div style="font-size:3rem;margin-bottom:.8rem;">🛡️</div>
          <div class="section-title">System Offline</div>
          <div style="color:var(--text-dim);font-size:.9rem;line-height:1.7;">
            Ensure <code>best_model.h5</code> is in the same directory,<br>
            then click <b>⚡ Initialize System</b> in the sidebar.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Sync live settings into existing detector (no reload needed)
    det = st.session_state.detector
    det.threshold         = threshold
    det.use_face_detection = use_face_det

    # ── Upload ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">📸 Image Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Face Image</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag & drop or browse — JPG, JPEG, PNG, BMP",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0;color:var(--text-dim);
                    font-size:.85rem;letter-spacing:.08em;">
          ↑  Drop an image to begin real-time liveness analysis
        </div>
        """, unsafe_allow_html=True)
    else:
        raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
        img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)

        if img_bgr is None:
            st.error("Could not decode image — please try another file.")
            return

        col_left, col_right = st.columns(2, gap="medium")

        with col_left:
            st.markdown('<div class="section-label">Original</div>',
                        unsafe_allow_html=True)
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                     use_column_width=True)

        detect_btn = st.button("🔍 Run Detection", type="primary",
                               use_container_width=True)

        if detect_btn:
            with st.spinner("🔬 Scanning biometric signature…"):
                time.sleep(0.3)   # brief UX pause for animation feel
                try:
                    cls, confidence, color, bbox, face_conf = \
                        det.predict_frame(img_bgr)
                except Exception as exc:
                    st.error(f"Detection error: {exc}")
                    return

            # Update stats
            st.session_state.total_predictions += 1
            if cls == "REAL":
                st.session_state.real_count += 1
            elif cls == "SPOOF":
                st.session_state.spoof_count += 1

            # Save history
            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "filename":  uploaded.name[:28],
                "cls":       cls,
                "confidence": confidence,
            })

            # Draw overlay
            result_img = draw_overlay(img_bgr, cls, confidence, bbox, color)

            with col_right:
                st.markdown('<div class="section-label">Detection Result</div>',
                            unsafe_allow_html=True)
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                         use_column_width=True)

            # Result panel (full width below)
            st.markdown(result_panel(cls, confidence, face_conf),
                        unsafe_allow_html=True)

            if cls == "REAL":
                st.balloons()

    # ── History ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🕒 Detection History</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-title">Activity Log</div>',
                unsafe_allow_html=True)
    render_history()

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="cyber-footer">
      <div class="footer-dev">⚡ Developed by Siva</div>
      <div class="footer-copy">
        AI/ML Final Year Project · BioSentinel v2.0 ·
        © 2025 All rights reserved
      </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
