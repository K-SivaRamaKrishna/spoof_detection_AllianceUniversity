# app.py — BioSentinel AI · Face Liveness Detection
# Author: Developed by Siva | AI/ML Final Year Project
# Features: Persistent history, warm skin-tone palette, polished UI

import streamlit as st
import cv2
import numpy as np
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# ── Bootstrap ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCRIPT_DIR    = Path(__file__).parent.absolute()
DEFAULT_MODEL = str(SCRIPT_DIR / "best_model.h5")
HISTORY_FILE  = str(SCRIPT_DIR / "detection_history.json")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioSentinel AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS — WARM SKIN-TONE PALETTE · ORGANIC LUXURY · BIOMETRIC EDITORIAL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
  /* Skin-tone warm palette */
  --sand:     #f5ede0;
  --linen:    #ede0d4;
  --blush:    #d4a898;
  --copper:   #b5704f;
  --sienna:   #8b4a2f;
  --umber:    #5c3317;
  --walnut:   #3a1f0d;

  /* Accent & UI */
  --teal:     #2d7d6e;
  --teal-lt:  #4aab98;
  --teal-glow:rgba(45,125,110,0.22);
  --danger:   #c0392b;
  --danger-lt:#e74c3c;
  --gold:     #c9933a;
  --gold-lt:  #e8b45a;

  /* Surfaces */
  --bg:       #f7efe5;
  --panel:    rgba(255,248,240,0.88);
  --panel-dk: rgba(240,228,215,0.75);
  --border:   rgba(181,112,79,0.25);
  --border-md:rgba(181,112,79,0.40);

  /* Text */
  --tx-h:     #2a1208;
  --tx-b:     #4a2c18;
  --tx-m:     #7a5038;
  --tx-l:     #a07858;
  --tx-xl:    #c8a888;
}

/* ── BASE ── */
html, body, [class*="css"], .stApp {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--tx-b) !important;
}

.stApp {
  background:
    radial-gradient(ellipse 100% 55% at 0% 0%,   rgba(181,112,79,.13) 0%,transparent 55%),
    radial-gradient(ellipse 80%  70% at 100% 100%,rgba(201,147,58,.10) 0%,transparent 60%),
    radial-gradient(ellipse 60%  80% at 50%  40%,rgba(212,168,152,.08) 0%,transparent 65%)
    var(--bg) !important;
  background-attachment: fixed !important;
}

/* Subtle grain texture via repeating pattern */
.stApp::before {
  content: '';
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
  opacity: .55;
}

/* ── HERO ── */
.hero {
  position: relative; text-align: center;
  padding: 3.5rem 2rem 3rem; margin-bottom: 2.5rem;
  background: linear-gradient(135deg,
    rgba(255,250,245,.92) 0%,
    rgba(240,225,210,.88) 50%,
    rgba(232,215,200,.92) 100%);
  border: 1px solid var(--border-md);
  border-radius: 28px;
  box-shadow: 0 8px 60px rgba(90,40,10,.10), 0 2px 12px rgba(90,40,10,.06);
  overflow: hidden;
  animation: heroIn .8s cubic-bezier(.22,1,.36,1) forwards;
}
@keyframes heroIn { from{opacity:0;transform:translateY(-20px)} to{opacity:1;transform:translateY(0)} }

.hero::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--copper), var(--gold), var(--teal-lt), var(--blush), var(--copper));
  background-size: 300% auto;
  animation: gradBar 5s linear infinite;
}
@keyframes gradBar { 0%{background-position:0%} 100%{background-position:300%} }

.hero::after {
  content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--border-md), transparent);
}

.hero-badge {
  display: inline-flex; align-items: center; gap: .5rem;
  background: var(--teal); color: #fff;
  font-family: 'JetBrains Mono', monospace; font-size: .72rem; font-weight: 700;
  letter-spacing: .22em; text-transform: uppercase;
  padding: .38rem 1.3rem; border-radius: 50px;
  box-shadow: 0 4px 18px rgba(45,125,110,.35);
  margin-bottom: 1.4rem;
}
.badge-dot {
  width: 8px; height: 8px; border-radius: 50%; background: #a0ffee;
  animation: blink 1.6s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.3;transform:scale(1.5)} }

.hero-title {
  font-family: 'Playfair Display', serif !important;
  font-size: clamp(2.2rem, 5.5vw, 4rem) !important;
  font-weight: 900 !important; line-height: 1.08 !important;
  color: var(--walnut) !important;
  letter-spacing: -.01em !important;
  margin: .3rem 0 .5rem !important;
}
.hero-title span {
  background: linear-gradient(135deg, var(--copper), var(--sienna));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-sub {
  font-size: 1.05rem; color: var(--tx-m); letter-spacing: .06em;
  font-weight: 400; margin-top: .4rem; max-width: 500px; margin-inline: auto;
}

/* ── SECTION LABELS ── */
.sec-tag {
  font-family: 'JetBrains Mono', monospace; font-size: .67rem; font-weight: 700;
  letter-spacing: .3em; color: var(--copper); text-transform: uppercase;
  margin-bottom: .35rem;
}
.sec-h {
  font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 700;
  color: var(--walnut); margin-bottom: 1.1rem; line-height: 1.2;
}

/* ── CARDS ── */
.gc {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 1.8rem 2rem;
  box-shadow: 0 4px 32px rgba(90,40,10,.07), 0 1px 4px rgba(90,40,10,.04);
  backdrop-filter: blur(16px);
  transition: border-color .3s, box-shadow .3s;
}
.gc:hover {
  border-color: var(--border-md);
  box-shadow: 0 8px 48px rgba(90,40,10,.12);
}

/* ── RESULT CARDS ── */
.r-real {
  background: linear-gradient(135deg, rgba(45,125,110,.10), rgba(74,171,152,.06));
  border: 2px solid rgba(45,125,110,.50);
  border-radius: 22px; padding: 2.2rem; text-align: center;
  box-shadow: 0 0 50px rgba(45,125,110,.15), inset 0 0 40px rgba(45,125,110,.04);
  animation: rFade .5s cubic-bezier(.22,1,.36,1) forwards;
}
.r-spoof {
  background: linear-gradient(135deg, rgba(192,57,43,.10), rgba(231,76,60,.06));
  border: 2px solid rgba(192,57,43,.50);
  border-radius: 22px; padding: 2.2rem; text-align: center;
  box-shadow: 0 0 50px rgba(192,57,43,.15), inset 0 0 40px rgba(192,57,43,.04);
  animation: rFade .5s cubic-bezier(.22,1,.36,1) forwards;
}
.r-none {
  background: rgba(160,120,88,.07); border: 1px solid rgba(160,120,88,.25);
  border-radius: 22px; padding: 2.2rem; text-align: center;
  animation: rFade .5s cubic-bezier(.22,1,.36,1) forwards;
}
@keyframes rFade { from{opacity:0;transform:scale(.94)} to{opacity:1;transform:scale(1)} }

.r-icon  { font-size: 5rem; display: block; margin-bottom: .6rem; }
.r-lbl   { font-family: 'Playfair Display', serif; font-size: 2.4rem; font-weight: 900; letter-spacing: .03em; margin: .3rem 0; }
.r-lbl.g { color: var(--teal); }
.r-lbl.r { color: var(--danger); }
.r-lbl.x { color: var(--tx-l); }
.r-sub   { font-size: .95rem; color: var(--tx-m); letter-spacing: .08em; margin-top: .3rem; }

/* ── CONFIDENCE BAR ── */
.cb-wrap { margin-top: 1.4rem; }
.cb-top  { display: flex; justify-content: space-between; font-size: .82rem; color: var(--tx-m); margin-bottom: .5rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 500; }
.cb-num  { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 700; }
.cb-num.g { color: var(--teal); } .cb-num.r { color: var(--danger); } .cb-num.x { color: var(--tx-l); }
.cb-track { height: 14px; border-radius: 14px; background: rgba(90,40,10,.09); overflow: hidden; }
.cb-g { height:100%; border-radius:14px; background: linear-gradient(90deg, var(--teal), var(--teal-lt)); box-shadow: 0 0 16px rgba(45,125,110,.45); }
.cb-r { height:100%; border-radius:14px; background: linear-gradient(90deg, var(--danger), var(--danger-lt)); box-shadow: 0 0 16px rgba(192,57,43,.45); }
.cb-x { height:100%; border-radius:14px; background: linear-gradient(90deg, var(--tx-xl), var(--tx-l)); }

/* ── STAT MINI CARDS ── */
.sc-row { display: flex; gap: .8rem; margin: .8rem 0; flex-wrap: wrap; }
.sc {
  flex: 1; min-width: 72px; text-align: center;
  background: rgba(255,248,240,.7); border: 1px solid var(--border);
  border-radius: 14px; padding: .9rem .4rem;
  transition: all .25s ease;
}
.sc:hover { background: rgba(255,248,240,.95); border-color: var(--copper); transform: translateY(-3px); box-shadow: 0 6px 20px rgba(90,40,10,.09); }
.sc-v { font-family: 'Playfair Display', serif; font-size: 1.9rem; font-weight: 900; line-height: 1; }
.sc-v.b { color: var(--copper); }
.sc-v.g { color: var(--teal); }
.sc-v.r { color: var(--danger); }
.sc-v.p { color: var(--gold); }
.sc-l { font-size: .62rem; color: var(--tx-l); letter-spacing: .14em; text-transform: uppercase; margin-top: .35rem; font-weight: 600; }

/* ── HISTORY TABLE ── */
.ht { width: 100%; border-collapse: collapse; font-size: .88rem; }
.ht th {
  font-family: 'JetBrains Mono', monospace; font-size: .62rem; letter-spacing: .24em;
  color: var(--copper); text-transform: uppercase; padding: .85rem 1rem;
  border-bottom: 1px solid var(--border-md); text-align: left; background: rgba(245,237,224,.5);
}
.ht td { padding: .7rem 1rem; border-bottom: 1px solid rgba(181,112,79,.08); color: var(--tx-b); vertical-align: middle; }
.ht tr:hover td { background: rgba(181,112,79,.05); }
.ht tr:last-child td { border-bottom: none; }

/* history badges */
.bg-r { display:inline-flex;align-items:center;gap:.35rem;padding:.24rem .85rem;border-radius:50px;font-size:.72rem;font-weight:700;letter-spacing:.1em;background:rgba(45,125,110,.12);border:1px solid rgba(45,125,110,.4);color:var(--teal); }
.bg-s { display:inline-flex;align-items:center;gap:.35rem;padding:.24rem .85rem;border-radius:50px;font-size:.72rem;font-weight:700;letter-spacing:.1em;background:rgba(192,57,43,.10);border:1px solid rgba(192,57,43,.4);color:var(--danger); }
.bg-n { display:inline-flex;align-items:center;gap:.35rem;padding:.24rem .85rem;border-radius:50px;font-size:.72rem;font-weight:700;letter-spacing:.1em;background:rgba(160,120,88,.10);border:1px solid rgba(160,120,88,.3);color:var(--tx-m); }

/* ── THUMBNAIL in history ── */
.hist-thumb { width:44px;height:44px;object-fit:cover;border-radius:8px;border:1px solid var(--border); }
.hist-thumb-ph { width:44px;height:44px;border-radius:8px;border:1px dashed var(--border);display:flex;align-items:center;justify-content:center;font-size:1.2rem; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f0e4d4 0%, #e8d8c4 100%) !important;
  border-right: 1px solid var(--border-md) !important;
}
.sb-logo { text-align:center; padding:1.5rem 0 1.2rem; border-bottom:1px solid var(--border); margin-bottom:1.2rem; }
.sb-logo-i { font-size:3.2rem; display:block; filter:drop-shadow(0 4px 12px rgba(90,40,10,.25)); }
.sb-logo-t { font-family:'Playfair Display',serif; font-size:.82rem; font-weight:700; letter-spacing:.18em; color:var(--walnut); text-transform:uppercase; margin-top:.5rem; }
.sb-logo-s { font-family:'JetBrains Mono',monospace; font-size:.62rem; color:var(--tx-m); letter-spacing:.14em; margin-top:.25rem; }
.sb-sec { font-family:'JetBrains Mono',monospace; font-size:.60rem; font-weight:700; letter-spacing:.28em; color:var(--copper); text-transform:uppercase; padding:.9rem 0 .45rem; border-top:1px solid var(--border); margin-top:1rem; }
.sb-ok { text-align:center; margin-top:.7rem; padding:.6rem; border-radius:12px; background:rgba(45,125,110,.10); border:1px solid rgba(45,125,110,.35); font-size:.76rem; color:var(--teal); font-family:'JetBrains Mono',monospace; letter-spacing:.1em; font-weight:700; }

/* ── BUTTONS ── */
.stButton > button {
  width: 100%;
  font-family: 'DM Sans', sans-serif !important;
  font-size: .84rem !important; font-weight: 600 !important;
  letter-spacing: .06em !important;
  border-radius: 12px !important; padding: .72rem 1.2rem !important;
  transition: all .25s ease !important;
  border: 1px solid var(--border-md) !important;
  background: rgba(255,248,240,.85) !important;
  color: var(--tx-b) !important;
  box-shadow: 0 2px 10px rgba(90,40,10,.07) !important;
}
.stButton > button:hover {
  border-color: var(--copper) !important;
  background: rgba(255,248,240,.98) !important;
  box-shadow: 0 4px 20px rgba(90,40,10,.13) !important;
  transform: translateY(-2px) !important;
  color: var(--walnut) !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--sienna), var(--copper)) !important;
  color: #fff !important; border: none !important;
  font-size: .9rem !important; font-weight: 700 !important;
  box-shadow: 0 6px 28px rgba(139,74,47,.35) !important;
  letter-spacing: .08em !important;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 10px 40px rgba(139,74,47,.48) !important;
  transform: translateY(-3px) !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] > div > div > div {
  background: linear-gradient(90deg, var(--copper), var(--gold)) !important;
}

/* ── TEXT INPUT ── */
[data-testid="stTextInput"] input {
  background: rgba(255,248,240,.9) !important;
  border: 1px solid var(--border-md) !important;
  color: var(--tx-b) !important; border-radius: 10px !important;
  font-family: 'JetBrains Mono', monospace !important; font-size: .82rem !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
  background: rgba(255,248,240,.6) !important;
  border: 2px dashed var(--border-md) !important;
  border-radius: 18px !important; transition: all .3s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--copper) !important;
  background: rgba(255,248,240,.9) !important;
  box-shadow: 0 0 28px rgba(181,112,79,.12) !important;
}

/* ── CAMERA INPUT ── */
[data-testid="stCameraInput"] {
  background: rgba(255,248,240,.6) !important;
  border: 2px dashed var(--border-md) !important;
  border-radius: 18px !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] { border-radius: 14px !important; }

/* ── RADIO ── */
[data-testid="stRadio"] label { color: var(--tx-b) !important; font-weight: 500; }

/* ── CHECKBOX ── */
[data-testid="stCheckbox"] span { color: var(--tx-b) !important; }

/* ── METRICS ── */
[data-testid="metric-container"] {
  background: rgba(255,248,240,.8) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
}

/* ── DIVIDER ── */
.warm-divider {
  height: 1px; margin: 2rem 0;
  background: linear-gradient(90deg, transparent, var(--border-md), transparent);
}

/* ── FOOTER ── */
.footer { text-align:center; margin-top:4rem; padding:2rem; position:relative; }
.footer::before {
  content:''; position:absolute; top:0; left:50%; transform:translateX(-50%);
  width:220px; height:1px;
  background: linear-gradient(90deg, transparent, var(--copper), var(--gold), transparent);
}
.footer-dev { font-family:'Playfair Display',serif; font-size:1rem; font-weight:700; color:var(--walnut); letter-spacing:.08em; }
.footer-copy { font-size:.76rem; color:var(--tx-l); margin-top:.35rem; letter-spacing:.1em; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--linen); }
::-webkit-scrollbar-thumb { background: linear-gradient(var(--blush), var(--copper)); border-radius: 10px; }

/* ── MISC ── */
.stMarkdown p { color: var(--tx-b) !important; }
code { background: rgba(181,112,79,.12) !important; color: var(--sienna) !important; border-radius:4px !important; padding: .1rem .35rem !important; font-family:'JetBrains Mono',monospace !important; }

/* ── HISTORY EMPTY STATE ── */
.empty-hist {
  text-align:center; padding:2.5rem 1rem; color:var(--tx-l);
  font-size:.92rem; letter-spacing:.08em;
}
.empty-hist .big { font-size:3rem; display:block; margin-bottom:.8rem; opacity:.5; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PERSISTENT HISTORY — JSON FILE BASED
# ─────────────────────────────────────────────────────────────────────────────
def _load_history() -> list:
    """Load detection history from JSON file. Returns list of records."""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def _save_history(history: list) -> None:
    """Persist history list to JSON file."""
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history[-200:], f, indent=2)  # keep last 200 entries
    except Exception:
        pass


def _append_history(record: dict) -> None:
    """Append one record to persistent history and session state."""
    st.session_state.history.append(record)
    _save_history(st.session_state.history)


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE — initialise once per session
# ─────────────────────────────────────────────────────────────────────────────
if "history_loaded" not in st.session_state:
    loaded = _load_history()
    st.session_state.history        = loaded
    st.session_state.total          = len(loaded)
    st.session_state.real           = sum(1 for e in loaded if e.get("cls") == "REAL")
    st.session_state.spoof          = sum(1 for e in loaded if e.get("cls") == "SPOOF")
    st.session_state.detector       = None
    st.session_state.history_loaded = True


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTOR — cached
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_detector(model_path: str, threshold: float, use_face: bool):
    from predict_live import LiveSpoofDetector
    return LiveSpoofDetector(
        model_path=model_path,
        threshold=threshold,
        use_face_detection=use_face,
        smooth_frames=5,
        camera_id=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  HTML HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _conf_bar(conf: float, cls: str) -> str:
    nc = "g" if cls == "REAL" else ("r" if cls == "SPOOF" else "x")
    fc = "cb-g" if cls == "REAL" else ("cb-r" if cls == "SPOOF" else "cb-x")
    return f"""
    <div class="cb-wrap">
      <div class="cb-top">
        <span>Confidence Score</span>
        <span class="cb-num {nc}">{conf:.1f}%</span>
      </div>
      <div class="cb-track">
        <div class="{fc}" style="width:{conf:.1f}%; transition: width .8s ease;"></div>
      </div>
    </div>"""


def _result_card(cls: str, conf: float, face_conf: float) -> str:
    M = {
        "REAL":    ("r-real",  "✅",  "GENUINE FACE",   "g"),
        "SPOOF":   ("r-spoof", "🚨",  "SPOOF DETECTED", "r"),
        "NO_FACE": ("r-none",  "🔍",  "NO FACE FOUND",  "x"),
    }
    card, icon, lbl, nc = M.get(cls, M["NO_FACE"])
    conf_s = f"{conf:.1f}%" if cls != "NO_FACE" else "N/A"
    fc_s   = f"{face_conf:.3f}" if face_conf > 0 else "—"
    bar    = _conf_bar(conf, cls) if cls != "NO_FACE" else ""
    detail = f"""
    <div style="margin-top:1.2rem;display:flex;gap:1.5rem;justify-content:center;flex-wrap:wrap;">
      <div style="text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;color:var(--tx-l);letter-spacing:.2em;text-transform:uppercase;margin-bottom:.25rem;">Model Confidence</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:900;color:var(--tx-b);">{conf_s}</div>
      </div>
      <div style="width:1px;background:var(--border);"></div>
      <div style="text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:.62rem;color:var(--tx-l);letter-spacing:.2em;text-transform:uppercase;margin-bottom:.25rem;">Face Score</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:900;color:var(--tx-b);">{fc_s}</div>
      </div>
    </div>"""
    return f"""
    <div class="{card}">
      <span class="r-icon">{icon}</span>
      <div class="r-lbl {nc}">{lbl}</div>
      {bar}
      {detail}
    </div>"""


def _history_badge(cls: str) -> str:
    icons = {"REAL": "●", "SPOOF": "▲"}
    icon  = icons.get(cls, "○")
    return {
        "REAL":    f'<span class="bg-r">{icon} REAL</span>',
        "SPOOF":   f'<span class="bg-s">{icon} SPOOF</span>',
    }.get(cls, f'<span class="bg-n">○ NO FACE</span>')


# ─────────────────────────────────────────────────────────────────────────────
#  CV2 OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
def _draw(img: np.ndarray, cls: str, conf: float, bbox, color) -> np.ndarray:
    out = img.copy()
    if bbox is not None:
        x, y, w, h = bbox
        # Soft rounded-corner bracket style
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 1)
        cl = 24
        for px, py, dx, dy in [
            (x,   y,   cl, 0), (x,   y,   0, cl),
            (x+w, y,  -cl, 0), (x+w, y,   0, cl),
            (x,   y+h, cl, 0), (x,   y+h, 0,-cl),
            (x+w, y+h,-cl, 0), (x+w, y+h, 0,-cl),
        ]:
            cv2.line(out, (px, py), (px+dx, py+dy), color, 3)

    lbl = f"REAL  {conf:.1f}%" if cls == "REAL" else (f"SPOOF  {conf:.1f}%" if cls == "SPOOF" else "NO FACE")
    bg  = (45, 165, 120) if cls == "REAL" else ((60, 60, 210) if cls == "SPOOF" else (130, 110, 90))
    tw  = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)[0][0]
    cv2.rectangle(out, (6, 6), (tw + 26, 48), (30, 20, 12), -1)
    cv2.rectangle(out, (6, 6), (tw + 26, 48), bg, 2)
    cv2.putText(out, lbl, (14, 36), cv2.FONT_HERSHEY_DUPLEX, 0.75, bg, 1)

    if cls != "NO_FACE":
        bx, by, bw, bh = 6, 54, 240, 12
        cv2.rectangle(out, (bx, by), (bx+bw, by+bh), (50, 40, 30), -1)
        cv2.rectangle(out, (bx, by), (bx + int(bw * conf / 100), by+bh), bg, -1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sb-logo">
          <span class="sb-logo-i">🛡️</span>
          <div class="sb-logo-t">BioSentinel AI</div>
          <div class="sb-logo-s">v2.0 · LIVENESS DETECTION</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sb-sec">⚙ Configuration</div>', unsafe_allow_html=True)
        model_path = st.text_input("Model Path", value=DEFAULT_MODEL)
        threshold  = st.slider("Detection Threshold", 0.0, 1.0, 0.50, 0.01)
        use_face   = st.checkbox("Enable Face Detection (MTCNN)", value=True)
        init_btn   = st.button("⚡  Initialize System", use_container_width=True)

        if st.session_state.detector:
            st.markdown('<div class="sb-ok">✓ SYSTEM OPERATIONAL</div>', unsafe_allow_html=True)
        elif os.path.exists(model_path):
            st.info("Model found — click **Initialize System**")
        else:
            st.error("Model not found at specified path")

        # ── Stats
        t = st.session_state.total
        r = st.session_state.real
        s = st.session_state.spoof
        pct = f"{r/t*100:.0f}%" if t else "—"
        st.markdown('<div class="sb-sec">📊 Session Statistics</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sc-row">
          <div class="sc"><div class="sc-v b">{t}</div><div class="sc-l">Total</div></div>
          <div class="sc"><div class="sc-v g">{r}</div><div class="sc-l">Real</div></div>
          <div class="sc"><div class="sc-v r">{s}</div><div class="sc-l">Spoof</div></div>
          <div class="sc"><div class="sc-v p">{pct}</div><div class="sc-l">Real%</div></div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("↺ Clear Stats", use_container_width=True):
                st.session_state.update(total=0, real=0, spoof=0)
                st.success("Stats cleared!")
        with col2:
            if st.button("🗑 Clear History", use_container_width=True):
                st.session_state.history = []
                _save_history([])
                st.success("History cleared!")

        # ── Model info
        st.markdown('<div class="sb-sec">ℹ Model Info</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:.82rem;color:var(--tx-m);line-height:2;">
          <b style="color:var(--walnut)">Architecture:</b> MobileNetV3<br>
          <b style="color:var(--walnut)">Face Detect:</b> MTCNN<br>
          <b style="color:var(--walnut)">Framework:</b> TensorFlow 2.13<br>
          <b style="color:var(--walnut)">Interface:</b> Streamlit 1.28+<br>
          <b style="color:var(--walnut)">Python:</b> 3.10.20
        </div>""", unsafe_allow_html=True)

        # ── History persistence notice
        st.markdown('<div class="sb-sec">💾 Storage</div>', unsafe_allow_html=True)
        hist_path = HISTORY_FILE
        if os.path.exists(hist_path):
            st.markdown(f"""<div style="font-size:.72rem;color:var(--tx-m);font-family:'JetBrains Mono',monospace;word-break:break-all;">
            ✓ History saved to:<br><span style="color:var(--copper)">detection_history.json</span><br>
            <span style="color:var(--tx-l)">{len(st.session_state.history)} records stored</span></div>""",
                        unsafe_allow_html=True)
        else:
            st.markdown("""<div style="font-size:.72rem;color:var(--tx-l);">
            History will be saved after first detection.</div>""", unsafe_allow_html=True)

    return model_path, threshold, use_face, init_btn


# ─────────────────────────────────────────────────────────────────────────────
#  HISTORY TABLE (with thumbnails)
# ─────────────────────────────────────────────────────────────────────────────
def _history_table():
    h = st.session_state.history
    if not h:
        st.markdown("""
        <div class="empty-hist">
          <span class="big">🔍</span>
          No detections recorded yet.<br>
          Run an analysis above to begin building your history.
        </div>""", unsafe_allow_html=True)
        return

    # Pagination
    page_size = 15
    total_pages = max(1, (len(h) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=total_pages, step=1, label_visibility="collapsed")
    start = (page - 1) * page_size
    end   = min(start + page_size, len(h))
    slice_ = list(reversed(h))[start:end]

    rows = ""
    for i, e in enumerate(slice_):
        global_idx = len(h) - start - i
        thumb_html = ""
        if e.get("thumb_b64"):
            thumb_html = f'<img class="hist-thumb" src="data:image/jpeg;base64,{e["thumb_b64"]}" />'
        else:
            thumb_html = '<div class="hist-thumb-ph">🖼</div>'

        conf_val = f"{e['conf']:.1f}%" if e.get('cls') != 'NO_FACE' and e.get('conf') else "—"
        date_str = e.get("date", "")
        time_str = e.get("ts", "")
        dt_str   = f"{date_str}<br><span style='font-size:.72rem;color:var(--tx-l)'>{time_str}</span>" if date_str else time_str

        rows += f"""
        <tr>
          <td style="color:var(--tx-l);font-family:'JetBrains Mono',monospace;font-size:.76rem;">#{global_idx}</td>
          <td>{thumb_html}</td>
          <td style="font-size:.78rem;color:var(--tx-m);">{dt_str}</td>
          <td style="font-family:'JetBrains Mono',monospace;font-size:.76rem;color:var(--tx-b);max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{e.get('src','—')[:30]}</td>
          <td>{_history_badge(e.get('cls','NO_FACE'))}</td>
          <td style="font-family:'JetBrains Mono',monospace;color:var(--copper);font-weight:700;">{conf_val}</td>
        </tr>"""

    pagination = f"""<div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--tx-l);
      text-align:right;padding:.6rem 1rem;border-top:1px solid var(--border);">
      Showing {start+1}–{end} of {len(h)} records
    </div>"""

    st.markdown(f"""
    <div class="gc" style="padding:0;overflow:hidden;">
      <div style="overflow-x:auto;">
        <table class="ht">
          <thead>
            <tr>
              <th>#</th><th>Preview</th><th>Timestamp</th>
              <th>Source</th><th>Result</th><th>Confidence</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
      {pagination}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTION RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def _img_to_b64_thumb(img_bgr: np.ndarray, size: int = 80) -> str:
    """Convert BGR image to a small base64 JPEG thumbnail for history storage."""
    import base64
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    thumb = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _run_detection(img_bgr: np.ndarray, source_label: str,
                   threshold: float, use_face: bool):
    det = st.session_state.detector
    det.threshold          = threshold
    det.use_face_detection = use_face

    with st.spinner("🔬  Analysing biometric signature…"):
        time.sleep(0.25)
        try:
            cls, conf, color, bbox, face_conf = det.predict_frame(img_bgr)
        except Exception as exc:
            st.error(f"Detection error: {exc}")
            return

    # ── Counters
    st.session_state.total += 1
    if cls == "REAL":
        st.session_state.real  += 1
    elif cls == "SPOOF":
        st.session_state.spoof += 1

    # ── Thumbnail
    thumb_b64 = _img_to_b64_thumb(img_bgr)

    # ── Persist history record
    now = datetime.now()
    record = {
        "ts":        now.strftime("%H:%M:%S"),
        "date":      now.strftime("%d %b %Y"),
        "src":       source_label[:40],
        "cls":       cls,
        "conf":      round(float(conf), 2),
        "face_conf": round(float(face_conf), 4),
        "thumb_b64": thumb_b64,
    }
    _append_history(record)

    # ── Draw annotated frame
    result_img = _draw(img_bgr, cls, conf, bbox, color)

    # ── Display
    st.markdown('<div class="warm-divider"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="sec-tag">📷 Input Frame</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
    with c2:
        st.markdown('<div class="sec-tag">🔍 Detection Output</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(_result_card(cls, conf, face_conf), unsafe_allow_html=True)

    if cls == "REAL":
        st.balloons()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Hero
    st.markdown("""
    <div class="hero">
      <div class="hero-badge">
        <span class="badge-dot"></span>
        System Active · BioSentinel v2.0
      </div>
      <div class="hero-title">
        AI <span>Face Liveness</span><br>Detection System
      </div>
      <div class="hero-sub">
        Real-Time Deep Learning Biometric Analysis &amp; Anti-Spoofing
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar
    model_path, threshold, use_face, init_btn = _sidebar()

    # ── Init
    if init_btn:
        if not os.path.exists(model_path):
            st.error(f"Model not found: `{model_path}`")
        else:
            with st.spinner("⚡  Loading neural network weights…"):
                try:
                    _load_detector.clear()
                    det = _load_detector(model_path, threshold, use_face)
                    st.session_state.detector = det
                    st.success("✅  BioSentinel AI is operational — system ready!")
                except Exception as exc:
                    st.error(f"Initialization failed: {exc}")
                    st.session_state.detector = None

    # ── Gate — system not ready
    if st.session_state.detector is None:
        st.markdown("""
        <div class="gc" style="text-align:center;padding:3.5rem 2rem;margin-top:1.5rem;">
          <div style="font-size:4.5rem;margin-bottom:1rem;opacity:.7;">🛡️</div>
          <div class="sec-h" style="margin-bottom:.8rem;">System Offline</div>
          <div style="color:var(--tx-m);font-size:.95rem;line-height:1.9;max-width:500px;margin:0 auto;">
            Place <code>best_model.h5</code> in the same folder as <code>app.py</code>,
            then click <b style="color:var(--copper)">⚡ Initialize System</b> in the sidebar to begin.
          </div>
        </div>""", unsafe_allow_html=True)

        # still show history even when system is offline
        if st.session_state.history:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sec-tag">🕒 Previous Sessions</div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-h">Detection History</div>', unsafe_allow_html=True)
            _history_table()
        return

    # ── Input mode
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-tag">📡 Detection Mode</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-h">Choose Your Input Source</div>', unsafe_allow_html=True)

    mode = st.radio(
        "mode",
        ["📁  Upload Image", "📷  Live Camera"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload branch
    if "Upload" in mode:
        uploaded = st.file_uploader(
            "upload",
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed",
        )
        if uploaded:
            raw     = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error("Could not decode image — please try another file.")
                return
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍  Run Liveness Detection", type="primary", use_container_width=True):
                _run_detection(img_bgr, uploaded.name, threshold, use_face)
        else:
            st.markdown("""
            <div style="text-align:center;padding:1.6rem 0;color:var(--tx-l);font-size:.95rem;letter-spacing:.06em;">
              ↑ &nbsp; Drop or browse a face image to begin analysis
            </div>""", unsafe_allow_html=True)

    # ── Camera branch
    else:
        st.markdown("""
        <div style="color:var(--tx-m);font-size:.9rem;letter-spacing:.05em;margin-bottom:.9rem;line-height:1.7;">
          Click <b style="color:var(--walnut)">Take Photo</b> to capture from your webcam,
          then press <b style="color:var(--walnut)">Run Liveness Detection</b>.
        </div>""", unsafe_allow_html=True)

        cam_img = st.camera_input("camera", label_visibility="collapsed")
        if cam_img:
            raw     = np.frombuffer(cam_img.read(), np.uint8)
            img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error("Could not read camera frame — please try again.")
                return
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍  Run Liveness Detection", type="primary", use_container_width=True):
                _run_detection(img_bgr, "camera_snapshot", threshold, use_face)

    # ── History
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-tag">🕒 Activity Log</div>', unsafe_allow_html=True)

    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown('<div class="sec-h">Detection History</div>', unsafe_allow_html=True)
    with h_col2:
        if st.session_state.history:
            st.markdown(f"""<div style="text-align:right;padding-top:.4rem;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--tx-l);">
            {len(st.session_state.history)} total records</span></div>""", unsafe_allow_html=True)

    _history_table()

    # ── Footer
    st.markdown("""
    <div class="footer">
      <div class="footer-dev">⚡ Developed by Siva</div>
      <div class="footer-copy">
        AI / ML Final Year Project &nbsp;·&nbsp; BioSentinel v2.0 &nbsp;·&nbsp; © 2025 All Rights Reserved
      </div>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
