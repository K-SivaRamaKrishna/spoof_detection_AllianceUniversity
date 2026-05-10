# app.py — BioSentinel AI · Face Liveness Detection
# Fixed: no pkg_resources | Camera + Upload | Bold Presentation UI
# Author: Developed by Siva | AI/ML Final Year Project

import streamlit as st
import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Bootstrap ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCRIPT_DIR    = Path(__file__).parent.absolute()
DEFAULT_MODEL = str(SCRIPT_DIR / "best_model.h5")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioSentinel AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS — BOLD NEON · HIGH-CONTRAST · PRESENTATION-READY
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;700;800&display=swap');

:root{
  --nb:#00e5ff; --np:#c026d3; --nc:#00ffbd;
  --ng:#00ff88; --nr:#ff1744; --ny:#ffd600;
  --bg:#03050f; --panel:rgba(5,15,35,0.92);
  --gb:rgba(0,229,255,0.2); --gbs:rgba(0,229,255,0.08);
  --tm:#ddeeff; --td:#6890b0;
}

html,body,[class*="css"],.stApp{
  background:var(--bg) !important;
  font-family:'Exo 2',sans-serif !important;
  color:var(--tm) !important;
}

.stApp{
  background:
    radial-gradient(ellipse 90% 70% at 15% 5%,rgba(0,229,255,.10) 0%,transparent 55%),
    radial-gradient(ellipse 70% 90% at 85% 90%,rgba(192,38,211,.12) 0%,transparent 55%),
    radial-gradient(ellipse 60% 60% at 50% 50%,rgba(0,255,189,.04) 0%,transparent 65%)
    var(--bg) !important;
  background-attachment:fixed !important;
}
.stApp::after{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:
    linear-gradient(rgba(0,229,255,.04) 1px,transparent 1px),
    linear-gradient(90deg,rgba(0,229,255,.04) 1px,transparent 1px);
  background-size:60px 60px;
}

/* HERO */
.hero{
  position:relative;text-align:center;
  padding:3rem 2rem 2.5rem;margin-bottom:2rem;
  background:linear-gradient(135deg,
    rgba(0,229,255,.10) 0%,rgba(192,38,211,.13) 50%,rgba(0,255,189,.07) 100%);
  border:1px solid rgba(0,229,255,.28);
  border-radius:24px;backdrop-filter:blur(18px);overflow:hidden;
  animation:heroIn .9s ease forwards;
}
@keyframes heroIn{from{opacity:0;transform:translateY(-24px)}to{opacity:1;transform:translateY(0)}}
.hero::before{
  content:'';position:absolute;top:0;left:-100%;
  width:55%;height:2px;
  background:linear-gradient(90deg,transparent,var(--nb),var(--np),transparent);
  animation:sweep 2.8s linear infinite;
}
@keyframes sweep{0%{left:-60%}100%{left:160%}}
.hero-eye{
  font-size:4.5rem;display:block;margin-bottom:.6rem;
  filter:drop-shadow(0 0 24px rgba(0,229,255,.8));
  animation:eyePulse 2.5s ease-in-out infinite;
}
@keyframes eyePulse{
  0%,100%{filter:drop-shadow(0 0 16px rgba(0,229,255,.6))}
  50%    {filter:drop-shadow(0 0 40px rgba(192,38,211,.9))}
}
.hero-title{
  font-family:'Orbitron',monospace !important;
  font-size:clamp(2rem,5vw,3.8rem) !important;
  font-weight:900 !important;line-height:1.05 !important;
  background:linear-gradient(90deg,var(--nb) 0%,var(--np) 50%,var(--nc) 100%);
  -webkit-background-clip:text !important;-webkit-text-fill-color:transparent !important;
  background-clip:text !important;margin:.4rem 0 !important;
  background-size:200% auto;
  animation:shimmer 4s linear infinite;
}
@keyframes shimmer{0%{background-position:0% center}100%{background-position:200% center}}
.hero-sub{
  font-size:1.15rem;color:var(--td);letter-spacing:.18em;
  text-transform:uppercase;margin-top:.5rem;font-weight:600;
}
.hero-pill{
  display:inline-flex;align-items:center;gap:.5rem;margin-top:1.1rem;
  font-family:'Orbitron',monospace;font-size:.78rem;font-weight:700;
  letter-spacing:.2em;color:var(--ng);
  background:rgba(0,255,136,.1);border:1px solid rgba(0,255,136,.35);
  border-radius:50px;padding:.38rem 1.4rem;
  box-shadow:0 0 16px rgba(0,255,136,.2);
}
.blink{width:10px;height:10px;border-radius:50%;
  background:var(--ng);box-shadow:0 0 10px var(--ng);
  animation:blinkDot 1.4s ease-in-out infinite;display:inline-block;}
@keyframes blinkDot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(1.6)}}

/* SECTION LABELS */
.sec-tag{
  font-family:'Orbitron',monospace;font-size:.68rem;font-weight:700;
  letter-spacing:.3em;color:var(--nb);text-transform:uppercase;margin-bottom:.5rem;
}
.sec-h{
  font-family:'Orbitron',monospace;font-size:1.15rem;font-weight:700;
  color:var(--tm);margin-bottom:1rem;
}

/* GLASS CARD */
.gc{
  background:var(--panel);border:1px solid var(--gb);
  border-radius:18px;padding:1.6rem 1.8rem;
  backdrop-filter:blur(20px);position:relative;overflow:hidden;
  transition:border-color .3s,box-shadow .3s;
}
.gc:hover{border-color:rgba(0,229,255,.45);box-shadow:0 0 36px rgba(0,229,255,.13);}

/* RESULT CARDS */
.r-real{
  background:linear-gradient(135deg,rgba(0,255,136,.10),rgba(0,229,255,.06));
  border:2px solid rgba(0,255,136,.55);border-radius:20px;
  padding:2rem;text-align:center;
  box-shadow:0 0 44px rgba(0,255,136,.20),inset 0 0 44px rgba(0,255,136,.05);
  animation:rFade .5s ease forwards;
}
.r-spoof{
  background:linear-gradient(135deg,rgba(255,23,68,.12),rgba(192,38,211,.08));
  border:2px solid rgba(255,23,68,.55);border-radius:20px;
  padding:2rem;text-align:center;
  box-shadow:0 0 44px rgba(255,23,68,.22),inset 0 0 44px rgba(255,23,68,.05);
  animation:rFade .5s ease forwards;
}
.r-none{
  background:rgba(80,80,80,.12);border:1px solid rgba(130,130,130,.3);
  border-radius:20px;padding:2rem;text-align:center;animation:rFade .5s ease forwards;
}
@keyframes rFade{from{opacity:0;transform:scale(.93)}to{opacity:1;transform:scale(1)}}
.r-icon{font-size:4.5rem;margin-bottom:.7rem;display:block;}
.r-lbl{font-family:'Orbitron',monospace;font-size:2.4rem;font-weight:900;letter-spacing:.1em;margin:.3rem 0;}
.r-lbl.g{color:var(--ng);text-shadow:0 0 30px rgba(0,255,136,.65);}
.r-lbl.r{color:var(--nr);text-shadow:0 0 30px rgba(255,23,68,.65);}
.r-lbl.x{color:#888;}
.r-sub{font-size:.98rem;color:var(--td);letter-spacing:.1em;margin-top:.3rem;}

/* CONFIDENCE BAR */
.cb-wrap{margin-top:1.3rem;}
.cb-top{display:flex;justify-content:space-between;font-size:.84rem;color:var(--td);margin-bottom:.45rem;letter-spacing:.1em;text-transform:uppercase;}
.cb-num{font-family:'Orbitron',monospace;font-size:1.2rem;font-weight:800;}
.cb-num.g{color:var(--ng)}.cb-num.r{color:var(--nr)}.cb-num.x{color:#888;}
.cb-track{height:16px;border-radius:16px;background:rgba(255,255,255,.07);overflow:hidden;}
.cb-g{height:100%;border-radius:16px;background:linear-gradient(90deg,#00c853,#00ff88);box-shadow:0 0 20px rgba(0,255,136,.6);}
.cb-r{height:100%;border-radius:16px;background:linear-gradient(90deg,#c62828,#ff1744);box-shadow:0 0 20px rgba(255,23,68,.6);}
.cb-x{height:100%;border-radius:16px;background:linear-gradient(90deg,#555,#888);}

/* STAT CARDS */
.sc-row{display:flex;gap:.9rem;margin:.8rem 0;flex-wrap:wrap;}
.sc{flex:1;min-width:76px;text-align:center;
  background:rgba(0,229,255,.05);border:1px solid rgba(0,229,255,.18);
  border-radius:14px;padding:.9rem .4rem;transition:all .3s;}
.sc:hover{background:rgba(0,229,255,.1);border-color:var(--nb);transform:translateY(-3px);}
.sc-v{font-family:'Orbitron',monospace;font-size:1.8rem;font-weight:900;line-height:1;}
.sc-v.b{color:var(--nb)}.sc-v.g{color:var(--ng)}.sc-v.r{color:var(--nr)}.sc-v.p{color:var(--np)}
.sc-l{font-size:.65rem;color:var(--td);letter-spacing:.12em;text-transform:uppercase;margin-top:.3rem;}

/* HISTORY TABLE */
.ht{width:100%;border-collapse:collapse;font-size:.9rem;}
.ht th{font-family:'Orbitron',monospace;font-size:.65rem;letter-spacing:.22em;color:var(--nb);
  text-transform:uppercase;padding:.8rem .9rem;border-bottom:1px solid rgba(0,229,255,.18);text-align:left;}
.ht td{padding:.65rem .9rem;border-bottom:1px solid rgba(255,255,255,.04);color:var(--tm);vertical-align:middle;}
.ht tr:hover td{background:rgba(0,229,255,.05);}
.bg-r{display:inline-block;padding:.22rem .8rem;border-radius:50px;font-size:.74rem;font-weight:700;letter-spacing:.1em;background:rgba(0,255,136,.12);border:1px solid rgba(0,255,136,.4);color:var(--ng);}
.bg-s{display:inline-block;padding:.22rem .8rem;border-radius:50px;font-size:.74rem;font-weight:700;letter-spacing:.1em;background:rgba(255,23,68,.12);border:1px solid rgba(255,23,68,.4);color:var(--nr);}
.bg-n{display:inline-block;padding:.22rem .8rem;border-radius:50px;font-size:.74rem;font-weight:700;letter-spacing:.1em;background:rgba(130,130,130,.12);border:1px solid rgba(130,130,130,.3);color:#aaa;}

/* SIDEBAR */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#040b1e 0%,#060f28 100%) !important;
  border-right:1px solid rgba(0,229,255,.15) !important;
}
.sb-logo{text-align:center;padding:1.4rem 0 1.1rem;border-bottom:1px solid rgba(0,229,255,.14);margin-bottom:1.3rem;}
.sb-logo-i{font-size:3.5rem;filter:drop-shadow(0 0 20px rgba(0,229,255,.7));animation:eyePulse 2.5s ease-in-out infinite;display:block;}
.sb-logo-t{font-family:'Orbitron',monospace;font-size:.74rem;font-weight:700;letter-spacing:.28em;color:var(--nb);text-transform:uppercase;margin-top:.45rem;}
.sb-sec{font-family:'Orbitron',monospace;font-size:.62rem;font-weight:700;letter-spacing:.26em;color:var(--np);text-transform:uppercase;padding:.9rem 0 .45rem;border-top:1px solid rgba(192,38,211,.15);margin-top:.9rem;}
.sb-ok{text-align:center;margin-top:.6rem;padding:.55rem;border-radius:10px;background:rgba(0,255,136,.08);border:1px solid rgba(0,255,136,.3);font-size:.78rem;color:var(--ng);font-family:'Orbitron',monospace;letter-spacing:.1em;}

/* BUTTONS */
.stButton>button{
  width:100%;font-family:'Orbitron',monospace !important;font-size:.78rem !important;
  font-weight:700 !important;letter-spacing:.15em !important;text-transform:uppercase !important;
  border-radius:12px !important;padding:.72rem 1rem !important;transition:all .3s ease !important;
  border:1px solid rgba(0,229,255,.4) !important;
  background:linear-gradient(135deg,rgba(0,229,255,.10),rgba(192,38,211,.10)) !important;
  color:var(--nb) !important;box-shadow:0 0 14px rgba(0,229,255,.10) !important;
}
.stButton>button:hover{
  border-color:var(--nb) !important;
  background:linear-gradient(135deg,rgba(0,229,255,.22),rgba(192,38,211,.22)) !important;
  box-shadow:0 0 28px rgba(0,229,255,.28) !important;transform:translateY(-2px) !important;
}
.stButton>button[kind="primary"]{
  background:linear-gradient(135deg,#00b4d8,#9d4edd) !important;
  color:#fff !important;border:none !important;font-size:.88rem !important;
  box-shadow:0 0 28px rgba(0,180,216,.4) !important;
}
.stButton>button[kind="primary"]:hover{
  box-shadow:0 0 44px rgba(0,180,216,.6) !important;transform:translateY(-3px) !important;
}

/* SLIDER */
[data-testid="stSlider"]>div>div>div{background:linear-gradient(90deg,var(--nb),var(--np)) !important;}

/* TEXT INPUT */
[data-testid="stTextInput"] input{
  background:rgba(0,229,255,.05) !important;border:1px solid rgba(0,229,255,.25) !important;
  color:var(--tm) !important;border-radius:10px !important;font-family:'Exo 2',sans-serif !important;
}

/* UPLOAD / CAMERA */
[data-testid="stFileUploader"]{
  background:rgba(0,229,255,.04) !important;border:2px dashed rgba(0,229,255,.35) !important;
  border-radius:16px !important;transition:all .3s !important;
}
[data-testid="stFileUploader"]:hover{
  border-color:var(--nb) !important;background:rgba(0,229,255,.08) !important;
  box-shadow:0 0 24px rgba(0,229,255,.15) !important;
}
[data-testid="stCameraInput"]{
  background:rgba(0,229,255,.04) !important;border:2px dashed rgba(0,229,255,.35) !important;
  border-radius:16px !important;
}

/* ALERTS */
[data-testid="stAlert"]{border-radius:14px !important;backdrop-filter:blur(8px) !important;}

/* FOOTER */
.footer{text-align:center;margin-top:3.5rem;padding:1.8rem;position:relative;}
.footer::before{
  content:'';position:absolute;top:0;left:50%;transform:translateX(-50%);
  width:260px;height:1px;
  background:linear-gradient(90deg,transparent,var(--nb),var(--np),transparent);
  box-shadow:0 0 14px var(--nb);
}
.footer-dev{font-family:'Orbitron',monospace;font-size:.88rem;font-weight:800;letter-spacing:.22em;color:var(--nb);text-shadow:0 0 14px rgba(0,229,255,.5);}
.footer-copy{font-size:.78rem;color:var(--td);margin-top:.35rem;letter-spacing:.1em;}

/* SCROLLBAR */
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:rgba(0,0,0,.3);}
::-webkit-scrollbar-thumb{background:linear-gradient(var(--nb),var(--np));border-radius:10px;}

/* MISC */
.stMarkdown p{color:var(--tm) !important;}
[data-testid="metric-container"]{background:rgba(0,229,255,.04) !important;border:1px solid rgba(0,229,255,.15) !important;border-radius:12px !important;}
[data-testid="stCheckbox"] span{color:var(--tm) !important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in dict(total=0, real=0, spoof=0, history=[], detector=None).items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTOR — cached to avoid reloading on every rerun
#  NOTE: we import LiveSpoofDetector INSIDE the cached function.
#        This avoids the 'pkg_resources' error that occurs when tensorflow
#        is imported at module level before pkg_resources is available.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_detector(model_path: str, threshold: float, use_face: bool):
    from predict_live import LiveSpoofDetector  # lazy import — no pkg_resources issue
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
      <div class="cb-top"><span>Confidence Score</span>
        <span class="cb-num {nc}">{conf:.1f}%</span></div>
      <div class="cb-track"><div class="{fc}" style="width:{conf:.1f}%"></div></div>
    </div>"""


def _result_card(cls: str, conf: float, face_conf: float) -> str:
    M = {
        "REAL":    ("r-real",  "🛡️",  "GENUINE FACE",    "g"),
        "SPOOF":   ("r-spoof", "⚠️",  "SPOOF DETECTED",  "r"),
        "NO_FACE": ("r-none",  "🔍",  "NO FACE FOUND",   "x"),
    }
    card, icon, lbl, nc = M.get(cls, M["NO_FACE"])
    conf_s = f"{conf:.1f}%" if cls != "NO_FACE" else "N/A"
    fc_s   = f"{face_conf:.3f}" if face_conf > 0 else "—"
    bar    = _conf_bar(conf, cls) if cls != "NO_FACE" else ""
    return f"""
    <div class="{card}">
      <span class="r-icon">{icon}</span>
      <div class="r-lbl {nc}">{lbl}</div>
      <div class="r-sub">Model Confidence · {conf_s}</div>
      {bar}
      <div style="margin-top:1rem;font-size:.8rem;color:var(--td);letter-spacing:.1em;text-transform:uppercase;">
        Face Detection Score · {fc_s}
      </div>
    </div>"""


def _history_badge(cls: str) -> str:
    return {"REAL": '<span class="bg-r">REAL</span>',
            "SPOOF": '<span class="bg-s">SPOOF</span>'}.get(cls, '<span class="bg-n">NO FACE</span>')


# ─────────────────────────────────────────────────────────────────────────────
#  CV2 OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
def _draw(img: np.ndarray, cls: str, conf: float, bbox, color) -> np.ndarray:
    out = img.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
        cl = 22
        for px,py,dx,dy in [(x,y,cl,0),(x,y,0,cl),(x+w,y,-cl,0),(x+w,y,0,cl),
                             (x,y+h,cl,0),(x,y+h,0,-cl),(x+w,y+h,-cl,0),(x+w,y+h,0,-cl)]:
            cv2.line(out,(px,py),(px+dx,py+dy),color,3)
    lbl = f"REAL {conf:.1f}%" if cls=="REAL" else (f"SPOOF {conf:.1f}%" if cls=="SPOOF" else "NO FACE")
    bg  = (0,210,80) if cls=="REAL" else ((30,30,230) if cls=="SPOOF" else (120,120,120))
    tw  = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.9,2)[0][0]
    cv2.rectangle(out,(6,6),(tw+30,54),(0,0,0),-1)
    cv2.rectangle(out,(6,6),(tw+30,54),bg,2)
    cv2.putText(out,lbl,(14,40),cv2.FONT_HERSHEY_SIMPLEX,0.9,bg,2)
    if cls != "NO_FACE":
        bx,by,bw,bh=6,62,270,14
        cv2.rectangle(out,(bx,by),(bx+bw,by+bh),(50,50,50),-1)
        cv2.rectangle(out,(bx,by),(bx+int(bw*conf/100),by+bh),bg,-1)
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
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sb-sec">⚙ Configuration</div>', unsafe_allow_html=True)
        model_path = st.text_input("Model Path", value=DEFAULT_MODEL)
        threshold  = st.slider("Detection Threshold", 0.0, 1.0, 0.50, 0.01)
        use_face   = st.checkbox("Enable Face Detection (MTCNN)", value=True)
        init_btn   = st.button("⚡ Initialize System", use_container_width=True)

        if st.session_state.detector:
            st.markdown('<div class="sb-ok">✓ SYSTEM READY</div>', unsafe_allow_html=True)
        elif os.path.exists(model_path):
            st.info("Model found — click **Initialize System**")
        else:
            st.error("Model not found at specified path")

        t, r, s = st.session_state.total, st.session_state.real, st.session_state.spoof
        pct = f"{r/t*100:.0f}%" if t else "—"
        st.markdown('<div class="sb-sec">📊 Statistics</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sc-row">
          <div class="sc"><div class="sc-v b">{t}</div><div class="sc-l">Total</div></div>
          <div class="sc"><div class="sc-v g">{r}</div><div class="sc-l">Real</div></div>
          <div class="sc"><div class="sc-v r">{s}</div><div class="sc-l">Spoof</div></div>
          <div class="sc"><div class="sc-v p">{pct}</div><div class="sc-l">Real%</div></div>
        </div>""", unsafe_allow_html=True)

        if st.button("↺ Reset Stats", use_container_width=True):
            st.session_state.update(total=0, real=0, spoof=0, history=[])
            st.success("Stats cleared!")

        st.markdown('<div class="sb-sec">ℹ Model Info</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:.83rem;color:var(--td);line-height:1.9;">
          <b style="color:var(--nb)">Architecture:</b> MobileNetV3<br>
          <b style="color:var(--nb)">Face Detect:</b> MTCNN<br>
          <b style="color:var(--nb)">Framework:</b> TensorFlow 2.13<br>
          <b style="color:var(--nb)">Interface:</b> Streamlit 1.28+
        </div>""", unsafe_allow_html=True)
    return model_path, threshold, use_face, init_btn


# ─────────────────────────────────────────────────────────────────────────────
#  HISTORY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def _history_table():
    h = st.session_state.history
    if not h:
        st.markdown("""<div style="text-align:center;padding:1.8rem;color:var(--td);
          font-size:.9rem;letter-spacing:.1em;">No detections yet · run an analysis above</div>""",
                    unsafe_allow_html=True)
        return
    rows = "".join(f"""
      <tr>
        <td style="color:var(--td);font-size:.8rem;">#{len(h)-i}</td>
        <td style="font-size:.8rem;color:var(--td);">{e['ts']}</td>
        <td style="font-family:'Orbitron',monospace;font-size:.78rem;">{e['src']}</td>
        <td>{_history_badge(e['cls'])}</td>
        <td style="font-family:'Orbitron',monospace;color:var(--nb);">
          {f"{e['conf']:.1f}%" if e['cls'] != 'NO_FACE' else '—'}
        </td>
      </tr>""" for i,e in enumerate(reversed(h[-25:])))
    st.markdown(f"""
    <div class="gc" style="overflow-x:auto;">
      <table class="ht">
        <thead><tr><th>#</th><th>Time</th><th>Source</th><th>Result</th><th>Confidence</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  DETECTION RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def _run_detection(img_bgr: np.ndarray, source_label: str,
                   threshold: float, use_face: bool):
    det = st.session_state.detector
    det.threshold          = threshold
    det.use_face_detection = use_face

    with st.spinner("🔬 Analysing biometric signature…"):
        time.sleep(0.25)
        try:
            cls, conf, color, bbox, face_conf = det.predict_frame(img_bgr)
        except Exception as exc:
            st.error(f"Detection error: {exc}")
            return

    st.session_state.total += 1
    if cls == "REAL":
        st.session_state.real += 1
    elif cls == "SPOOF":
        st.session_state.spoof += 1
    st.session_state.history.append(
        {"ts": datetime.now().strftime("%H:%M:%S"),
         "src": source_label[:28], "cls": cls, "conf": conf})

    result_img = _draw(img_bgr, cls, conf, bbox, color)

    c1, c2 = st.columns(2, gap="medium")
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
    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <span class="hero-eye">🛡️</span>
      <div class="hero-title">AI Face Spoof Detection System</div>
      <div class="hero-sub">Real-Time Deep Learning Biometric Liveness Analysis</div>
      <div class="hero-pill">
        <span class="blink"></span>
        SYSTEM ACTIVE · BIOSENTINEL v2.0
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────
    model_path, threshold, use_face, init_btn = _sidebar()

    # ── Init ──────────────────────────────────────────────────────────────
    if init_btn:
        if not os.path.exists(model_path):
            st.error(f"Model not found: `{model_path}`")
        else:
            with st.spinner("⚡ Loading neural network weights…"):
                try:
                    _load_detector.clear()
                    det = _load_detector(model_path, threshold, use_face)
                    st.session_state.detector = det
                    st.success("✅ BioSentinel AI is operational — system ready!")
                except Exception as exc:
                    st.error(f"Initialization failed: {exc}")
                    st.session_state.detector = None

    # ── Gate ──────────────────────────────────────────────────────────────
    if st.session_state.detector is None:
        st.markdown("""
        <div class="gc" style="text-align:center;padding:3rem 2rem;">
          <div style="font-size:4rem;margin-bottom:.8rem;">🛡️</div>
          <div class="sec-h">System Offline</div>
          <div style="color:var(--td);font-size:.95rem;line-height:1.9;max-width:480px;margin:0 auto;">
            Place <code style="color:var(--nb);background:rgba(0,229,255,.1);
            padding:.1rem .4rem;border-radius:4px;">best_model.h5</code> in the same
            folder as <code style="color:var(--nb);background:rgba(0,229,255,.1);
            padding:.1rem .4rem;border-radius:4px;">app.py</code>,
            then click <b style="color:var(--nb)">⚡ Initialize System</b> in the sidebar.
          </div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Input mode ────────────────────────────────────────────────────────
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

    # ── Upload branch ─────────────────────────────────────────────────────
    if "Upload" in mode:
        uploaded = st.file_uploader(
            "upload",
            type=["jpg","jpeg","png","bmp"],
            label_visibility="collapsed",
        )
        if uploaded:
            raw = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error("Could not decode image — try another file.")
                return
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 Run Liveness Detection", type="primary",
                         use_container_width=True):
                _run_detection(img_bgr, uploaded.name, threshold, use_face)
        else:
            st.markdown("""
            <div style="text-align:center;padding:1.4rem 0;color:var(--td);
                        font-size:.95rem;letter-spacing:.1em;">
              ↑ &nbsp; Drop or browse a face image to begin analysis
            </div>""", unsafe_allow_html=True)

    # ── Camera branch ─────────────────────────────────────────────────────
    else:
        st.markdown("""
        <div style="color:var(--td);font-size:.9rem;letter-spacing:.1em;margin-bottom:.9rem;">
          Click <b style="color:var(--nb)">Take Photo</b>, then press
          <b style="color:var(--nb)">Run Liveness Detection</b>.
        </div>""", unsafe_allow_html=True)

        cam_img = st.camera_input("camera", label_visibility="collapsed")
        if cam_img:
            raw = np.frombuffer(cam_img.read(), np.uint8)
            img_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error("Could not read camera frame — try again.")
                return
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔍 Run Liveness Detection", type="primary",
                         use_container_width=True):
                _run_detection(img_bgr, "camera_snapshot", threshold, use_face)

    # ── History ───────────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-tag">🕒 Activity Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-h">Detection History</div>', unsafe_allow_html=True)
    _history_table()

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
      <div class="footer-dev">⚡ Developed by Siva</div>
      <div class="footer-copy">
        AI / ML Final Year Project &nbsp;·&nbsp; BioSentinel v2.0
        &nbsp;·&nbsp; © 2025 All Rights Reserved
      </div>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
