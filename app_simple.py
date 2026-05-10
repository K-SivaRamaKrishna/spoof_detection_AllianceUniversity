# app_simple.py - Ultra-Modern Futuristic AI Face Spoof Detection System

import streamlit as st
import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

# Set environment variable before importing
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Import cv2
import cv2

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your working script's class
from predict_live import LiveSpoofDetector

# ===== GITHUB PATHS - Model in same directory =====
SCRIPT_DIR = Path(__file__).parent.absolute()
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "best_model.h5")
# ====================================================

# Page configuration
st.set_page_config(
    page_title="AI Face Spoof Detection System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ULTRA-MODERN FUTURISTIC CSS
# ============================================
st.markdown("""
<style>
    /* Import futuristic fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    .stApp {
        background: radial-gradient(ellipse at 20% 30%, #0a0a2a, #000000);
        font-family: 'Inter', sans-serif;
    }
    
    /* Floating particles background */
    .particles {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
    }
    
    .particle {
        position: absolute;
        background: radial-gradient(circle, rgba(0, 255, 255, 0.8), transparent);
        border-radius: 50%;
        animation: float 15s infinite ease-in-out;
        opacity: 0.3;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) translateX(0px); opacity: 0.3; }
        25% { transform: translateY(-20px) translateX(10px); opacity: 0.6; }
        50% { transform: translateY(0px) translateX(20px); opacity: 0.4; }
        75% { transform: translateY(20px) translateX(10px); opacity: 0.7; }
    }
    
    /* Main header - Cinematic Hero Section */
    .hero-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(0, 255, 255, 0.3);
        box-shadow: 0 0 40px rgba(0, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #00ffff, #ff00ff, #00ffff);
        border-radius: 20px;
        z-index: -1;
        opacity: 0.5;
        animation: borderGlow 3s ease-in-out infinite;
    }
    
    @keyframes borderGlow {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.8; }
    }
    
    .glow-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00ffff, #ff00ff, #00ffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: titlePulse 2s ease-in-out infinite;
        letter-spacing: 2px;
    }
    
    @keyframes titlePulse {
        0%, 100% { text-shadow: 0 0 20px rgba(0, 255, 255, 0.5); }
        50% { text-shadow: 0 0 40px rgba(0, 255, 255, 0.8); }
    }
    
    .glow-subtitle {
        color: rgba(0, 255, 255, 0.8);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        letter-spacing: 1px;
    }
    
    .system-status {
        display: inline-block;
        background: rgba(0, 255, 0, 0.2);
        border: 1px solid #00ff00;
        border-radius: 20px;
        padding: 0.3rem 1rem;
        margin-top: 1rem;
        font-size: 0.8rem;
        color: #00ff00;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; box-shadow: 0 0 5px #00ff00; }
        50% { opacity: 0.7; box-shadow: 0 0 20px #00ff00; }
    }
    
    /* Scanning animation */
    .scan-line {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00ffff, #ff00ff, #00ffff, transparent);
        animation: scan 2s linear infinite;
    }
    
    @keyframes scan {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(1000%); }
    }
    
    /* Sidebar futuristic styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 10, 42, 0.95), rgba(0, 0, 0, 0.95));
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 255, 255, 0.3);
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(0, 255, 255, 0.2);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 255, 255, 0.6);
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.2);
    }
    
    /* Result cards with neon effects */
    .real-card {
        background: linear-gradient(135deg, rgba(0, 255, 0, 0.15), rgba(0, 100, 0, 0.3));
        border: 2px solid #00ff00;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        animation: neonPulse 1s ease-in-out;
        box-shadow: 0 0 30px rgba(0, 255, 0, 0.3);
    }
    
    .spoof-card {
        background: linear-gradient(135deg, rgba(255, 0, 0, 0.15), rgba(100, 0, 0, 0.3));
        border: 2px solid #ff0000;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        animation: shake 0.5s ease-in-out;
        box-shadow: 0 0 30px rgba(255, 0, 0, 0.3);
    }
    
    @keyframes neonPulse {
        0% { transform: scale(0.95); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }
    
    /* Upload zone futuristic */
    .upload-zone {
        border: 2px dashed rgba(0, 255, 255, 0.5);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        background: rgba(0, 255, 255, 0.05);
    }
    
    .upload-zone:hover {
        border-color: #00ffff;
        background: rgba(0, 255, 255, 0.1);
        box-shadow: 0 0 40px rgba(0, 255, 255, 0.2);
    }
    
    /* Confidence meter circular */
    .confidence-meter {
        width: 200px;
        height: 200px;
        margin: 0 auto;
        position: relative;
    }
    
    /* Animated buttons */
    .futuristic-btn {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border: none;
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        width: 100%;
    }
    
    .futuristic-btn::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.5);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .futuristic-btn:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .futuristic-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics cards */
    .metric-card {
        background: rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(0, 255, 255, 0.3);
    }
    
    /* Loading animation */
    .loader {
        width: 60px;
        height: 60px;
        border: 3px solid rgba(0, 255, 255, 0.3);
        border-top: 3px solid #00ffff;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .radar {
        position: relative;
        width: 100px;
        height: 100px;
        margin: 0 auto;
    }
    
    .radar::before {
        content: '';
        position: absolute;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        background: conic-gradient(from 0deg, transparent, #00ffff);
        animation: radar 2s linear infinite;
    }
    
    @keyframes radar {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* History table */
    .history-table {
        background: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Footer */
    .futuristic-footer {
        margin-top: 3rem;
        padding: 2rem;
        text-align: center;
        border-top: 1px solid rgba(0, 255, 255, 0.3);
        background: rgba(0, 0, 0, 0.5);
        border-radius: 20px;
    }
    
    /* Webcam placeholder */
    .webcam-placeholder {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.6));
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 2px solid rgba(0, 255, 255, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    /* Custom slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
    }
    
    /* Animations for prediction transitions */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
</style>

<!-- Floating Particles Background -->
<script>
    function createParticles() {
        const particlesContainer = document.createElement('div');
        particlesContainer.className = 'particles';
        document.body.appendChild(particlesContainer);
        
        for (let i = 0; i < 50; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.width = Math.random() * 5 + 'px';
            particle.style.height = particle.style.width;
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 15 + 's';
            particle.style.animationDuration = 10 + Math.random() * 10 + 's';
            particlesContainer.appendChild(particle);
        }
    }
    createParticles();
</script>
""", unsafe_allow_html=True)

# Initialize session state
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'real_count' not in st.session_state:
    st.session_state.real_count = 0
if 'spoof_count' not in st.session_state:
    st.session_state.spoof_count = 0
if 'no_face_count' not in st.session_state:
    st.session_state.no_face_count = 0
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

def init_detector(model_path, threshold, use_face_detection):
    """Initialize the detector using your working class"""
    try:
        if not os.path.exists(model_path):
            st.error(f"Model not found at: {model_path}")
            return None
        detector = LiveSpoofDetector(
            model_path=model_path,
            threshold=threshold,
            use_face_detection=use_face_detection,
            smooth_frames=5,
            camera_id=0
        )
        return detector
    except Exception as e:
        st.error(f"Failed to initialize detector: {str(e)}")
        return None

def create_circular_meter(confidence, class_name):
    """Create a circular confidence meter using HTML/CSS"""
    color = "#00ff00" if class_name == "REAL" else "#ff0000"
    normalized = confidence / 100
    
    meter_html = f"""
    <div class="confidence-meter">
        <svg width="200" height="200" viewBox="0 0 200 200">
            <circle cx="100" cy="100" r="85" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="15"/>
            <circle cx="100" cy="100" r="85" fill="none" stroke="{color}" stroke-width="15"
                    stroke-dasharray="{2 * 3.14159 * 85}" stroke-dashoffset="{2 * 3.14159 * 85 * (1 - normalized)}"
                    stroke-linecap="round" transform="rotate(-90 100 100)">
                <animate attributeName="stroke-dashoffset" from="{2 * 3.14159 * 85}" to="{2 * 3.14159 * 85 * (1 - normalized)}" dur="1s" fill="freeze"/>
            </circle>
            <text x="100" y="100" text-anchor="middle" dominant-baseline="middle" fill="{color}" font-size="30" font-weight="bold">
                {confidence:.0f}%
                <animate attributeName="opacity" values="0;1" dur="0.5s" fill="freeze"/>
            </text>
        </svg>
    </div>
    """
    return meter_html

def main():
    # Hero Header Section
    st.markdown("""
    <div class="hero-header">
        <div class="scan-line"></div>
        <div class="glow-title">🛡️ AI FACE SPOOF DETECTION SYSTEM</div>
        <div class="glow-subtitle">Next-Generation Deep Learning Liveness Detection</div>
        <div class="system-status">
            <span>🟢 SYSTEM ACTIVE</span>
        </div>
        <div class="radar" style="margin: 20px auto 0;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Futuristic Navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <div style="font-size: 3rem;">🎭</div>
            <div style="font-size: 1.2rem; font-weight: bold; background: linear-gradient(135deg, #00ffff, #ff00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                AI SECURITY
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Configuration
        st.markdown("### ⚡ SYSTEM CONFIGURATION")
        model_path = st.text_input("Model Path", value=str(DEFAULT_MODEL))
        threshold = st.slider("CLASSIFICATION THRESHOLD", 0.0, 1.0, 0.5, 0.05,
                             help="Adjust detection sensitivity")
        use_face_detection = st.checkbox("ENABLE FACE DETECTION", value=True)
        
        # Initialize Detector Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 INITIALIZE AI", use_container_width=True):
                with st.spinner("🔄 Bootstrapping AI Model..."):
                    st.session_state.detector = init_detector(
                        model_path, threshold, use_face_detection
                    )
                    if st.session_state.detector:
                        st.success("✅ AI System Online")
                        st.balloons()
                    else:
                        st.error("❌ AI Boot Failed")
        
        if not st.session_state.detector:
            if os.path.exists(model_path):
                st.info("✅ Model Detected\nClick INITIALIZE AI to start")
        
        st.markdown("---")
        
        # Live Statistics
        st.markdown("### 📊 LIVE STATS")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("TOTAL", st.session_state.total_predictions)
        with col2:
            st.metric("✅ REAL", st.session_state.real_count)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("⚠️ SPOOF", st.session_state.spoof_count)
        with col2:
            valid = st.session_state.real_count + st.session_state.spoof_count
            accuracy = (st.session_state.real_count / max(1, valid)) * 100
            st.metric("ACCURACY", f"{accuracy:.1f}%")
        
        if st.button("🔄 RESET STATISTICS", use_container_width=True):
            st.session_state.total_predictions = 0
            st.session_state.real_count = 0
            st.session_state.spoof_count = 0
            st.session_state.no_face_count = 0
            st.session_state.prediction_history = []
            st.success("Statistics Reset")
        
        st.markdown("---")
        
        # AI Status Monitor
        st.markdown("### 🔍 AI STATUS")
        st.markdown("""
        <div class="glass-card" style="font-size: 0.8rem;">
            <div>🟢 Model: <span style="color: #00ff00;">MobileNetV3</span></div>
            <div>🔵 Detector: <span style="color: #00ffff;">MTCNN</span></div>
            <div>⚡ Mode: <span style="color: #ffff00;">Production</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Prediction History
        if st.session_state.prediction_history:
            st.markdown("### 📜 ACTIVITY LOG")
            for pred in st.session_state.prediction_history[-5:]:
                icon = "✅" if pred['class'] == "REAL" else "⚠️"
                st.markdown(f"""
                <div class="glass-card" style="margin: 0.3rem 0; padding: 0.5rem;">
                    <small>{pred['timestamp'].strftime('%H:%M:%S')}</small><br>
                    {icon} <strong>{pred['class']}</strong> - {pred['confidence']:.1f}%
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; font-size: 0.7rem; opacity: 0.6;">
            🔒 Enterprise-Grade Security<br>
            🎭 AI/ML Final Year Project
        </div>
        """, unsafe_allow_html=True)
    
    # Main Dashboard
    if st.session_state.detector is None:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3rem;">
            <div style="font-size: 3rem;">🤖</div>
            <h2>AI System Ready for Initialization</h2>
            <p>Click <strong>INITIALIZE AI</strong> in the sidebar to activate the detection system</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Detection Mode Selection
    mode = st.radio("SELECT MODE", ["📸 IMAGE DETECTION", "🎥 LIVE WEBCAM"], horizontal=True)
    
    if mode == "📸 IMAGE DETECTION":
        st.markdown("### 📤 UPLOAD TARGET IMAGE")
        
        uploaded_file = st.file_uploader(
            "", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📸 INPUT")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            if st.button("🔍 EXECUTE DETECTION", use_container_width=True):
                with st.spinner("🔬 Analyzing with AI..."):
                    st.session_state.detector.threshold = threshold
                    st.session_state.detector.use_face_detection = use_face_detection
                    
                    class_name, confidence, color, bbox, face_conf = \
                        st.session_state.detector.predict_frame(image)
                    
                    st.session_state.total_predictions += 1
                    if class_name == "REAL":
                        st.session_state.real_count += 1
                    elif class_name == "SPOOF":
                        st.session_state.spoof_count += 1
                    else:
                        st.session_state.no_face_count += 1
                    
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now(),
                        'class': class_name,
                        'confidence': confidence
                    })
                    
                    # Draw result
                    result_img = image.copy()
                    if bbox is not None:
                        x, y, w, h = bbox
                        cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 3)
                        corner_len = 20
                        cv2.line(result_img, (x, y), (x+corner_len, y), color, 3)
                        cv2.line(result_img, (x, y), (x, y+corner_len), color, 3)
                        cv2.line(result_img, (x+w, y), (x+w-corner_len, y), color, 3)
                        cv2.line(result_img, (x+w, y), (x+w, y+corner_len), color, 3)
                        cv2.line(result_img, (x, y+h), (x+corner_len, y+h), color, 3)
                        cv2.line(result_img, (x, y+h), (x, y+h-corner_len), color, 3)
                        cv2.line(result_img, (x+w, y+h), (x+w-corner_len, y+h), color, 3)
                        cv2.line(result_img, (x+w, y+h), (x+w, y+h-corner_len), color, 3)
                    
                    with col2:
                        st.markdown("#### 🎯 AI VERDICT")
                        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_column_width=True)
                        
                        if class_name == "REAL":
                            st.markdown(f"""
                            <div class="real-card fade-in-up">
                                <div style="font-size: 4rem;">✅</div>
                                <h1 style="font-size: 2rem;">GENUINE FACE</h1>
                                <h3>Confidence: {confidence:.1f}%</h3>
                                <div class="confidence-meter" style="width: 100%; height: auto;">
                                    <div style="background: linear-gradient(90deg, #00ff00, #00cc00); 
                                                width: {confidence}%; height: 30px; border-radius: 15px;
                                                animation: slideIn 1s ease-out;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                            
                        elif class_name == "SPOOF":
                            st.markdown(f"""
                            <div class="spoof-card fade-in-up">
                                <div style="font-size: 4rem;">⚠️</div>
                                <h1 style="font-size: 2rem;">SPOOF DETECTED</h1>
                                <h3>Confidence: {confidence:.1f}%</h3>
                                <div class="confidence-meter" style="width: 100%; height: auto;">
                                    <div style="background: linear-gradient(90deg, #ff0000, #cc0000); 
                                                width: {confidence}%; height: 30px; border-radius: 15px;
                                                animation: slideIn 1s ease-out;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("No Face Detected")
                        
                        st.markdown(create_circular_meter(confidence, class_name), unsafe_allow_html=True)
    
    else:  # Live Webcam Mode
        st.markdown("### 🎥 LIVE SURVEILLANCE FEED")
        st.info("📹 Webcam integration available in premium version. Currently showing simulation.")
        
        st.markdown("""
        <div class="webcam-placeholder">
            <div style="font-size: 4rem;">📷</div>
            <h3>Camera Feed Preview</h3>
            <p>Connect webcam for real-time AI detection</p>
            <div class="radar" style="width: 100px; height: 100px; margin: 20px auto;"></div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="futuristic-footer">
        <div style="font-size: 1.5rem;">🎭</div>
        <div style="font-weight: bold; background: linear-gradient(135deg, #00ffff, #ff00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Developed by Siva
        </div>
        <div style="font-size: 0.8rem; opacity: 0.7;">AI/ML Final Year Project | Advanced Biometric Security System</div>
        <div style="margin-top: 1rem; font-size: 0.7rem; opacity: 0.5;">© 2024 All Rights Reserved</div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
