 # app_simple.py - Error-free version with all UI improvements

import streamlit as st
import numpy as np
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

# Set environment variable before importing cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Now import cv2
try:
    import cv2
except ImportError as e:
    st.error(f"Error importing OpenCV: {e}")
    st.stop()

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
    page_title="Face Spoof Detection System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with better styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
    }
    .logo-container {
        text-align: center;
        margin-bottom: 1rem;
    }
    .real-result {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        animation: pulse 1s ease-in-out;
    }
    .spoof-result {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        animation: shake 0.5s ease-in-out;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        transition: transform 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stAlert {
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: bold;
    }
</style>
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

def main():
    # Logo and Header
    st.markdown("""
    <div class="logo-container">
        <div style="font-size: 4rem;">🎭</div>
    </div>
    <div class="main-header">
        <h1>Face Spoof Detection System</h1>
        <p>Advanced Detection of Presentation Attacks using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Model configuration
        st.markdown("### Model Configuration")
        model_path = st.text_input("Model Path", value=str(DEFAULT_MODEL))
        threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05,
                             help="Adjust sensitivity: Lower = more sensitive to spoofs")
        use_face_detection = st.checkbox("Enable Face Detection", value=True,
                                         help="Detect and crop face before classification")
        
        # Initialize detector button
        if st.button("🚀 Initialize Detector"):
            with st.spinner("🔄 Initializing detector (this may take a moment)..."):
                st.session_state.detector = init_detector(
                    model_path, threshold, use_face_detection
                )
                if st.session_state.detector:
                    st.success("✅ Detector initialized successfully!")
                    st.balloons()
                else:
                    st.error("❌ Failed to initialize detector")
        
        if not st.session_state.detector:
            if os.path.exists(model_path):
                st.info(f"✅ Model found\n\nClick 'Initialize Detector' to start")
            else:
                st.error(f"❌ Model not found")
                st.info("Please ensure 'best_model.h5' is in the same directory")
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📊 Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", st.session_state.total_predictions)
        with col2:
            st.metric("✅ Real", st.session_state.real_count)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("⚠️ Spoof", st.session_state.spoof_count)
        with col2:
            if st.session_state.total_predictions > 0:
                valid = st.session_state.real_count + st.session_state.spoof_count
                accuracy = (st.session_state.real_count / max(1, valid)) * 100
                st.metric("Real %", f"{accuracy:.1f}%")
        
        if st.button("📈 Reset Statistics"):
            st.session_state.total_predictions = 0
            st.session_state.real_count = 0
            st.session_state.spoof_count = 0
            st.session_state.no_face_count = 0
            st.session_state.prediction_history = []
            st.success("Statistics reset!")
        
        # Prediction History
        if st.session_state.prediction_history:
            st.markdown("---")
            st.markdown("### 📜 Recent History")
            for pred in st.session_state.prediction_history[-5:]:
                icon = "✅" if pred['class'] == "REAL" else "⚠️"
                st.write(f"{icon} **{pred['class']}**: {pred['confidence']:.1f}%")
                st.progress(pred['confidence']/100)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **Model:** MobileNetV3  
        **Face Detector:** MTCNN  
        
        **Detects:**
        - ✨ REAL faces
        - ⚠️ Print attacks
        - 📱 Replay attacks  
        - 🎭 Mask attacks
        """)
    
    # Main content
    if st.session_state.detector is None:
        st.info("""
        ### 🚀 Welcome to Face Spoof Detection System!
        
        #### Quick Start Guide:
        1. **Initialize Detector** - Click the button in the sidebar
        2. **Upload Image** - Choose a clear face image
        3. **Detect** - Click Detect to analyze
        
        #### Features:
        - 📸 Single Image Detection
        - 🎯 Real-time Confidence Scoring
        - 📊 Detailed Statistics
        - 📜 Prediction History
        """)
        return
    
    # Single Image Detection
    st.markdown("## 📸 Upload Image for Detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image of a face for spoof detection"
    )
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                    caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Detect Spoof", type="primary"):
            with st.spinner("🔍 Analyzing image..."):
                # Update detector settings
                st.session_state.detector.threshold = threshold
                st.session_state.detector.use_face_detection = use_face_detection
                
                # Make prediction
                class_name, confidence, color, bbox, face_conf = \
                    st.session_state.detector.predict_frame(image)
                
                # Update statistics
                st.session_state.total_predictions += 1
                if class_name == "REAL":
                    st.session_state.real_count += 1
                elif class_name == "SPOOF":
                    st.session_state.spoof_count += 1
                else:
                    st.session_state.no_face_count += 1
                
                # Add to history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'class': class_name,
                    'confidence': confidence
                })
                
                # Draw result on image
                result_img = image.copy()
                
                # Draw bounding box
                if bbox is not None:
                    x, y, w, h = bbox
                    cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 3)
                    
                    # Add corner markers
                    corner_len = 20
                    cv2.line(result_img, (x, y), (x+corner_len, y), color, 3)
                    cv2.line(result_img, (x, y), (x, y+corner_len), color, 3)
                    cv2.line(result_img, (x+w, y), (x+w-corner_len, y), color, 3)
                    cv2.line(result_img, (x+w, y), (x+w, y+corner_len), color, 3)
                    cv2.line(result_img, (x, y+h), (x+corner_len, y+h), color, 3)
                    cv2.line(result_img, (x, y+h), (x, y+h-corner_len), color, 3)
                    cv2.line(result_img, (x+w, y+h), (x+w-corner_len, y+h), color, 3)
                    cv2.line(result_img, (x+w, y+h), (x+w, y+h-corner_len), color, 3)
                
                # Add text
                if class_name == "NO_FACE":
                    text = "❌ No Face Detected"
                    text_color = (128, 128, 128)
                elif class_name == "REAL":
                    text = f"✅ REAL ({confidence:.1f}%)"
                    text_color = (0, 255, 0)
                else:
                    text = f"⚠️ SPOOF ({confidence:.1f}%)"
                    text_color = (0, 0, 255)
                
                # Add text on image
                cv2.putText(result_img, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, text_color, 2)
                
                # Add confidence bar
                if class_name != "NO_FACE":
                    bar_x, bar_y = 10, 70
                    bar_w, bar_h = 300, 20
                    cv2.rectangle(result_img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), 
                                 (100, 100, 100), 1)
                    fill_w = int(bar_w * (confidence / 100))
                    cv2.rectangle(result_img, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), 
                                 text_color, -1)
                    cv2.putText(result_img, f"Confidence: {confidence:.1f}%", 
                               (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 255), 1)
                
                with col2:
                    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                            caption="Detection Result", use_column_width=True)
                    
                    # Show result cards
                    if class_name == "REAL":
                        st.markdown(f"""
                        <div class="real-result">
                            <h2>✅ REAL Face Detected</h2>
                            <p>This appears to be a GENUINE face</p>
                            <hr>
                            <h3>Confidence: {confidence:.1f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                        
                    elif class_name == "SPOOF":
                        st.markdown(f"""
                        <div class="spoof-result">
                            <h2>⚠️ SPOOF Attack Detected!</h2>
                            <p>This is a FAKE/SPOOF attempt</p>
                            <hr>
                            <h3>Confidence: {confidence:.1f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        st.warning("""
                        ### ❌ No Face Detected
                        
                        Could not detect a clear face in the image.
                        
                        **Tips:**
                        - Ensure good lighting
                        - Face should be clearly visible
                        """)
                    
                    # Display confidence gauge
                    if class_name != "NO_FACE":
                        st.markdown("### Confidence Score")
                        st.progress(confidence/100, text=f"{confidence:.1f}%")
                    
                    if face_conf > 0 and bbox:
                        st.info(f"📐 Face Detection Quality: {face_conf:.2f}")

if __name__ == "__main__":
    main()
