# app_simple.py - Fixed for Streamlit 1.28.0

import streamlit as st
import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime
from pathlib import Path

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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
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
if 'detector' not in st.session_state:
    st.session_state.detector = None

def init_detector(model_path, threshold, use_face_detection):
    """Initialize the detector using your working class"""
    try:
        detector = LiveSpoofDetector(
            model_path=model_path,
            threshold=threshold,
            use_face_detection=use_face_detection,
            smooth_frames=5,
            camera_id=0
        )
        return detector
    except Exception as e:
        st.error(f"Failed to initialize detector: {e}")
        return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🎭 Face Spoof Detection System</h1>
        <p style="color: white; margin: 0;">Upload an image to detect REAL vs SPOOF faces</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        model_path = st.text_input("Model Path", value=str(DEFAULT_MODEL))
        threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)
        use_face_detection = st.checkbox("Enable Face Detection", value=True)
        
        # Initialize detector button
        if st.button("🚀 Initialize Detector"):
            with st.spinner("Initializing detector (this may take a moment)..."):
                st.session_state.detector = init_detector(
                    model_path, threshold, use_face_detection
                )
                if st.session_state.detector:
                    st.success("✅ Detector initialized successfully!")
                else:
                    st.error("❌ Failed to initialize detector")
        
        # Check if model exists
        if not st.session_state.detector:
            if os.path.exists(model_path):
                st.info(f"✅ Model found\nClick 'Initialize Detector' to start")
            else:
                st.error(f"❌ Model not found at: {model_path}")
                st.info("Please ensure 'best_model.h5' is in the same directory as this app")
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📊 Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", st.session_state.total_predictions)
        with col2:
            st.metric("Real", st.session_state.real_count)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Spoof", st.session_state.spoof_count)
        with col2:
            if st.session_state.total_predictions > 0:
                accuracy = (st.session_state.real_count / st.session_state.total_predictions) * 100
                st.metric("Real %", f"{accuracy:.1f}%")
        
        if st.button("📈 Reset Statistics"):
            st.session_state.total_predictions = 0
            st.session_state.real_count = 0
            st.session_state.spoof_count = 0
            st.success("Statistics reset!")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("**Model:** MobileNetV3\n**Face Detector:** MTCNN")
    
    # Main content
    if st.session_state.detector is None:
        st.warning("⚠️ Please initialize the detector using the button in the sidebar")
        st.markdown("""
        ### How to use:
        1. Make sure `best_model.h5` is in the same directory as this app
        2. Click **Initialize Detector** in the sidebar
        3. Wait for the model to load
        4. Upload an image below
        5. Click **Detect** to see the result
        """)
        return
    
    # Single Image Detection
    st.markdown("## 📸 Upload Image for Detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp']
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
            with st.spinner("Analyzing image..."):
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
                    bg_color = (128, 128, 128)
                elif class_name == "REAL":
                    text = f"✅ REAL ({confidence:.1f}%)"
                    bg_color = (0, 255, 0)
                else:
                    text = f"⚠️ SPOOF ({confidence:.1f}%)"
                    bg_color = (0, 0, 255)
                
                # Background for text
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                cv2.rectangle(result_img, (10, 10), (10 + text_size[0] + 20, 55), 
                             (0, 0, 0), -1)
                cv2.rectangle(result_img, (10, 10), (10 + text_size[0] + 20, 55), 
                             bg_color, 2)
                cv2.putText(result_img, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, bg_color, 2)
                
                # Add confidence bar for valid detections
                if class_name != "NO_FACE":
                    bar_x, bar_y = 10, 70
                    bar_w, bar_h = 300, 20
                    cv2.rectangle(result_img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), 
                                 (100, 100, 100), 1)
                    fill_w = int(bar_w * (confidence / 100))
                    cv2.rectangle(result_img, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), 
                                 bg_color, -1)
                    
                    # Add confidence text
                    cv2.putText(result_img, f"Confidence: {confidence:.1f}%", 
                               (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 255), 1)
                
                with col2:
                    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                            caption="Detection Result", use_column_width=True)
                    
                    # Show result cards
                    if class_name == "REAL":
                        st.success(f"""
                        ### ✅ REAL Face Detected
                        
                        **Confidence:** {confidence:.1f}%
                        **Decision:** This is a GENUINE face
                        """)
                        st.balloons()
                        
                    elif class_name == "SPOOF":
                        st.error(f"""
                        ### ⚠️ SPOOF Attack Detected!
                        
                        **Confidence:** {confidence:.1f}%
                        **Decision:** This is a FAKE/SPOOF attempt
                        """)
                        
                    else:
                        st.warning(f"""
                        ### ❌ No Face Detected
                        
                        Could not detect a clear face in the image.
                        """)
                    
                    if face_conf > 0 and bbox:
                        st.info(f"📐 Face Detection Confidence: {face_conf:.2f}")

if __name__ == "__main__":
    main()
