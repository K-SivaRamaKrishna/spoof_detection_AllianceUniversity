# app_simple.py - Streamlit app with WebRTC for online webcam

import streamlit as st
import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

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
    .real-box {
        background-color: #10b981;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .spoof-box {
        background-color: #ef4444;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
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
if 'latest_prediction' not in st.session_state:
    st.session_state.latest_prediction = None

class VideoTransformer(VideoTransformerBase):
    """Video transformer for real-time face spoof detection"""
    
    def __init__(self):
        self.detector = None
        self.threshold = 0.5
        self.use_face_detection = True
        
    def set_detector(self, detector, threshold, use_face_detection):
        self.detector = detector
        self.threshold = threshold
        self.use_face_detection = use_face_detection
    
    def transform(self, frame):
        if self.detector is None:
            return frame.to_ndarray(format="bgr24")
        
        # Get frame as numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Make prediction
        class_name, confidence, color, bbox, face_conf = self.detector.predict_frame(img)
        
        # Draw bounding box
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            
            # Add corner markers
            corner_len = 15
            cv2.line(img, (x, y), (x+corner_len, y), color, 2)
            cv2.line(img, (x, y), (x, y+corner_len), color, 2)
            cv2.line(img, (x+w, y), (x+w-corner_len, y), color, 2)
            cv2.line(img, (x+w, y), (x+w, y+corner_len), color, 2)
            cv2.line(img, (x, y+h), (x+corner_len, y+h), color, 2)
            cv2.line(img, (x, y+h), (x, y+h-corner_len), color, 2)
            cv2.line(img, (x+w, y+h), (x+w-corner_len, y+h), color, 2)
            cv2.line(img, (x+w, y+h), (x+w, y+h-corner_len), color, 2)
        
        # Add text
        if class_name == "NO_FACE":
            text = "No Face Detected"
        else:
            text = f"{class_name} ({confidence:.1f}%)"
        
        # Background for text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img, (10, 10), (10 + text_size[0] + 20, 45), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (10 + text_size[0] + 20, 45), color, 2)
        cv2.putText(img, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Update session state for display
        st.session_state.latest_prediction = {
            'class': class_name,
            'confidence': confidence,
            'face_conf': face_conf
        }
        
        return img

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
        <p style="color: white; margin: 0;">Real-time detection of presentation attacks using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        model_path = st.text_input("Model Path", value=str(DEFAULT_MODEL))
        threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05)
        use_face_detection = st.checkbox("Enable Face Detection", value=True)
        
        # Initialize detector button
        if st.button("🚀 Initialize Detector", use_container_width=True):
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
        
        if st.button("📈 Reset Statistics", use_container_width=True):
            st.session_state.total_predictions = 0
            st.session_state.real_count = 0
            st.session_state.spoof_count = 0
            st.success("Statistics reset!")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("**Model:** MobileNetV3\n**Face Detector:** MTCNN\n**Webcam:** WebRTC")
    
    # Main content
    if st.session_state.detector is None:
        st.warning("⚠️ Please initialize the detector using the button in the sidebar")
        st.markdown("""
        ### How to use:
        1. Make sure `best_model.h5` is in the same directory as this app
        2. Click **Initialize Detector** in the sidebar
        3. Wait for the model to load
        4. Upload an image or start webcam detection
        """)
        return
    
    # Tabs
    tab1, tab2 = st.tabs(["📸 Single Image Detection", "🎥 Live Webcam Detection"])
    
    # Tab 1: Single Image Detection
    with tab1:
        st.markdown("### Single Image Detection")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                        caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Detect", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    class_name, confidence, color, bbox, face_conf = \
                        st.session_state.detector.predict_frame(image)
                    
                    st.session_state.total_predictions += 1
                    if class_name == "REAL":
                        st.session_state.real_count += 1
                    elif class_name == "SPOOF":
                        st.session_state.spoof_count += 1
                    
                    # Draw result
                    result_img = image.copy()
                    if bbox is not None:
                        x, y, w, h = bbox
                        cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
                    
                    if class_name == "NO_FACE":
                        text = "No Face Detected"
                    else:
                        text = f"{class_name} ({confidence:.1f}%)"
                    
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    cv2.rectangle(result_img, (10, 10), (10 + text_size[0] + 20, 50), 
                                 (0, 0, 0), -1)
                    cv2.rectangle(result_img, (10, 10), (10 + text_size[0] + 20, 50), color, 2)
                    cv2.putText(result_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9, color, 2)
                    
                    with col2:
                        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                                caption="Detection Result", use_container_width=True)
                        
                        if class_name == "REAL":
                            st.success(f"✅ **{class_name}**")
                            st.markdown(f"**Confidence:** {confidence:.1f}%")
                            st.progress(confidence/100)
                        elif class_name == "SPOOF":
                            st.error(f"⚠️ **{class_name}**")
                            st.markdown(f"**Confidence:** {confidence:.1f}%")
                            st.progress(confidence/100)
                        else:
                            st.warning(f"❓ **{class_name}**")
    
    # Tab 2: Live Webcam Detection with WebRTC
    with tab2:
        st.markdown("### Live Webcam Detection")
        st.info("📹 Click 'Start' to access your webcam. Works on mobile and desktop!")
        
        # Create video transformer
        ctx = webrtc_streamer(
            key="spoof-detection",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=VideoTransformer,
            async_processing=True,
            media_stream_constraints={"video": True, "audio": False},
        )
        
        # Update transformer with detector
        if ctx.video_transformer:
            ctx.video_transformer.set_detector(
                st.session_state.detector, 
                threshold, 
                use_face_detection
            )
        
        # Display real-time prediction info
        if st.session_state.latest_prediction:
            pred = st.session_state.latest_prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if pred['class'] == "REAL":
                    st.success(f"✅ **{pred['class']}**")
                elif pred['class'] == "SPOOF":
                    st.error(f"⚠️ **{pred['class']}**")
                else:
                    st.warning("❓ No Face")
            
            with col2:
                if pred['class'] != "NO_FACE":
                    st.metric("Confidence", f"{pred['confidence']:.1f}%")
                    st.progress(pred['confidence']/100)
            
            with col3:
                if pred.get('face_conf', 0) > 0:
                    st.metric("Face Confidence", f"{pred['face_conf']:.2f}")

if __name__ == "__main__":
    main()
