# app_simple.py - Enhanced with WebRTC, better UI, history, and face detection

import streamlit as st
import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import pandas as pd
from PIL import Image
import base64

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
    
    /* Logo styling */
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    
    /* Confidence bar styling */
    .confidence-bar {
        background: linear-gradient(90deg, #10b981, #ef4444);
        border-radius: 10px;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Card styling */
    .card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Result boxes */
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
    
    /* Animations */
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
    
    /* Loading animation */
    .loader {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
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
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'loading' not in st.session_state:
    st.session_state.loading = False

class VideoTransformer(VideoTransformerBase):
    """Video transformer for real-time face spoof detection"""
    
    def __init__(self):
        self.detector = None
        self.threshold = 0.5
        self.use_face_detection = True
        self.result = None
        
    def set_detector(self, detector, threshold, use_face_detection):
        self.detector = detector
        self.threshold = threshold
        self.use_face_detection = use_face_detection
    
    def recv(self, frame):
        if self.detector is None:
            return frame
        
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Make prediction
        class_name, confidence, color, bbox, face_conf = self.detector.predict_frame(img)
        
        # Store result for display
        self.result = {
            'class': class_name,
            'confidence': confidence,
            'face_conf': face_conf
        }
        
        # Draw bounding box
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            
            # Add corner markers
            corner_len = 20
            cv2.line(img, (x, y), (x+corner_len, y), color, 3)
            cv2.line(img, (x, y), (x, y+corner_len), color, 3)
            cv2.line(img, (x+w, y), (x+w-corner_len, y), color, 3)
            cv2.line(img, (x+w, y), (x+w, y+corner_len), color, 3)
            cv2.line(img, (x, y+h), (x+corner_len, y+h), color, 3)
            cv2.line(img, (x, y+h), (x, y+h-corner_len), color, 3)
            cv2.line(img, (x+w, y+h), (x+w-corner_len, y+h), color, 3)
            cv2.line(img, (x+w, y+h), (x+w, y+h-corner_len), color, 3)
        
        # Add text with background
        if class_name == "NO_FACE":
            text = "❌ No Face Detected"
            bg_color = (128, 128, 128)
        elif class_name == "REAL":
            text = f"✅ REAL ({confidence:.1f}%)"
            bg_color = (0, 255, 0)
        else:
            text = f"⚠️ SPOOF ({confidence:.1f}%)"
            bg_color = (0, 0, 255)
        
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.rectangle(img, (10, 10), (10 + text_size[0] + 20, 50), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (10 + text_size[0] + 20, 50), bg_color, 2)
        cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bg_color, 2)
        
        # Add confidence bar
        if class_name != "NO_FACE":
            bar_x, bar_y = 10, 65
            bar_w, bar_h = 300, 15
            cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (100, 100, 100), 1)
            fill_w = int(bar_w * (confidence / 100))
            cv2.rectangle(img, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), bg_color, -1)
            cv2.putText(img, f"Confidence: {confidence:.1f}%", (bar_x, bar_y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

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

def create_logo():
    """Create a simple logo using emoji and text"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="logo-container">
            <div style="font-size: 4rem;">🎭</div>
        </div>
        """, unsafe_allow_html=True)

def display_confidence_gauge(confidence):
    """Display confidence as a gauge chart"""
    from streamlit.components.v1 import html
    
    gauge_html = f"""
    <div style="width: 100%; background: #f0f0f0; border-radius: 10px; padding: 2px;">
        <div style="width: {confidence}%; background: linear-gradient(90deg, #10b981, #ef4444); 
                    border-radius: 10px; padding: 10px; text-align: center; color: white;">
            {confidence:.1f}%
        </div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)

def main():
    # Create logo
    create_logo()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎭 Face Spoof Detection System</h1>
        <p>Advanced Real-time Detection of Presentation Attacks using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Model configuration
        model_path = st.text_input("Model Path", value=str(DEFAULT_MODEL))
        threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05,
                             help="Adjust sensitivity: Lower = more sensitive to spoofs")
        use_face_detection = st.checkbox("Enable Face Detection", value=True,
                                         help="Detect and crop face before classification")
        
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
        
        if not st.session_state.detector:
            if os.path.exists(model_path):
                st.info(f"✅ Model found\nClick 'Initialize Detector' to start")
            else:
                st.error(f"❌ Model not found at: {model_path}")
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📊 Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Predictions", st.session_state.total_predictions)
        with col2:
            st.metric("Real", st.session_state.real_count)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Spoof", st.session_state.spoof_count)
        with col2:
            if st.session_state.total_predictions > 0:
                accuracy = (st.session_state.real_count / max(1, st.session_state.total_predictions)) * 100
                st.metric("Real %", f"{accuracy:.1f}%")
        
        if st.button("📈 Reset Statistics"):
            st.session_state.total_predictions = 0
            st.session_state.real_count = 0
            st.session_state.spoof_count = 0
            st.session_state.prediction_history = []
            st.success("Statistics reset!")
        
        # Prediction History
        if st.session_state.prediction_history:
            st.markdown("---")
            st.markdown("### 📜 Recent History")
            history_df = pd.DataFrame(st.session_state.prediction_history[-5:])
            for _, row in history_df.iterrows():
                icon = "✅" if row['class'] == "REAL" else "⚠️"
                st.write(f"{icon} {row['class']}: {row['confidence']:.1f}%")
                st.progress(row['confidence']/100)
        
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
        
        **Accuracy:** 95%+
        """)
    
    # Main content - Only show if detector is initialized
    if st.session_state.detector is None:
        st.warning("⚠️ Please initialize the detector using the button in the sidebar")
        
        col1, col2, col3 = st.columns(3)
        with col2:
            st.info("""
            ### 🚀 Quick Start Guide
            
            1. **Initialize Detector** - Click the button in the sidebar
            2. **Choose Detection Mode** - Upload image or use webcam
            3. **View Results** - Real-time feedback and confidence scores
            
            ### Features
            - 📸 Single Image Detection
            - 🎥 Live Webcam Detection
            - 📊 Real-time Statistics
            - 📜 Prediction History
            - 🎯 Confidence Scoring
            """)
        return
    
    # Tabs for different modes
    tab1, tab2 = st.tabs(["📸 Single Image Detection", "🎥 Live Webcam Detection"])
    
    # Tab 1: Single Image Detection
    with tab1:
        st.markdown("### 📸 Upload Image for Detection")
        
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
                # Show loading animation
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
                    
                    # Add text with background
                    if class_name == "NO_FACE":
                        text = "❌ No Face Detected"
                        bg_color = (128, 128, 128)
                    elif class_name == "REAL":
                        text = f"✅ REAL ({confidence:.1f}%)"
                        bg_color = (0, 255, 0)
                    else:
                        text = f"⚠️ SPOOF ({confidence:.1f}%)"
                        bg_color = (0, 0, 255)
                    
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
                        
                        # Show result cards with animations
                        if class_name == "REAL":
                            st.markdown("""
                            <div class="real-result">
                                <h2>✅ REAL Face Detected</h2>
                                <p>This appears to be a GENUINE face</p>
                                <hr>
                                <p>Confidence: {}%</p>
                            </div>
                            """.format(confidence), unsafe_allow_html=True)
                            st.balloons()
                            
                        elif class_name == "SPOOF":
                            st.markdown("""
                            <div class="spoof-result">
                                <h2>⚠️ SPOOF Attack Detected!</h2>
                                <p>This is a FAKE/SPOOF attempt</p>
                                <hr>
                                <p>Confidence: {}%</p>
                            </div>
                            """.format(confidence), unsafe_allow_html=True)
                            
                        else:
                            st.warning("""
                            ### ❌ No Face Detected
                            
                            Could not detect a clear face in the image.
                            
                            **Tips:**
                            - Ensure good lighting
                            - Face should be clearly visible
                            - Try a different image
                            """)
                        
                        # Display confidence gauge
                        if class_name != "NO_FACE":
                            st.markdown("### Confidence Score")
                            display_confidence_gauge(confidence)
                        
                        if face_conf > 0 and bbox:
                            st.info(f"📐 Face Detection Quality: {face_conf:.2f}")
    
    # Tab 2: Live Webcam Detection
    with tab2:
        st.markdown("### 🎥 Live Webcam Detection")
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
        if ctx.video_transformer and ctx.video_transformer.result:
            result = ctx.video_transformer.result
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if result['class'] == "REAL":
                    st.success(f"✅ **{result['class']}**")
                elif result['class'] == "SPOOF":
                    st.error(f"⚠️ **{result['class']}**")
                else:
                    st.warning("❓ No Face")
            
            with col2:
                if result['class'] != "NO_FACE":
                    st.metric("Confidence", f"{result['confidence']:.1f}%")
            
            with col3:
                if result.get('face_conf', 0) > 0:
                    st.metric("Face Quality", f"{result['face_conf']:.2f}")
            
            # Update statistics for live predictions
            if result['class'] != "NO_FACE":
                # To avoid counting every frame, we count once per session
                if 'last_counted' not in st.session_state:
                    st.session_state.last_counted = None
                
                current_time = time.time()
                if (st.session_state.last_counted is None or 
                    current_time - st.session_state.last_counted > 2):
                    st.session_state.total_predictions += 1
                    if result['class'] == "REAL":
                        st.session_state.real_count += 1
                    elif result['class'] == "SPOOF":
                        st.session_state.spoof_count += 1
                    st.session_state.last_counted = current_time

if __name__ == "__main__":
    main()
