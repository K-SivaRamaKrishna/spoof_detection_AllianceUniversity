# app_simple_fixed.py - Fixed version with correct Streamlit parameters

import streamlit as st
import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your working script's class
from predict_live import LiveSpoofDetector

# Page configuration
st.set_page_config(
    page_title="Spoof Detection System",
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
        <p style="color: white; margin: 0;">Real-time detection of presentation attacks using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Model path - same as your working script
        
        DEFAULT_MODEL = "best_model.h5"
        
        model_path = st.text_input("Model Path", value=DEFAULT_MODEL)
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
                st.error(f"❌ Model not found")
        
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
    
    # Main content - Only show if detector is initialized
    if st.session_state.detector is None:
        st.warning("⚠️ Please initialize the detector using the button in the sidebar")
        return
    
    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["📸 Single Image Detection", "🎥 Live Webcam Detection"])
    
    # Tab 1: Single Image Detection
    with tab1:
        st.markdown("### Single Image Detection")
        
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
            
            if st.button("🔍 Detect"):
                with st.spinner("Analyzing image..."):
                    # Use the detector's predict_frame method
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
                        cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
                        
                        # Add corner markers
                        corner_len = 15
                        cv2.line(result_img, (x, y), (x+corner_len, y), color, 2)
                        cv2.line(result_img, (x, y), (x, y+corner_len), color, 2)
                        cv2.line(result_img, (x+w, y), (x+w-corner_len, y), color, 2)
                        cv2.line(result_img, (x+w, y), (x+w, y+corner_len), color, 2)
                        cv2.line(result_img, (x, y+h), (x+corner_len, y+h), color, 2)
                        cv2.line(result_img, (x, y+h), (x, y+h-corner_len), color, 2)
                        cv2.line(result_img, (x+w, y+h), (x+w-corner_len, y+h), color, 2)
                        cv2.line(result_img, (x+w, y+h), (x+w, y+h-corner_len), color, 2)
                    
                    # Add text
                    if class_name == "NO_FACE":
                        text = "No Face Detected"
                    else:
                        text = f"{class_name} ({confidence:.1f}%)"
                    
                    # Background for text
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    cv2.rectangle(result_img, (10, 10), (10 + text_size[0] + 20, 50), 
                                 (0, 0, 0), -1)
                    cv2.rectangle(result_img, (10, 10), (10 + text_size[0] + 20, 50), 
                                 color, 2)
                    cv2.putText(result_img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9, color, 2)
                    
                    with col2:
                        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                                caption="Detection Result", use_column_width=True)
                        
                        # Show result
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
    
    # Tab 2: Live Webcam Detection
    with tab2:
        st.markdown("### Live Webcam Detection")
        
        camera_id = st.number_input("Camera ID", min_value=0, max_value=5, value=0, step=1)
        
        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("▶️ Start Webcam")
        with col2:
            stop_btn = st.button("⏹️ Stop Webcam")
        
        # Video placeholder
        video_placeholder = st.empty()
        info_placeholder = st.empty()
        fps_placeholder = st.empty()
        
        if start_btn and st.session_state.detector:
            # Update detector settings
            st.session_state.detector.threshold = threshold
            st.session_state.detector.use_face_detection = use_face_detection
            
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                st.error("Could not open webcam")
            else:
                frame_count = 0
                fps_start = time.time()
                
                while not stop_btn:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to grab frame")
                        break
                    
                    # Mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Make prediction
                    class_name, confidence, color, bbox, face_conf = \
                        st.session_state.detector.predict_frame(frame)
                    
                    # Update stats in real-time
                    if class_name != "NO_FACE":
                        st.session_state.total_predictions += 1
                        if class_name == "REAL":
                            st.session_state.real_count += 1
                        elif class_name == "SPOOF":
                            st.session_state.spoof_count += 1
                    
                    # Draw on frame
                    result_frame = frame.copy()
                    
                    # Draw bounding box
                    if bbox is not None:
                        x, y, w, h = bbox
                        cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, 2)
                        
                        # Add corner markers
                        corner_len = 15
                        cv2.line(result_frame, (x, y), (x+corner_len, y), color, 2)
                        cv2.line(result_frame, (x, y), (x, y+corner_len), color, 2)
                        cv2.line(result_frame, (x+w, y), (x+w-corner_len, y), color, 2)
                        cv2.line(result_frame, (x+w, y), (x+w, y+corner_len), color, 2)
                        cv2.line(result_frame, (x, y+h), (x+corner_len, y+h), color, 2)
                        cv2.line(result_frame, (x, y+h), (x, y+h-corner_len), color, 2)
                        cv2.line(result_frame, (x+w, y+h), (x+w-corner_len, y+h), color, 2)
                        cv2.line(result_frame, (x+w, y+h), (x+w, y+h-corner_len), color, 2)
                    
                    # Add text
                    if class_name == "NO_FACE":
                        text = "No Face Detected"
                    else:
                        text = f"{class_name} ({confidence:.1f}%)"
                    
                    # Background for text
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    cv2.rectangle(result_frame, (10, 10), (10 + text_size[0] + 20, 50), 
                                 (0, 0, 0), -1)
                    cv2.rectangle(result_frame, (10, 10), (10 + text_size[0] + 20, 50), 
                                 color, 2)
                    cv2.putText(result_frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.9, color, 2)
                    
                    # Calculate FPS
                    frame_count += 1
                    if time.time() - fps_start >= 1.0:
                        fps = frame_count
                        fps_placeholder.metric("FPS", f"{fps}")
                        frame_count = 0
                        fps_start = time.time()
                    
                    # Update display
                    video_placeholder.image(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                                           channels="RGB", use_column_width=True)
                    
                    # Update info
                    if class_name == "REAL":
                        info_placeholder.success(f"✅ {class_name} ({confidence:.1f}%)")
                    elif class_name == "SPOOF":
                        info_placeholder.error(f"⚠️ {class_name} ({confidence:.1f}%)")
                    else:
                        info_placeholder.warning("❓ No Face Detected")
                    
                    time.sleep(0.03)
                
                cap.release()
                video_placeholder.empty()
                info_placeholder.empty()
                fps_placeholder.empty()

if __name__ == "__main__":
    main()
