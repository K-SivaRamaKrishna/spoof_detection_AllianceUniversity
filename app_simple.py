# app_simple.py - Enhanced with WebRTC, better UI, loading animations, and history

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
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Custom button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Confidence bar animation */
    @keyframes slideIn {
        from {
            width: 0%;
        }
        to {
            width: var(--target-width);
        }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Success/Error cards */
    .real-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    
    .spoof-card {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Loading spinner */
    .loading-spinner {
        text-align: center;
        padding: 2rem;
    }
    
    /* History table styling */
    .history-table {
        max-height: 400px;
        overflow-y: auto;
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
if 'processing' not in st.session_state:
    st.session_state.processing = False

class VideoTransformer(VideoTransformerBase):
    """Video transformer for real-time face spoof detection"""
    
    def __init__(self):
        self.detector = None
        self.threshold = 0.5
        self.use_face_detection = True
        self.frame_count = 0
        
    def set_detector(self, detector, threshold, use_face_detection):
        self.detector = detector
        self.threshold = threshold
        self.use_face_detection = use_face_detection
    
    def recv(self, frame):
        if self.detector is None:
            return frame
        
        # Get frame as numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Process every few frames for performance
        self.frame_count += 1
        if self.frame_count % 3 == 0:  # Process every 3rd frame
            # Make prediction
            class_name, confidence, color, bbox, face_conf = self.detector.predict_frame(img)
            
            # Draw bounding box
            if bbox is not None:
                x, y, w, h = bbox
                # Draw rectangle
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
            
            # Add text
            if class_name == "NO_FACE":
                text = "❌ No Face Detected"
                bg_color = (128, 128, 128)
            else:
                text = f"{class_name} ({confidence:.1f}%)"
                bg_color = color
            
            # Background for text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(img, (10, 10), (10 + text_size[0] + 20, 45), (0, 0, 0), -1)
            cv2.rectangle(img, (10, 10), (10 + text_size[0] + 20, 45), bg_color, 2)
            cv2.putText(img, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bg_color, 2)
            
            # Confidence bar
            if class_name != "NO_FACE":
                bar_x, bar_y = 10, 55
                bar_w, bar_h = 200, 10
                cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (100, 100, 100), 1)
                fill_w = int(bar_w * (confidence / 100))
                cv2.rectangle(img, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), bg_color, -1)
            
            # Update session state for display
            st.session_state.latest_prediction = {
                'class': class_name,
                'confidence': confidence,
                'face_conf': face_conf,
                'timestamp': datetime.now()
            }
            
            # Update statistics
            if class_name != "NO_FACE":
                st.session_state.total_predictions += 1
                if class_name == "REAL":
                    st.session_state.real_count += 1
                else:
                    st.session_state.spoof_count += 1
                
                # Add to history
                st.session_state.prediction_history.insert(0, {
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'class': class_name,
                    'confidence': confidence,
                    'face_conf': face_conf if face_conf else 0
                })
                # Keep only last 20
                st.session_state.prediction_history = st.session_state.prediction_history[:20]
        
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

def create_confidence_bar(confidence, class_name):
    """Create a custom confidence bar"""
    color = "#10b981" if class_name == "REAL" else "#ef4444"
    return f"""
    <div style="background: #e5e7eb; border-radius: 10px; padding: 2px; margin: 10px 0;">
        <div style="background: {color}; width: {confidence}%; border-radius: 8px; padding: 8px; text-align: center; color: white;">
            {confidence:.1f}% Confidence
        </div>
    </div>
    """

def main():
    # Header with logo and title
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.markdown("## 🎭")
    with col_title:
        st.markdown("""
        <div class="main-header">
            <h1 style="color: white; margin: 0;">Face Spoof Detection System</h1>
            <p style="color: white; margin: 0; opacity: 0.9;">Advanced Deep Learning for Presentation Attack Detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100?text=Face+Detection+AI", use_column_width=True)
        st.markdown("## ⚙️ Settings")
        
        model_path = st.text_input("Model Path", value=str(DEFAULT_MODEL))
        threshold = st.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.05, 
                             help="Adjust sensitivity: Lower = more REAL, Higher = more SPOOF")
        use_face_detection = st.checkbox("Enable Face Detection", value=True,
                                        help="Detect and crop faces before classification")
        
        # Initialize detector button with loading state
        if st.button("🚀 Initialize Detector", use_container_width=True):
            st.session_state.processing = True
            with st.spinner("🔄 Loading model and initializing detector..."):
                st.session_state.detector = init_detector(
                    model_path, threshold, use_face_detection
                )
                if st.session_state.detector:
                    st.success("✅ Detector initialized successfully!")
                    st.balloons()
                else:
                    st.error("❌ Failed to initialize detector")
            st.session_state.processing = False
        
        # Check if model exists
        if not st.session_state.detector:
            if os.path.exists(model_path):
                st.info(f"✅ Model found at: {os.path.basename(model_path)}\nClick 'Initialize Detector' to start")
            else:
                st.error(f"❌ Model not found")
                st.info("Please ensure 'best_model.h5' is in the same directory")
        
        st.markdown("---")
        
        # Statistics Panel
        st.markdown("### 📊 Live Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Predictions", st.session_state.total_predictions)
        with col2:
            st.metric("✅ Real", st.session_state.real_count, delta_color="normal")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("⚠️ Spoof", st.session_state.spoof_count, delta_color="inverse")
        with col2:
            if st.session_state.total_predictions > 0:
                accuracy = (st.session_state.real_count / st.session_state.total_predictions) * 100
                st.metric("Real %", f"{accuracy:.1f}%")
        
        if st.button("📈 Reset Statistics", use_container_width=True):
            st.session_state.total_predictions = 0
            st.session_state.real_count = 0
            st.session_state.spoof_count = 0
            st.session_state.prediction_history = []
            st.success("Statistics reset!")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **Model Architecture:** MobileNetV3  
        **Face Detector:** MTCNN  
        **Framework:** TensorFlow 2.13  
        
        **Detects:**
        - 📸 Print attacks
        - 📱 Replay attacks  
        - 🎭 Mask attacks
        - 💻 Digital display attacks
        """)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📸 Single Image", "🎥 Live Webcam", "📊 History & Analytics"])
    
    # Tab 1: Single Image Detection
    with tab1:
        st.markdown("## 📸 Single Image Detection")
        st.markdown("Upload an image for detailed spoof analysis")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            # Read the image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                        caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                if st.session_state.detector is None:
                    st.warning("⚠️ Please initialize the detector first!")
                else:
                    with st.spinner("🔄 Analyzing image..."):
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
                        st.session_state.prediction_history.insert(0, {
                            'timestamp': datetime.now().strftime("%H:%M:%S"),
                            'class': class_name,
                            'confidence': confidence,
                            'face_conf': face_conf if face_conf else 0
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
                            bar_w, bar_h = 300, 25
                            cv2.rectangle(result_img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), 
                                         (100, 100, 100), 2)
                            fill_w = int(bar_w * (confidence / 100))
                            cv2.rectangle(result_img, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), 
                                         bg_color, -1)
                            
                            # Add confidence text
                            cv2.putText(result_img, f"Confidence: {confidence:.1f}%", 
                                       (bar_x + bar_w//2 - 60, bar_y + 18), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        with col2:
                            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                                    caption="Analysis Result", use_column_width=True)
                            
                            # Show result cards
                            if class_name == "REAL":
                                st.markdown(f"""
                                <div class="real-card">
                                    <h2>✅ REAL Face</h2>
                                    <p style="font-size: 24px; margin: 10px 0;">{confidence:.1f}% Confidence</p>
                                    <p>This is a GENUINE face with high authenticity</p>
                                </div>
                                """, unsafe_allow_html=True)
                                st.balloons()
                                
                            elif class_name == "SPOOF":
                                st.markdown(f"""
                                <div class="spoof-card">
                                    <h2>⚠️ SPOOF Attack</h2>
                                    <p style="font-size: 24px; margin: 10px 0;">{confidence:.1f}% Confidence</p>
                                    <p>Potential presentation attack detected!</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            else:
                                st.warning("""
                                ### ❌ No Face Detected
                                Could not detect a clear face in the image.
                                
                                **Tips:**
                                - Ensure good lighting
                                - Face should be clearly visible
                                - Try a different image
                                """)
                            
                            if face_conf > 0 and bbox:
                                st.info(f"📊 Face Detection Quality: {face_conf:.2f}")
    
    # Tab 2: Live Webcam Detection with WebRTC
    with tab2:
        st.markdown("## 🎥 Live Webcam Detection")
        st.markdown("Real-time spoof detection using your browser's camera")
        
        st.info("📹 Click 'Start' to access your camera. Works on mobile and desktop!")
        
        if st.session_state.detector is None:
            st.warning("⚠️ Please initialize the detector first!")
        else:
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
            if hasattr(st.session_state, 'latest_prediction') and st.session_state.latest_prediction:
                pred = st.session_state.latest_prediction
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if pred['class'] == "REAL":
                        st.success(f"✅ **Current: {pred['class']}**")
                    elif pred['class'] == "SPOOF":
                        st.error(f"⚠️ **Current: {pred['class']}**")
                    else:
                        st.warning("❓ **No Face Detected**")
                
                with col2:
                    if pred['class'] != "NO_FACE":
                        st.metric("Confidence", f"{pred['confidence']:.1f}%")
                        st.progress(pred['confidence']/100)
                
                with col3:
                    if pred.get('face_conf', 0) > 0:
                        st.metric("Face Quality", f"{pred['face_conf']:.2f}")
    
    # Tab 3: History and Analytics
    with tab3:
        st.markdown("## 📊 Detection History & Analytics")
        
        if len(st.session_state.prediction_history) > 0:
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Detections", len(st.session_state.prediction_history))
            with col2:
                real_count = sum(1 for p in st.session_state.prediction_history if p['class'] == "REAL")
                st.metric("Real Detections", real_count)
            with col3:
                spoof_count = sum(1 for p in st.session_state.prediction_history if p['class'] == "SPOOF")
                st.metric("Spoof Detections", spoof_count)
            with col4:
                avg_conf = np.mean([p['confidence'] for p in st.session_state.prediction_history if p['class'] != "NO_FACE"]) if len(st.session_state.prediction_history) > 0 else 0
                st.metric("Avg Confidence", f"{avg_conf:.1f}%")
            
            st.markdown("---")
            
            # History table
            st.markdown("### Recent Detections")
            
            # Create dataframe for history
            import pandas as pd
            df = pd.DataFrame(st.session_state.prediction_history[:20])
            
            # Add color styling
            def color_class(val):
                if val == "REAL":
                    return 'background-color: #10b981; color: white'
                elif val == "SPOOF":
                    return 'background-color: #ef4444; color: white'
                return ''
            
            styled_df = df.style.applymap(color_class, subset=['class'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Download history
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv,
                file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No detection history yet. Upload an image or use live webcam to see results here.")

if __name__ == "__main__":
    main()
