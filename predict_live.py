"""
Live webcam prediction script with MTCNN face detection
Usage: python src/predict_live.py
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from mtcnn import MTCNN
import argparse
import time
from collections import deque
from pathlib import Path

# ===== YOUR EXACT PATHS =====
DEFAULT_MODEL = "best_model.h5"
# =============================

class LiveSpoofDetector:
    def __init__(self, model_path, threshold=0.5, use_face_detection=True, 
                 smooth_frames=5, camera_id=0):
        """
        Initialize live spoof detector with webcam
        """
        print("\n" + "="*60)
        print("🎥 LIVE SPOOF DETECTION - INITIALIZING")
        print("="*60)
        
        # Load model
        print(f"📂 Loading model from: {model_path}")
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
        
        # Initialize face detector
        print("🔍 Initializing MTCNN face detector...")
        self.face_detector = MTCNN()
        print("✅ Face detector ready")
        
        # Settings
        self.threshold = threshold
        self.use_face_detection = use_face_detection
        self.target_size = (224, 224)
        self.camera_id = camera_id
        
        # Smoothing
        self.confidence_history = deque(maxlen=smooth_frames)
        self.smoothed_confidence = 0.5
        
        # FPS calculation
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Stats
        self.total_frames = 0
        self.real_count = 0
        self.spoof_count = 0
        self.no_face_count = 0
        
        print(f"\n⚙️ Configuration:")
        print(f"  Threshold: {self.threshold}")
        print(f"  Face detection: {'ON' if use_face_detection else 'OFF'}")
        print(f"  Smoothing frames: {smooth_frames}")
        print(f"  Camera ID: {camera_id}")
        print("="*60)

    def detect_face(self, frame):
        """Detect face using MTCNN"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_detector.detect_faces(rgb_frame)
        
        if not faces:
            return None, None
        
        # Get largest face
        largest_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = largest_face['box']
        
        # Add margin (20%)
        margin = int(max(w, h) * 0.2)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(frame.shape[1] - x, w + 2 * margin)
        h = min(frame.shape[0] - y, h + 2 * margin)
        
        return (x, y, w, h), largest_face['confidence']

    def resize_with_padding(self, image):
        """Resize with padding to target size"""
        h, w = image.shape[:2]
        
        # Calculate scale
        scale = min(self.target_size[0]/w, self.target_size[1]/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((self.target_size[1], self.target_size[0], 3), 
                         dtype=np.uint8)
        x_offset = (self.target_size[0] - new_w) // 2
        y_offset = (self.target_size[1] - new_h) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded

    def preprocess_face(self, face_img):
        """Preprocess face for model input"""
        # Resize with padding
        processed = self.resize_with_padding(face_img)
        
        # Convert to RGB if needed
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        elif processed.shape[2] == 4:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGRA2RGB)
        elif processed.shape[2] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # Apply MobileNetV3 preprocessing
        processed = preprocess_input(processed.astype(np.float32))
        
        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)
        
        return processed

    def predict_frame(self, frame):
        """Make prediction on a single frame"""
        bbox = None
        face_conf = 0
        
        if self.use_face_detection:
            # Detect face
            bbox, face_conf = self.detect_face(frame)
            
            if bbox is not None and face_conf > 0.9:
                x, y, w, h = bbox
                face_img = frame[y:y+h, x:x+w]
                
                # Preprocess and predict
                processed = self.preprocess_face(face_img)
                prediction = self.model.predict(processed, verbose=0)[0][0]
                
                # Update stats
                self.total_frames += 1
            else:
                # No face detected
                self.no_face_count += 1
                self.total_frames += 1
                return "NO_FACE", 0, (128, 128, 128), None, 0
        else:
            # Use full frame
            processed = self.preprocess_face(frame)
            prediction = self.model.predict(processed, verbose=0)[0][0]
            self.total_frames += 1
        
        # Smooth prediction
        self.confidence_history.append(prediction)
        self.smoothed_confidence = np.mean(self.confidence_history)
        
        # Determine class
        if self.smoothed_confidence > self.threshold:
            class_name = "SPOOF"
            confidence = self.smoothed_confidence * 100
            color = (0, 0, 255)  # Red
            self.spoof_count += 1
        else:
            class_name = "REAL"
            confidence = (1 - self.smoothed_confidence) * 100
            color = (0, 255, 0)  # Green
            self.real_count += 1
        
        return class_name, confidence, color, bbox, face_conf

    def update_fps(self):
        """Calculate FPS"""
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()

    def draw_info(self, frame, class_name, confidence, color, bbox, face_conf):
        """Draw all information on frame"""
        overlay = frame.copy()
        
        # Draw face bounding box
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Add corner markers
            corner_len = 15
            # Top-left
            cv2.line(frame, (x, y), (x+corner_len, y), color, 2)
            cv2.line(frame, (x, y), (x, y+corner_len), color, 2)
            # Top-right
            cv2.line(frame, (x+w, y), (x+w-corner_len, y), color, 2)
            cv2.line(frame, (x+w, y), (x+w, y+corner_len), color, 2)
            # Bottom-left
            cv2.line(frame, (x, y+h), (x+corner_len, y+h), color, 2)
            cv2.line(frame, (x, y+h), (x, y+h-corner_len), color, 2)
            # Bottom-right
            cv2.line(frame, (x+w, y+h), (x+w-corner_len, y+h), color, 2)
            cv2.line(frame, (x+w, y+h), (x+w, y+h-corner_len), color, 2)
        
        # Semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Main prediction
        if class_name == "NO_FACE":
            cv2.putText(frame, "No Face Detected", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            text = f"{class_name} ({confidence:.1f}%)"
            cv2.putText(frame, text, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Confidence bar
        bar_x, bar_y = 20, 80
        bar_w, bar_h = 250, 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), 
                     (100, 100, 100), 1)
        fill_w = int(bar_w * (confidence / 100))
        if class_name == "REAL":
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), 
                         (0, 255, 0), -1)
        elif class_name == "SPOOF":
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), 
                         (0, 0, 255), -1)
        
        # Stats
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Threshold: {self.threshold:.2f}", (20, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if face_conf > 0:
            cv2.putText(frame, f"Face conf: {face_conf:.2f}", (20, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Counters
        cv2.putText(frame, f"Real: {self.real_count}", (20, 175),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.putText(frame, f"Spoof: {self.spoof_count}", (120, 175),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Instructions
        instructions = [
            "'q' - Quit",
            "'s' - Save screenshot",
            "'+/-' - Adjust threshold",
            "'f' - Toggle face detection"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, 200 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Face detection status
        status = "Face Detection: ON" if self.use_face_detection else "Face Detection: OFF"
        status_color = (0, 255, 0) if self.use_face_detection else (0, 0, 255)
        cv2.putText(frame, status, (20, 290),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        return frame

    def run(self):
        """Main loop for live prediction"""
        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n🎥 Live prediction started!")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  '+' - Increase threshold")
        print("  '-' - Decrease threshold")
        print("  'f' - Toggle face detection")
        print("-" * 40)
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Failed to grab frame")
                    break
                
                # Flip horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Make prediction
                class_name, confidence, color, bbox, face_conf = self.predict_frame(frame)
                
                # Update FPS
                self.update_fps()
                
                # Draw information
                frame = self.draw_info(frame, class_name, confidence, color, 
                                      bbox, face_conf)
                
                # Show frame
                cv2.imshow('Live Spoof Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                    
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = int(time.time())
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Screenshot saved: {filename}")
                    
                elif key == ord('+') or key == ord('='):
                    self.threshold = min(1.0, self.threshold + 0.05)
                    print(f"📈 Threshold: {self.threshold:.2f}")
                    
                elif key == ord('-') or key == ord('_'):
                    self.threshold = max(0.0, self.threshold - 0.05)
                    print(f"📉 Threshold: {self.threshold:.2f}")
                    
                elif key == ord('f'):
                    self.use_face_detection = not self.use_face_detection
                    status = "ON" if self.use_face_detection else "OFF"
                    print(f"👤 Face detection: {status}")
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
            
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            
            # Print session stats
            print("\n" + "="*60)
            print("📊 SESSION STATISTICS")
            print("="*60)
            print(f"Total frames: {self.total_frames}")
            print(f"Real detected: {self.real_count}")
            print(f"Spoof detected: {self.spoof_count}")
            print(f"No face: {self.no_face_count}")
            print(f"Average FPS: {self.fps:.1f}")
            print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Live webcam spoof detection')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='Path to model file')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--no-face-detection', action='store_true',
                        help='Disable face detection')
    parser.add_argument('--smooth', type=int, default=5,
                        help='Number of frames for smoothing (default: 5)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        # Try alternative paths
        alt_model = os.path.join(MODELS_DIR, 'best_model_phase1.h5')
        if os.path.exists(alt_model):
            print(f"✅ Found alternative model: {alt_model}")
            args.model = alt_model
        else:
            return
    
    # Create and run detector
    try:
        detector = LiveSpoofDetector(
            model_path=args.model,
            threshold=args.threshold,
            use_face_detection=not args.no_face_detection,
            smooth_frames=args.smooth,
            camera_id=args.camera
        )
        detector.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    main()
