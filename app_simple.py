# Update requirements.txt (remove av and streamlit-webrtc)
cat > requirements.txt << EOF
streamlit==1.28.0
opencv-python-headless==4.8.1.78
numpy==1.23.5
mtcnn==0.1.1
Pillow==10.0.0
tensorflow-cpu==2.13.0
protobuf==3.20.3
EOF

# Add and commit
git add requirements.txt app_simple.py
git commit -m "Fix: Use only opencv-python-headless, remove conflicting packages"
git push origin main
