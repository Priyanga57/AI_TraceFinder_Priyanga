import streamlit as st
import numpy as np
import cv2
from inference import predict_scanner, predict_tamper

st.set_page_config(page_title="🖨️ AI Trace Finder", layout="centered")

# 🎨 Background and header
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #1f1c2c, #928DAB);
        }
        .result-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            margin-top: 15px;
            text-align: center;
            color: white;
            font-family: 'Helvetica Neue';
        }
    </style>
""", unsafe_allow_html=True)

st.title("🖨️ AI Trace Finder")
st.markdown("Upload an image to identify the **scanner model** and check for **tampering** 🔍")

uploaded_file = st.file_uploader("📂 Upload Scanned Image", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file:
    # Display uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Prediction buttons
    if st.button("🚀 Analyze Image"):
        with st.spinner("Running model inference... Please wait ⏳"):
            # --- Scanner prediction ---
            try:
                s_label, s_conf = predict_scanner(img)
                if s_conf < 20:
                    s_label = "Unknown"
            except Exception as e:
                st.error(f"Scanner model error: {e}")
                s_label, s_conf = "Unknown", 0.0

            # --- Tamper prediction ---
            try:
                verdict, prob = predict_tamper(img)
            except Exception as e:
                st.error(f"Tamper model error: {e}")
                verdict, prob = "Clean", 0.0

        # --- Results display ---
        st.markdown("### 📊 Results")
        st.markdown(f"""
        <div class='result-card'>
            <h3>🖨️ Scanner Identification</h3>
            <p><b>Scanner:</b> {s_label}</p>
            <p><b>Confidence:</b> {s_conf:.2f}%</p>
            <progress value="{s_conf}" max="100"></progress>
        </div>
        <div class='result-card'>
            <h3>🕵️ Tamper Detection</h3>
            <p><b>Status:</b> {verdict}</p>
            <p><b>Probability:</b> {prob:.2f}%</p>
            <progress value="{prob}" max="100"></progress>
        </div>
        """, unsafe_allow_html=True)
