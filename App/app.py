import streamlit as st
from inference import predict_scanner, predict_tamper_patch, predict_tamper_pair
from PIL import Image
import pandas as pd

# -----------------
# Page config
# -----------------
st.set_page_config(
    page_title="🖨 AI Trace Finder",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🖨 AI Trace Finder - Scanner & Tamper Detection")
st.markdown("Upload scanned image(s) to detect the **scanner model** and check for **tampering** 🕵️‍♀️🔍")

# Sidebar for instructions
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. Upload a single image to detect the scanner and patch tampering.
    2. Upload two images to check for pairwise tampering.
    3. Results include:
       - Predicted scanner model 🖨
       - Tamper detection ✅ / ❌
       - Confidence scores 📊
    """)

# -----------------
# Single Image Analysis
# -----------------
st.header("🖼 Single Image Scanner Prediction + Patch Tamper Detection")
uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"], key="single_upload")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔎 Analyze Single Image"):
        # Save temporarily
        image_path = "temp_image.png"
        image.save(image_path)

        # Scanner prediction
        scanner = predict_scanner(image_path)
        st.success(f"🖨 Predicted Scanner Model: **{scanner}**")

        # Patch tamper prediction
        tamper_patch, score_patch = predict_tamper_patch(image_path)
        status = "❌ Tampered" if tamper_patch else "✅ Original"
        st.warning(f"🛡 Patch Tamper Detection: **{status}** (Score: {score_patch:.3f})")

        # Dashboard-style metrics
        st.subheader("📊 Single Image Metrics")
        df_metrics = pd.DataFrame({
            "Metric": ["Patch Tamper Score"],
            "Value": [score_patch]
        })
        st.bar_chart(df_metrics.set_index("Metric"))

# -----------------
# Pairwise Tamper Detection
# -----------------
st.header("🔗 Pairwise Tamper Detection")
uploaded_files = st.file_uploader(
    "Upload Two Images", type=["png","jpg","jpeg"], accept_multiple_files=True, key="pair_upload"
)

if uploaded_files and len(uploaded_files) == 2:
    image1 = Image.open(uploaded_files[0])
    image2 = Image.open(uploaded_files[1])
    st.image([image1, image2], caption=["Image 1", "Image 2"], width=300)

    if st.button("🔎 Analyze Pairwise Tamper"):
        # Save temporarily
        uploaded_files[0].save("temp_image1.png")
        uploaded_files[1].save("temp_image2.png")

        tamper_pair, score_pair = predict_tamper_pair("temp_image1.png", "temp_image2.png")
        status_pair = "❌ Tampered" if tamper_pair else "✅ Original"
        st.warning(f"🛡 Pairwise Tamper Detection: **{status_pair}** (Score: {score_pair:.3f})")

        # Pairwise dashboard
        st.subheader("📊 Pairwise Metrics")
        df_pair_metrics = pd.DataFrame({
            "Metric": ["Pairwise Tamper Score"],
            "Value": [score_pair]
        })
        st.bar_chart(df_pair_metrics.set_index("Metric"))

# -----------------
# Footer
# -----------------
st.markdown("---")
st.markdown("Developed with ❤️ using **Streamlit & TensorFlow**")
