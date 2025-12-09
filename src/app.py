import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

# Setup
st.set_page_config(page_title="ASL Recognition", layout="centered")
IMG_SIZE = (128, 128)

@st.cache_resource
def load_resources():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "model", "asl_detection_model.h5")
    indices_path = os.path.join(base_dir, "model", "class_indices.pkl")

    if not os.path.exists(model_path) or not os.path.exists(indices_path):
        return None, None

    model = tf.keras.models.load_model(model_path)
    
    with open(indices_path, "r") as f:
        class_indices = json.load(f)
        # Convert keys back to integers (JSON stores keys as strings)
        class_indices = {int(k): v for k, v in class_indices.items()}
        
    return model, class_indices

# UI Layout
st.title("🤟 ASL Sign Language Detection")
st.markdown("Upload a hand sign image (A-Z, Space, Delete, Nothing) to detect it.")

model, class_indices = load_resources()

if model is None:
    st.error("⚠️ Model files not found! Please run `src/train.py` first.")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocessing
        img = image.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with col2:
            st.write("### Prediction")
            if st.button("Analyze Sign"):
                with st.spinner("Processing..."):
                    preds = model.predict(img_array)
                    confidence = np.max(preds)
                    predicted_idx = np.argmax(preds)
                    predicted_label = class_indices.get(predicted_idx, "Unknown")

                st.success(f"**{predicted_label}**")
                st.progress(float(confidence))
                st.caption(f"Confidence: {confidence*100:.2f}%")
