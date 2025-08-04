import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import os
import warnings
warnings.filterwarnings("ignore")

st.markdown("""
<h1 style='text-align: center; color: green;'>🌾 Paddy Disease Classifier</h1>

---

### 👨‍💻 Developed by: [**Muhammad Haweras**]

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammad-haweras-7aa6b11b2/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white)](https://github.com/MuhammadHaweras)

---
### 🌱 What is *Paddy*?

Paddy refers to the **rice plant** in its raw, unprocessed form. It is one of the most essential crops in the world, especially in Asia. Paddy diseases significantly affect crop yield and quality, and early detection using AI can help farmers take action quickly.

---
""", unsafe_allow_html=True)

model_path = "rice_disease_model.keras"
model = tf.keras.models.load_model(model_path)

class_names = [
    'bacterial_leaf_blight',
    'bacterial_leaf_streak',
    'bacterial_panicle_blight',
    'blast',
    'brown_spot',
    'dead_heart',
    'downy_mildew',
    'hispa',
    'normal',
    'tungro'
]

st.markdown('<h3>🌿 Upload a Rice Leaf Image to Diagnose Disease</h3>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image of a rice leaf...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((300, 300))
    img_array = image.img_to_array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    img_batch = preprocess_input(img_batch)

    # Prediction
    preds = model.predict(img_batch, verbose=0)
    predicted_index = np.argmax(preds)
    predicted_label = class_names[predicted_index]
    confidence = np.max(preds) * 100

    st.image(img, caption="📷 Uploaded Image", use_container_width =True)
    st.success(f"🧠 **Prediction:** {predicted_label}")
    st.info(f"✅ **Confidence:** {confidence:.2f}%")
