import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Traffic Sign AI", page_icon="🚦")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# -----------------------------
# Class labels (EDIT if needed)
# -----------------------------
class_names = [
    "Speed Limit 20",
    "Speed Limit 30",
    "Speed Limit 50",
    "Speed Limit 60",
    "Speed Limit 70",
    "Speed Limit 80",
    "End of Speed Limit",
    "Stop",
    "No Entry"
]

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_image(image):
    image = np.array(image)

    # Convert RGB → OpenCV format if needed
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image = cv2.resize(image, (32, 32))   # ⚠️ MUST match training size
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image

# -----------------------------
# UI
# -----------------------------
st.title("🚦 Traffic Sign Recognition AI")
st.write("Upload a traffic sign image and get prediction from the trained model.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    processed = preprocess_image(image)

    # Predict
    prediction = model.predict(processed)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Output
    st.subheader("Prediction Result")

    st.success(f"🚦 Class: {class_names[class_index]}")
    st.info(f"🎯 Confidence: {confidence * 100:.2f}%")