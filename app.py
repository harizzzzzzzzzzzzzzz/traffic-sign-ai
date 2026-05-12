import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

MODEL_PATH = "model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

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

def preprocess(img):
    img = np.array(img)
    img = cv2.resize(img, (32, 32))   # ⚠️ match your training size
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title("🚦 Traffic Sign Recognition AI")

file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img)

    processed = preprocess(img)
    pred = model.predict(processed)

    class_id = np.argmax(pred)
    confidence = np.max(pred)

    st.success(f"Prediction: {class_names[class_id]}")
    st.info(f"Confidence: {confidence*100:.2f}%")