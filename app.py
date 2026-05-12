import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("model.h5")

# Class labels (EDIT if your dataset differs)
classes = [
    "Speed Limit 20", "Speed Limit 30", "Speed Limit 50",
    "Speed Limit 60", "Speed Limit 70", "Speed Limit 80",
    "End Speed Limit", "No Entry", "Stop"
]

st.title("🚦 Traffic Sign Recognition AI")
st.write("Upload a traffic sign image to predict")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

def preprocess(img):
    img = np.array(img)
    img = cv2.resize(img, (64, 64))   # IMPORTANT: match your model input size
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed = preprocess(image)

    prediction = model.predict(processed)
    class_index = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {classes[class_index]}")
    st.info(f"Confidence: {confidence:.2f}%")