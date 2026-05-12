import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

from PIL import Image

# Page settings
st.set_page_config(
    page_title="AI Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)

# Load model only once
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model(
        'model/traffic_model.h5'
    )
    return model

model = load_my_model()

# Load labels
labels = pd.read_csv('labels.csv')

# Title
st.title("🚦 AI Traffic Sign Recognition")

st.write("Upload a traffic sign image and AI will predict the sign.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose Traffic Sign Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Show image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Convert image
    img = np.array(image)

    # Resize image
    img = cv2.resize(img, (30, 30))

    # Expand dimensions
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    # Get class
    predicted_class = np.argmax(prediction)

    # Confidence
    confidence = np.max(prediction) * 100

    # Get sign name safely
    try:
        sign_name = labels.loc[
            labels['ClassId'] == predicted_class,
            'Name'
        ].values[0]
    except:
        sign_name = f"Class {predicted_class}"

    # Results
    st.success(f"🚦 Detected Sign: {sign_name}")

    st.info(f"📊 Confidence: {confidence:.2f}%")

    # Show prediction array
    st.subheader("Prediction Details")

    st.write(prediction)

# Footer
st.markdown("---")
st.caption("AI Traffic Sign Recognition using CNN + Streamlit")