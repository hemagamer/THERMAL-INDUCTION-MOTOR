import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model/kaggle/input/thermal-images-of-induction-motor
model = tf.keras.models.load_model('my_model.h5')

# Define the new class labels
class_labels = ['Fan', 'A&C10', 'A&B50', 'A&C30', 'Noload', 'Rotor-0', 'A&C&B30', 'A30', 'A&C&B10', 'A50', 'A10']

# Streamlit app
st.title("thermal-images-of-induction-motor Detection with CNN")
st.write("Upload an image, and the model will predict the type of engine.")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["bmp"])

# If an image is uploaded
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    image = image.resize((320, 240))  # Resize to 320x240
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(image)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence_scores = predictions[0]

    # Display results
    st.write("### Prediction Results")
    st.write(f"Predicted Class: {predicted_class}")

    # Display confidence scores
    st.write("### Confidence Scores")
    for label, score in zip(class_labels, confidence_scores):
        st.write(f"{label}: {score:.2%}")