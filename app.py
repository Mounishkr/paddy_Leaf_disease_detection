import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model("paddy_leaf_disease_model.h5")

# Preprocess function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# App Title
st.title("Paddy Leaf Disease Detection (Mounish Kumar)")

# File Upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    # Get class name
    class_names = ['blast', 'blb', 'healthy', 'hispa', 'leaf_folder', 'leaf_spot']
    predicted_class = class_names[np.argmax(predictions)]
    
    st.write(f"Predicted Disease: {predicted_class}")
