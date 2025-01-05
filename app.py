import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Streamlit Page Config (must be the first Streamlit command)
st.set_page_config(
    page_title="Paddy Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource  # Cache the model to avoid reloading
def load_trained_model():
    return load_model("paddy_disease_model.h5")

model = load_trained_model()

# Class labels
class_labels = ['Blast', 'BLB', 'Healthy', 'Hispa', 'Leaf Folder', 'Leaf Spot']

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #626262;
        }
        .main {
            background-color: #626262;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            color: #dddddd;
        }
        .uploaded-image {
            border: 2px solid #dddddd;
            border-radius: 5px;
            margin: 10px 0;
        }
        .stButton>button {
            color: white;
            background-color: #dddddd;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Page Header
st.title("🌾 Paddy Plant Disease Detection")
st.subheader("Upload a thermal image of a paddy leaf to predict the disease.")
st.write("This AI-powered application uses a trained deep learning model to identify diseases in paddy plants based on thermal images.")

# File Upload Section
uploaded_file = st.file_uploader("Upload a thermal image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Image Processing and Prediction
if uploaded_file is not None:
    try:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Process the image
        image = load_img(uploaded_file, target_size=(224, 224))
        image = img_to_array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make prediction
        with st.spinner("Analyzing the image..."):
            prediction = model.predict(image)
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction)

        # Display the results
        st.success("Analysis Complete!")
        st.write(f"### Predicted Disease: **{predicted_class}**")
        st.write(f"### Confidence: **{confidence * 100:.2f}%**")
        
        # Encourage users to take action
        if predicted_class != "Healthy":
            st.warning("⚠️ The leaf appears to have a disease. Consult an agricultural expert.")
        else:
            st.balloons()
            st.success("🎉 The leaf is healthy! Keep up the good work.")
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")

# Footer Section
st.markdown("---")
st.write("Developed by [Mounish](https://github.com/your-github-url) • Powered by Deep Learning")
st.write("🌐 [Project Repository](https://github.com/your-project-repo-url)")
