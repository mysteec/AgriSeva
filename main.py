import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('trained_model.keras')


# =========================
# PREDICTION FUNCTION
# =========================
def model_prediction(test_image):
    model = load_model()

    image = Image.open(test_image)
    image = image.resize((128, 128))

    input_arr = np.array(image) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)

    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence = np.max(prediction)

    return result_index, confidence


# =========================
# SIDEBAR
# =========================
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])


# =========================
# HOME PAGE
# =========================
if app_mode == "Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")

    image_path = "home_page.jpeg"

    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning("Home image not found.")

    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases.

    ### How It Works
    1. Upload Image on **Disease Recognition** page  
    2. Model analyzes the image  
    3. Get prediction + confidence  

    ### Why Choose Us?
    ✔ Accurate ML model  
    ✔ Simple UI  
    ✔ Fast predictions  

    👉 Go to **Disease Recognition** to get started!
    """)


# =========================
# ABOUT PAGE
# =========================
elif app_mode == "About":
    st.header("About Dataset")

    st.markdown("""
    This dataset is recreated using offline augmentation.

    📊 Dataset Details:
    - Train: 70,295 images  
    - Validation: 17,572 images  
    - Test: 33 images  
    - Total Classes: 38  

    It contains healthy and diseased crop leaves.
    """)


# =========================
# DISEASE RECOGNITION PAGE
# =========================
elif app_mode == "Disease Recognition":
    st.header("🔍 Disease Recognition")

    test_image = st.file_uploader("📤 Upload a plant leaf image")

    # Show image immediately
    if test_image is not None:
        st.image(test_image, use_container_width=True)

    # Class labels
    class_name = [
        'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
        'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy',
        'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot','Peach___healthy',
        'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
        'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
        'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch','Strawberry___healthy',
        'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
        'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
    ]

    # Predict button
    if st.button("🔍 Predict"):
        if test_image is None:
            st.warning("⚠️ Please upload an image first!")
        else:
            with st.spinner("Analyzing Image..."):
                result_index, confidence = model_prediction(test_image)

                st.success(f"🌱 Prediction: {class_name[result_index]}")
                st.info(f"📊 Confidence: {confidence * 100:.2f}%")