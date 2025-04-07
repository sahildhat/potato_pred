import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import requests

# Load saved model
model = tf.keras.models.load_model('new_potato_final.h5')

# Load treatment data
with open('treatments.json') as f:
    treatments = json.load(f)

# Class names mapping
class_names = ['Early_Blight', 'Healthy', 'Late_Blight']

# Preprocess function
def preprocess_image(img):
    if isinstance(img, str):  # Handle case where file path is passed
        img = Image.open(img)
    img = img.resize((256, 256))  # Match training size
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit app configuration
st.set_page_config(page_title="Potato Disease Diagnosis & Treatment", page_icon="🥔")

# Dark mode styling
st.markdown(
    """
    <style>
    body { background-color: #121212; color: white; }
    .stApp { background-color: #121212; }
    </style>
    """,
    unsafe_allow_html=True,
)

# App header
st.title("🥔 Potato Disease Diagnosis & Treatment")
st.markdown("---")

st.header("Potato Leaf Disease Prediction")
st.write("Upload an image of a potato leaf for disease diagnosis.")

# Real-time Webcam Capture
image_data = st.camera_input("Take a photo of the potato leaf")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file or image_data:
    img = Image.open(uploaded_file) if uploaded_file else Image.open(image_data)
    st.image(img, caption='Uploaded Leaf', use_container_width=True)
    
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    confidence = np.max(predictions[0]) * 100
    predicted_class = class_names[np.argmax(predictions[0])]
    
    st.subheader("Prediction Results")
    st.write(f"**Predicted Disease:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    
    # Confidence Bar Chart
    fig = plt.figure(figsize=(10, 4))
    plt.bar(class_names, predictions[0] * 100)
    plt.xticks(rotation=45)
    plt.ylabel('Confidence (%)')
    plt.ylim(0, 100)
    st.pyplot(fig)
    
    # Display treatment recommendations
    if predicted_class in treatments:
        st.subheader("Recommended Treatment")
        details = treatments[predicted_class]
        
        for key, value in details.items():
            if key == "Care and Management":
                st.write("### Care and Management")
                
                if "Fungicides" in value:
                    st.write("#### Fungicides")
                    for fungicide in value["Fungicides"]:
                        st.markdown(f"- **{fungicide['name']}**")
                        if "image" in fungicide and fungicide["image"]:
                            st.image(fungicide["image"], caption=fungicide["name"], width=150)
                        if "usage" in fungicide:
                            st.markdown(f"  - *Usage:* {fungicide['usage']}")
                        if "dosage" in fungicide:
                            st.markdown(f"  - *Dosage:* {fungicide['dosage']}")
                        if "link" in fungicide:
                            st.markdown(f"  - [More Info]({fungicide['link']})")
                
                if "Cultural Practices" in value:
                    st.write("#### Cultural Practices")
                    for practice in value["Cultural Practices"]:
                        st.markdown(f"- {practice}")
            else:
                st.write(f"### {key}")
                if isinstance(value, list):
                    for item in value:
                        st.markdown(f"- {item}")
                else:
                    st.markdown(value.strip())
    
    # Weather and Disease Risk Analysis
    API_KEY = "937e2835a845f822f2f64128e0eb75b6"  # Replace with your API key
    st.write("### **ENTER YOUR LOCATION FOR DISEASE RISK ANALYSIS:**")
    city = st.text_input("Enter your location:")
    
    if city:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url).json()
        temp = response['main']['temp']
        humidity = response['main']['humidity']
        st.write(f"Temperature: {temp}°C, Humidity: {humidity}%")
    
        # Enhanced Disease Risk Warnings
        if humidity > 85 and temp > 20:
            st.warning("High humidity and warm temperatures create ideal conditions for fungal infections like Late Blight.")

        if humidity < 50:
            st.info("Low humidity reduces disease risk, but dry conditions may stress the plants.")

        if temp < 10:
            st.warning("Cold temperatures can slow plant growth and make them more susceptible to disease.")

        if temp > 30:
            st.warning("Extreme heat may weaken the plant, making it more vulnerable to pests and diseases.")

        if humidity > 90:
            st.error("Very high humidity can lead to rapid disease spread. Consider improving air circulation around plants.")

        if 60 <= humidity <= 80 and 15 <= temp <= 25:
            st.success("Current weather conditions are optimal for healthy potato growth.")
    
    # Integration with E-commerce for Treatment Products
    st.write("### **LINK TO BUY**")
    st.markdown("[Buy Fungicides on Amazon](https://www.amazon.com/s?k=fungicides)")

st.markdown("*Always follow label instructions and wear proper protective equipment when applying treatments.*")
