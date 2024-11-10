import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
import streamlit as st
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
#import tensorflow as tf
#import tensorflow_hub as hub
import torch
import numpy as np
import cv2
import os
import ssl
from urllib.request import urlopen


# Streamlit app
st.title("Hypospadias Image Classification")
st.write("Upload an image to classify using a pretrained model.")

def main():
# Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Open image using PIL
        image = Image.open(uploaded_file)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)

def predict_class(image):
    tensorflow.keras.models.load_model(r'/content/drive/Mydrive/Ashoka dataset')
    model = tensorflow.keras.Sequential([hub.KerasLayer(classifier_model)])    
    test_image = image.resize((128, 128))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image/255.0
    test_image = np.expand_dims(test_image, axis = 0)
    classnames = ['normal', 
                 'abnormal']    
    predictions = model.predict(test_image)
    scores = tensorflow.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = classnames[np.argmax(scores)]
    return result

if __name__ == '__main__':
    main()               
import ssl
from urllib.request import urlopen


# Load YOLO model
@st.cache_resource
def load_model():
    ssl._create_default_https_context = ssl._create_unverified_context
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)
    return model

# Object Detection function
def detect_objects(image, model):
    # Convert image to numpy array
    img_array = np.array(image)
    # Convert RGB to BGR format (OpenCV standard)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Perform inference
    results = model(img_array)
    # Get detection results
    results_img = np.squeeze(results.render())  # Render the detected results on the image
    
    return results_img

# Streamlit UI
st.title("YOLO Object Detection App")
st.write("Upload an image to perform object detection using a trained YOLO model.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image using PIL
    image = Image.open(uploaded_file)
    
    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Processing...")
    
    # Load model
    model = load_model()
    
    # Perform object detection
    detected_img = detect_objects(image, model)
    
    # Convert BGR to RGB for displaying with Streamlit
    detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
    
    # Display detected image
    st.image(detected_img, caption="Detected Image", use_column_width=True)
