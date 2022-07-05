from pyexpat import model
import streamlit as st 
st.title("Monkey Pox Detector")
st.header("mHealth Lab, BME, BUET")

import keras
from PIL import Image, ImageOps
import numpy as np

def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 256, 256, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (256, 256)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255.0)
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction = model.predict(data)
    return prediction 

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png","jpeg","bmp"])
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = teachable_machine_classification(image, 'model.h5')
        if label < 0.5:
            st.write("It might be Monkeypox. You should visit a specialist immediately. Thank you.")
            st.write("Probability")
            st.write(1-label)
        else:
            st.write("It's most probably not monkeypox, but still you should visit a skin specialist. Thank you.")
            st.write("Probability")
            st.write(1-label)