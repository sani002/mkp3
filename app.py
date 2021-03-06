from pyexpat import model
import streamlit as st 
st.image('https://github.com/sani002/mkpapp/blob/main/Header.png?raw=true')
st.title("Monkey Pox Detector")

import keras
from PIL import Image, ImageOps
import numpy as np

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                            content:'This app is in its early stage. As a prototype, this app has not been verified by a dermatologist yet. Thank you.'; 
                            visibility: visible;
                            display: block;
                            position: relative;
                            #background-color: red;
                            padding: 5px;
                            top: 2px;
                        }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def teachable_machine_classification(img, weights_file):
    model = keras.models.load_model(weights_file)
    data = np.ndarray(shape=(1, 256, 256, 3), dtype=np.float32)
    image = img
    size = (256, 256)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 255.0)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction 
number = st.radio('Pick one', ['Upload from gallery', 'Capture by camera'])
if number=='Capture by camera':
    uploaded_file = st.camera_input("Take a picture")    
else:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png","jpeg","bmp"])
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded image.', use_column_width=True)
        st.write("")
        st.write("Result:")
        label = teachable_machine_classification(image, 'model.h5')
        if label < 0.5:
            st.write("It might be Monkeypox. You should visit a dermatologist immediately! Thank you.")
            st.write("")
            st.write("Accuracy:", (1-label)*100)
        else:
            st.write("It's most probably not monkeypox, but visiting a dermatologist always helps. Thank you.")
            st.write("")
            st.write("Accuracy:", (label)*100)
