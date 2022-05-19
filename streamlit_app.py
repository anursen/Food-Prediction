import streamlit as st
import pandas as pd
import numpy as np
# TF2 version
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
import cv2
from skimage import io


st.text('Welcome')
st.text('Please upload your your picture.')
#st.sidebar.subheader("yan kol")

try:
    uploaded_file = st.file_uploader(label = 'Upload Your File', type = ['png', 'jpg'])


    cake_url = uploaded_file
labelmap_url = "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_food_V1_labelmap.csv"
input_shape = (224, 224)

m = hub.KerasLayer('https://tfhub.dev/google/aiy/vision/classifier/food_V1/1')
image = np.asarray(io.imread(cake_url), dtype="float")
image = cv2.resize(image, dsize=input_shape, interpolation=cv2.INTER_CUBIC)
# Scale values to [0, 1].
image = image / image.max()
# The model expects an input of (?, 224, 224, 3).
images = np.expand_dims(image, 0)
# This assumes you're using TF2.
output = m(images)
predicted_index = output.numpy().argmax()
classes = list(pd.read_csv(labelmap_url)["name"])

st.text("Prediction: ", classes[predicted_index])
    
    
    
    
    
    
    
except:
    print('yes')
