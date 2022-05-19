import streamlit as st
import pandas as pd
import numpy as np

st.text('Welcome')
st.text('Please upload your Amazon data')
#st.sidebar.subheader("yan kol")

try:
    uploaded_file = st.file_uploader(label = 'Upload Your File', type = ['csv', 'xlsx'])
except:
    print('yes')
