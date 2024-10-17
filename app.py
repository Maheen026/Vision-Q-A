import streamlit as st 
import os
from PIL import Image
import requests
from io import BytesIO
from transformers import ViltForQuestionAnswering, ViltProcessor

st.set_page_config(layout="wide", page_title="VQA")
st.title("Visual Question Answering Tool")
st.write("Upload an image and enter query to get a response")


col1, col2 = st.columns(2)

#Image Upload
with col1:
    upload_file = st.file_uploader("Upload Image" , type=['jpg','jpeg','png'])
    st.image(upload_file, use_column_width = True)

with col2:
    pass