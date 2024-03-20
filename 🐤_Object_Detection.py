import streamlit as st
from PIL import Image

from model import object_detection, show_results

st.title("Vamos a calar el proyecto")

photo = st.camera_input("Pajarito pajarito 🐤")


if photo:
    image = Image.open(photo)
    results, image, model = object_detection(image=image)
    fig = show_results(results, image, model)
    st.pyplot(fig)

