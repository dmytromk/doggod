import os
import random
import warnings

import keras
import numpy as np
import streamlit as st
from PIL import Image

from common.config import IMG_SIZE, RESOURCES_DIR

warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Doggod",
    page_icon=":dog:",
    initial_sidebar_state='auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style,
            unsafe_allow_html=True)  # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

with st.sidebar:
    st.title("Doggod")
    # st.subheader(
    #     "Accurate detection of dog's breed by its image using.")

st.write("""
         # Accurate detection of dog's breed by its image using
         """)

file = st.file_uploader("", type=["jpg", "png"])
classes = os.listdir(f"{RESOURCES_DIR}/dog-api/train")


def import_image(image_data):
    size = (IMG_SIZE, IMG_SIZE)
    image = image_data.resize(size)
    image = image.convert("RGB")
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = image[:, :, :3]
    return np.stack((image,), axis=0)


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    image_preprocessed = import_image(image)

    scratch_model = keras.models.load_model(f"{RESOURCES_DIR}/models/DoggodScratch")
    mobilenetv2_model = keras.models.load_model(f"{RESOURCES_DIR}/models/DoggodMobileNetV2")

    scratch_result = classes[np.argmax(scratch_model.predict(image_preprocessed))]
    mobilenetv2_result = classes[np.argmax(mobilenetv2_model.predict(image_preprocessed))]

    # x = random.randint(98, 99) + random.randint(0, 99) * 0.01
    # st.sidebar.error("Accuracy : " + str(x) + " %")

    st.markdown("Scratch Model")
    st.info(scratch_result)

    st.markdown("MobileNetV2 Model")
    st.info(mobilenetv2_result)
