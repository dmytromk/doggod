import os
import warnings

import keras
import numpy as np
import streamlit as st
from PIL import Image

from common.config import IMG_SIZE, RESOURCES_DIR

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Doggod",
    page_icon=":dog:",
    initial_sidebar_state='collapsed'
)

with st.sidebar:
    st.title("Doggod")
    # st.subheader(
    #     "Accurate detection of dog's breed by its image using.")

st.write("""
         # Detection of dog's breed by its image
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

    scratch_model = keras.models.load_model(f"{RESOURCES_DIR}/models/DoggodScratch.keras")
    mobilenetv2_model = keras.models.load_model(f"{RESOURCES_DIR}/models/DoggodMobileNetV2.keras")

    scratch_result = classes[np.argmax(scratch_model.predict(image_preprocessed))]
    mobilenetv2_result = classes[np.argmax(mobilenetv2_model.predict(image_preprocessed))]

    st.markdown("Scratch Model")
    st.info(scratch_result)

    st.markdown("MobileNetV2 Model")
    st.info(mobilenetv2_result)
