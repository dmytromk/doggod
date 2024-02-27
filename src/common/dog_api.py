import glob
import imghdr
import os
import uuid
from pathlib import Path

from src.common.config import RESOURCES_DIR
from src.common.web_utils import get, download_image


def get_all_breeds_url() -> dict[str, list[str]]:
    base_url = "https://dog.ceo/api/breeds/list/all"
    breeds_list: dict[str, list[str | None]] = get(base_url)["message"]
    result = {}

    for breed in breeds_list.keys():
        result[breed] = get(f"https://dog.ceo/api/breed/{breed}/images")["message"]

    return result


def download_all_images() -> None:
    # There are nearly 18000 images here, can take it a while
    breed_urls = get_all_breeds_url()
    image_folder = RESOURCES_DIR + "/dog-api"

    for breed, image_urls in breed_urls.items():
        # Use 60% of the dog-api for training, 20% for the validation and 20% for testing
        images_amount = len(image_urls)
        training_proportion = int(images_amount * 0.6)
        validation_proportion = int(images_amount * 0.8)

        for url in image_urls[:training_proportion]:
            download_image(url, f"{image_folder}/train/{breed}/{uuid.uuid4().hex}.jpg")

        for url in image_urls[training_proportion:validation_proportion]:
            download_image(url, f"{image_folder}/validation/{breed}/{uuid.uuid4().hex}.jpg")

        for url in image_urls[validation_proportion:]:
            download_image(url, f"{image_folder}/test/{breed}/{uuid.uuid4().hex}.jpg")


def remove_images_unsupported_format() -> None:
    image_folder = RESOURCES_DIR + "/dog-api"
    image_extensions = [".jpg"]  # all present images file extensions
    img_type_accepted_by_tensorflow = ["jpeg"]

    for filepath in Path(image_folder).rglob("*"):
        if filepath.suffix.lower() in image_extensions:
            img_type = imghdr.what(filepath)
            if img_type is None:
                print(f"{filepath} is not an image")
                os.remove(filepath)
            elif img_type not in img_type_accepted_by_tensorflow:
                print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
                os.remove(filepath)
