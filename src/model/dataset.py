import uuid

from src.common.config import RESOURCES_DIR
from src.common.web_utils import get, download_image


def get_all_breeds_url() -> dict[str, list[str]]:
    base_url = "https://dog.ceo/api/breeds/list/all"
    breeds_list: dict[str, list[str | None]] = get(base_url)["message"]
    result = {}

    for breed, sub_breeds in breeds_list.items():
        if not sub_breeds:
            result[breed] = get(f"https://dog.ceo/api/breed/{breed}/images")["message"]

        else:
            for sub_breed in sub_breeds:
                result[f"{breed}_{sub_breed}"] = get(f"https://dog.ceo/api/breed/{breed}/{sub_breed}/images")["message"]

    return result


def download_all_images() -> None:
    breed_urls = get_all_breeds_url()
    image_folder = RESOURCES_DIR + "/images"

    for breed, image_urls in breed_urls.items():
        # Use 80% of the images for training, other - for the validation
        images_amount = len(image_urls)
        training_proportion = 0.8
        validation_index = images_amount - int(images_amount * training_proportion)

        for url in image_urls[:validation_index]:
            download_image(url, f"{image_folder}/train/{breed}/{uuid.uuid4().hex}.jpg")

        for url in image_urls[validation_index:]:
            download_image(url, f"{image_folder}/validation/{breed}/{uuid.uuid4().hex}.jpg")
