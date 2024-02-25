import uuid

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
    breed_urls = get_all_breeds_url()
    image_folder = RESOURCES_DIR + "/dog-api"

    for breed, image_urls in breed_urls.items():
        # Use 60% of the dog-api for training, other - for the validation
        images_amount = len(image_urls)
        training_proportion = 0.6
        validation_index = int(images_amount * training_proportion)

        for url in image_urls[:validation_index]:
            download_image(url, f"{image_folder}/train/{breed}/{uuid.uuid4().hex}.jpg")

        for url in image_urls[validation_index:]:
            download_image(url, f"{image_folder}/validation/{breed}/{uuid.uuid4().hex}.jpg")
