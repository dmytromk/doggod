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
        for url in image_urls:
            if breed == "tervuren" or breed == "vizsla" or breed == "waterdog" or breed == "weimaraner" or breed == "whippet" or breed == "wolfhound":
                download_image(url, f"{image_folder}/{breed}/{uuid.uuid4().hex}.jpg")
