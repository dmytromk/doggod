from pathlib import Path
from http import HTTPStatus

import requests

from src.common.config import RESOURCES_DIR


def download_image(url: str, filepath: str | bytes):
    response = requests.get(url)
    if response.status_code == HTTPStatus.OK:
        output_file = Path(f"{RESOURCES_DIR}/{filepath}")
        output_file.parent.mkdir(exist_ok=True, parents=True)
        output_file.write_bytes(response.content)
        print("Image downloaded successfully as ", filepath)
    else:
        print("Failed to download image: ", url)
