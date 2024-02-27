from http import HTTPStatus
from pathlib import Path
from typing import Any

import requests


def get(url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == HTTPStatus.OK:
        return response.json()
    return response.text


def download_image(url: str, filepath: str | bytes):
    response = requests.get(url)
    if response.status_code == HTTPStatus.OK:
        output_file = Path(filepath)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        output_file.write_bytes(response.content)
        print("Image downloaded successfully as ", filepath)
    else:
        print("Failed to download image: ", url)
