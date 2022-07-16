from __future__ import annotations

from pathlib import Path

import requests
from PIL import Image

__all__ = ["load_image_from_path_or_url"]


def load_image_from_path_or_url(path_or_url: str | Path) -> Image:
    if isinstance(path_or_url, str) and path_or_url.startswith("http"):
        im_ref = requests.get(path_or_url, stream=True).raw
    else:
        im_ref = path_or_url
    image = Image.open(im_path).convert("RGB")
    return image
