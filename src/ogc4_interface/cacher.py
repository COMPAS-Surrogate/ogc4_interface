import os
from urllib.parse import urlparse

import requests
from tqdm import tqdm

__all__ = ["Cacher"]

BASE_URL = "https://github.com/gwastro/4-ogc/raw/master"
HERE = os.path.dirname(__file__)


class Cacher:
    cache_dir = os.path.join(HERE, "cache_data")
    os.makedirs(cache_dir, exist_ok=True)

    def __init__(self, url: str):
        self.url = url
        if not self.exists:
            self._download()

    @property
    def fpath(self) -> str:
        if not hasattr(self, "_fpath"):
            fname = urlparse(self.url).path.split("/")[-1]
            self.__fpath = os.path.join(self.cache_dir, fname)
        return self.__fpath

    @property
    def exists(self) -> bool:
        return os.path.isfile(self.fpath)

    def _download(self):
        with requests.get(self.url, stream=True) as response:
            response.raise_for_status()
            # Get the total file size in bytes
            total_size = int(response.headers.get("content-length", 0))

            # Download the file with a progress bar

            with open(self.fpath, "wb") as file:
                for chunk in tqdm(
                    response.iter_content(chunk_size=8192),
                    total=total_size // 8192,
                    unit="KB",
                    unit_scale=True,
                    desc=f"Downloading {self.url}",
                ):
                    file.write(chunk)

    @classmethod
    def get(cls, url: str) -> str:
        return cls(url).fpath
