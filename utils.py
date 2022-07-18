"""
Module that contains the functions to download the KMINST dataset.
Author: David Cardozo <david.cardozo@me.com>
"""

from urllib.request import urlopen
from pathlib import Path

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


KMNIST_IMAGES = "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz"
KMNIST_LABELS = "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz"

def download_file(url: str, filepath:Path):
    """
    Wrapper around urllib to download blobs of objects
    """
    response = urlopen(url)
    CHUNK = 16 * 1024
    with open(filepath, "wb") as fd:
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            fd.write(chunk)

def download_kminst(dir_path: Path):
    """
    Download KMINST dataset in an specified directory
    """
    try:
        dir_path.mkdir(parents=True)
    except FileExistsError as exc:
        if dir_path.is_file():
            raise FileExistsError(f"{dir_path} is a file") from exc

    # Download train images
    download_file(KMNIST_IMAGES, dir_path / "k49-train-imgs.npz")
    logging.info("Downloaded k49-train-imgs.npz")
    # Download train labels
    download_file(KMNIST_LABELS, dir_path / "k49-train-labels.npz")
    logging.info("Downloaded k49-train-labels.npz")

if __name__ == "__main__":
    download_kminst(Path("data_dir"))