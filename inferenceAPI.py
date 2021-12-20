import requests
from PIL import Image
from io import BytesIO
import argparse

from config import URL


def getPrediction(FLAGS):
    img = Image.open(FLAGS.image).convert("RGB")
    byte_io = BytesIO()
    img.save(byte_io, "png")
    byte_io.seek(0)
    response = requests.post(
        URL,
        files={"my_file": ("1.png", byte_io, "image/png")},
        data={"model": FLAGS.model},
    )
    print(response.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="Pesos del modelo fecha/model.bin", required=True
    )

    parser.add_argument(
        "--image", type=str, help="Imagen de prueba .png", required=True
    )
    FLAGS, _ = parser.parse_known_args()
    getPrediction(FLAGS)
