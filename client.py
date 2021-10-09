import requests
from PIL import Image
from io import BytesIO
import argparse

URL = "http://127.0.0.1:8000/getPrediction"


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
        "--model", type=str, default="model.bin", help="Pesos del modelo"
    )
    parser.add_argument(
        "--image", type=str, default="2cg58.png", help="Imagen de prueba"
    )
    FLAGS, unparsed = parser.parse_known_args()
    getPrediction(FLAGS)
