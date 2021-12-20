import argparse
from PIL import Image
import pickle
import torch
from torchvision import transforms

from model import CaptchaModel
import config
from train import decode_predictions

IMAGE_WIDTH = config.IMAGE_WIDTH
IMAGE_HEIGHT = config.IMAGE_HEIGHT


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def input_data(image_dir):
    image = Image.open(image_dir).convert("RGB")
    image_resize = image.resize(
        size=(IMAGE_WIDTH, IMAGE_HEIGHT), resample=Image.BILINEAR
    )
    image.close()
    tensor_image = transform(image_resize).unsqueeze_(0)
    return tensor_image


def run_predict(FLAGS):
    # DECODIFICADOR DEL MENSAJE
    encoder_model = pickle.load(open(f"weights/encoder.pkl", "rb"))

    # CARGA E INFERENCIA DEL MODELO
    tensor_image = input_data(FLAGS.image)
    NRO_CHARS = len(encoder_model.classes_)
    model = CaptchaModel(NRO_CHARS)
    model.load_state_dict(torch.load(f"weights/{FLAGS.model}", map_location="cpu"))
    # model = torch.load(FLAGS.model)
    model.eval().to("cpu")
    encoded_prediction, _ = model(tensor_image)
    current_preds = decode_predictions(encoded_prediction, encoder_model)
    print(current_preds[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="Pesos del modelo fecha/model.bin", required=True
    )

    parser.add_argument(
        "--encoder", type=str, help="Pesos del encoder fecha/model.bin", required=True
    )

    parser.add_argument(
        "--image", type=str, default="./289682249H.jpg", help="Imagen de prueba"
    )
    FLAGS, unparsed = parser.parse_known_args()
    run_predict(FLAGS)
