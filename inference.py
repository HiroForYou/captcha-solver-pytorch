import glob
import os.path

import torch
import argparse
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
from sklearn import preprocessing

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
    filename_encoder = "encoder.pkl"
    image_files = glob.glob(os.path.join(config.DATA_DIR, "*.jpg"))
    targets_orig = [x.split("/")[-1][:-4] for x in image_files]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]
    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)
    pickle.dump(lbl_enc, open(filename_encoder, 'wb'))
    encoder_model = pickle.load(open(filename_encoder, "rb"))

     # CARGA E INFERENCIA DEL MODELO
    tensor_image = input_data(FLAGS.image)
    NRO_CHARS = len(lbl_enc.classes_)
    model = CaptchaModel(NRO_CHARS)
    model.load_state_dict(torch.load(FLAGS.model, map_location='cpu'))
    #model = torch.load(FLAGS.model)
    model.eval().to("cpu")
    encoded_prediction, _ = model(tensor_image)

    
    current_preds = decode_predictions(encoded_prediction, encoder_model)
    print(current_preds[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="./model.bin", help="Pesos del modelo"
    )
    parser.add_argument(
        "--image", type=str, default="./289682249H.jpg", help="Imagen de prueba"
    )
    FLAGS, unparsed = parser.parse_known_args()
    run_predict(FLAGS)
