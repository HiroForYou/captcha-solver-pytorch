# app.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle
from sklearn import preprocessing
import uvicorn
from io import BytesIO

from model import CaptchaModel
import config
from train import decode_predictions

IMAGE_WIDTH = config.IMAGE_WIDTH
IMAGE_HEIGHT = config.IMAGE_HEIGHT

app = FastAPI(
    title="Solver Captcha", description="Endpoint Solver Captcha", version="0.0.1"
)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def run_predict(model_file, encoder_file, image_file):
    encoder_model = pickle.load(open(f"weights/{encoder_file}", "rb"))
    NRO_CHARS = len(encoder_model.classes_)
    model = CaptchaModel(NRO_CHARS)
    model.load_state_dict(torch.load(f"weights/{model_file}", map_location="cpu"))
    model.eval().to("cpu")

    newStream = BytesIO(image_file.read())
    image = Image.open(newStream).convert("RGB")
    # image.show()
    image_resize = image.resize(
        size=(IMAGE_WIDTH, IMAGE_HEIGHT), resample=Image.BILINEAR
    )
    image.close()
    tensor_image = transform(image_resize).unsqueeze_(0)
    with torch.inference_mode():
        encoded_prediction, _ = model(tensor_image)
        current_preds = decode_predictions(encoded_prediction, encoder_model)
        return current_preds[0]


@app.get("/")
def home():
    return {"message": "Endpoint Solver Captcha"}


@app.post("/getPrediction")
def _file_upload(
    my_file: UploadFile = File(...), encoder: str = Form(...), model: str = Form(...)
):
    try:
        prediction = run_predict(model, encoder, my_file.file)
        return {"response": prediction}
    except:
        return {"error": "actualice el despliegue"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
