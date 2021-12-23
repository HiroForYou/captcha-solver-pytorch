import base64
import io
import json
import logging
import os
from PIL import Image
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
class Net(nn.Module):
    def __init__(self, num_chars):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3, 6), padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 6), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear1 = nn.Linear(1152, 64)
        self.drop1 = nn.Dropout(0.2)
        self.gru = nn.GRU(
            64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True
        )
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images, targets=None):
        bs, _, _, _ = images.size()
        x = F.relu(self.conv1(images))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        x = F.relu(self.linear1(x))
        x = self.drop1(x)
        x, _ = self.gru(x)
        x = self.output(x)
        x = x.permute(1, 0, 2)

        if targets is not None:
            log_probs = F.log_softmax(x, 2)
            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            target_lengths = torch.full(
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(
                log_probs, targets, input_lengths, target_lengths
            )
            return x, loss

        return x, None


def model_fn(model_dir):
    # 19 cambia en funcion del vocabulario, fijar esto al momento del deploy
    model = Net(num_chars=19)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Cargando modelo.")
    with open(os.path.join(model_dir, "best.bin"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device))

    model.eval()
    logger.info("Modelo cargado!")
    return model


def input_fn(request_body, content_type="application/json"):
    logger.info("Deserializando imagen.")
    if content_type == "application/json":
        input_data = json.loads(request_body)
        image_data = input_data["image"]
        image_data = Image.open(io.BytesIO(base64.b64decode(image_data)))
        image_transform = transforms.Compose(
            [
                transforms.Resize(size=(75, 300)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        return image_transform(image_data)
    raise Exception(
        f"Request no soportado, ContentType en content_type: {content_type}"
    )


def predict_fn(input_data, model):
    logger.info("Generando predicción...")
    if torch.cuda.is_available():
        input_data = input_data.view(1, 3, 75, 300).cuda()
    else:
        input_data = input_data.view(1, 3, 75, 300)

    with torch.no_grad():
        model.eval()
        encoded_prediction, _ = model(input_data)
        return encoded_prediction


def output_fn(prediction_output, accept="application/json"):
    logger.info("Serializando la salida.")

    result = prediction_output.cpu().numpy()

    if accept == "application/json":
        return json.dumps(result.tolist()), accept
    raise Exception(f"Requested no soportado ContentType in Accept: {accept}")
