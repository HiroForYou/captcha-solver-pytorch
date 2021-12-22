import requests
import json
from PIL import Image
import json
import base64
import io
import pickle
import torch

image = Image.open("2cg58.png").convert("RGB")
buffered = io.BytesIO()
image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue())

payload = {"image": img_str.decode()}

payload = json.dumps(payload)

API_ENDPOINT = "https://ri1o1mdiqd.execute-api.us-east-2.amazonaws.com/deploy/predict"

# sending post request and saving response as response object
r = requests.post(url=API_ENDPOINT, data=payload)

predict = json.loads(r.text)
encoder_model = pickle.load(open("encoder.pkl", "rb"))


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    print(preds.shape)
    preds = preds.view(1, 1, 38)
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        cap_preds.append(remove_duplicates(tp))
    return cap_preds


print(
    f"Imprimiendo predicción decodificada: {decode_predictions(torch.FloatTensor(predict), encoder_model)[0]}"
)