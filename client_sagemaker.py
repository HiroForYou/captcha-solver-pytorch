import base64
import boto3
import io
import json
from PIL import Image

from config import API_ENDPOINT, IMAGE_TEST

image = Image.open(IMAGE_TEST).convert("RGB")
buffered = io.BytesIO()
image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue())
payload = {"image": img_str.decode()}
payload = json.dumps(payload)
runtime = boto3.client("sagemaker-runtime", region_name="us-east-2")

response = runtime.invoke_endpoint(
    EndpointName=API_ENDPOINT, ContentType="application/json", Body=payload
)

result = json.loads(response["Body"].read().decode())
print(result)
