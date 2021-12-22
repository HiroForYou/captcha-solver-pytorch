import boto3
from PIL import Image
import json
import base64
import io

image = Image.open("./2cg58.png").convert("RGB")
buffered = io.BytesIO()
image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue())

payload = {"image": img_str.decode()}

payload = json.dumps(payload)

endpoint = "pytorch-inference-2021-10-15-03-29-08-633"

runtime = boto3.client("sagemaker-runtime", region_name="us-east-2")

response = runtime.invoke_endpoint(
    EndpointName=endpoint, ContentType="application/json", Body=payload
)

result = json.loads(response["Body"].read().decode())
