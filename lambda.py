import boto3
import json
import os

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
runtime = boto3.client("sagemaker-runtime", region_name="us-east-2")


def lambda_handler(event, context):
    print("Evento recibido: " + json.dumps(event, indent=2))
    payload = {"image": str.encode(event["image"]).decode()}

    payload = json.dumps(payload)

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME, ContentType="application/json", Body=payload
    )

    result = json.loads(response["Body"].read().decode())
    return result
