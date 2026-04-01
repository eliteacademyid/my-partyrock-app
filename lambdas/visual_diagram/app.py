import json, boto3, base64, random

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "amazon.nova-canvas-v1:0"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}

def handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    body = json.loads(event.get("body") or "{}")
    topic = body.get("topic", "")
    level = body.get("level", "Beginner")

    image_prompt = (
        f"Create a clear, educational diagram or visual representation of the concept: {topic}. "
        f"The visual should be appropriate for someone at the {level} level. "
        f"Include labeled components, key relationships, and visual metaphors that make the concept easier to understand. "
        f"Use a clean, infographic style that would work well in an educational setting."
    )

    payload = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": image_prompt},
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "width": 1280,
            "height": 720,
            "cfgScale": 5.0,
            "seed": random.randint(0, 2147483647),
        },
    }

    resp = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(resp["body"].read())
    image_b64 = result["images"][0]

    return {
        "statusCode": 200,
        "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
        "body": json.dumps({"image": image_b64}),
    }
