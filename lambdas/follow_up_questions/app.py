import json, boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}

@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)

@app.route("/", methods=["POST"])
def handler():
    body = request.get_json(force=True)
    topic = body.get("topic", "")
    level = body.get("level", "Beginner")
    history = body.get("history", [])
    message = body.get("message", "")

    system_prompt = (
        f"I am learning about the following topic: {topic}\n"
        f"My current knowledge level is: {level}\n\n"
        f"I have just read an explanation of this concept and seen some real-world examples. "
        f"I may have follow-up questions, points of confusion, or want to explore related ideas. "
        f"Please be ready to answer clearly and at the appropriate level of depth for my experience."
    )

    messages = list(history)
    messages.append({"role": "user", "content": message})

    def generate():
        resp = bedrock.invoke_model_with_response_stream(
            modelId=MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "system": system_prompt,
                "messages": messages,
            }),
        )
        for event in resp["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk.get("type") == "content_block_delta":
                yield chunk["delta"].get("text", "")

    return Response(stream_with_context(generate()), content_type="text/plain; charset=utf-8", headers=CORS_HEADERS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
