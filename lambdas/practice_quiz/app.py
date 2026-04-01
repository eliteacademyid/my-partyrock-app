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
    file_data = body.get("file_data")
    file_mime = body.get("file_mime")

    prompt = (
        f"You are a quiz creator helping learners test their understanding of: {topic}\n\n"
        f"Create a practice quiz with 5-7 questions that test comprehension of this concept. Include a mix of:\n"
        f"- Multiple choice questions\n"
        f"- True/False questions\n"
        f"- Short answer prompts\n\n"
        f"Match the difficulty and terminology to the {level} level "
        f"(default to Beginner). After each question, provide the correct answer with a brief explanation "
        f"of why it's correct. Make it educational and encouraging."
    )

    content = []
    if file_data and file_mime:
        if file_mime.startswith("image/"):
            content.append({"type": "image", "source": {"type": "base64", "media_type": file_mime, "data": file_data}})
        else:
            content.append({"type": "document", "source": {"type": "base64", "media_type": file_mime, "data": file_data}})
    content.append({"type": "text", "text": prompt})

    messages = [{"role": "user", "content": content}]

    def generate():
        resp = bedrock.invoke_model_with_response_stream(
            modelId=MODEL_ID,
            body=json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 3000, "messages": messages}),
        )
        for event in resp["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk.get("type") == "content_block_delta":
                yield chunk["delta"].get("text", "")

    return Response(stream_with_context(generate()), content_type="text/plain; charset=utf-8", headers=CORS_HEADERS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
