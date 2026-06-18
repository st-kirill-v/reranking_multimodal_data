from openai import OpenAI
import base64

client = OpenAI(api_key="sk-roGquc-nH0243N7t9DwzoA", base_url="http://10.32.15.88:4000/v1")

image_path = "data/datasets/docbench/81/extracted/pages/page_1.png"

with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="openai/qwen3-vl-30b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in one sentence."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }
    ],
    temperature=0,
)

print(response.choices[0].message.content)
