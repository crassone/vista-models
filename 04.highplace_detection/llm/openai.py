import base64
from io import BytesIO
from openai import OpenAI
import os
from PIL import Image, ImageDraw
import requests
from typing import List

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

GPT4_MESSAGE = f"""
There is a person in the area framed in red in the image.
Is the person on the roof of the building or at least two stories high?
"""

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f'{"/".join(current_dir.split("/")[:-2])}'

client = OpenAI(api_key=OPENAI_API_KEY)

def scale_to_width(img: Image.Image, width: int = 1280) -> Image.Image:
    height = round(img.height * width / img.width)
    return img.resize((width, height))

def generate_text_by_gpt4(row) -> str:
    relative_img_path = row['relative_img_path']
    image_path = f'{root_dir}/data/images/{relative_img_path}'
    image = Image.open(image_path)
    bbox = list(row[['left', 'top', 'right', 'bottom']].values)
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline='red', width=4)
    image = scale_to_width(image)

    buffer = BytesIO()
    image.save(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": GPT4_MESSAGE
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    try:
        message = response.json()['choices'][0]['text']
    except:
        message = 'No response from GPT-4'
    return message

def text2emb_by_gpt(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    return client.embeddings.create(input=[text], model=model).data[0].embedding
