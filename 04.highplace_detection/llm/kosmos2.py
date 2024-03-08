import math
import os
from PIL import Image
from typing import List
from .utils import get_device, set_seed

device = get_device()
set_seed()

MESSAFE_FORMAT = '<grounding> Question: Where is<phrase> the person</phrase><object><patch_index_%(coordinate1)><patch_index_%(coordinate2)></object> on? Answer:'

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f'{"/".join(current_dir.split("/")[:-2])}'

def transform_bbox2patch_index(image: Image.Image, bbox: List[float]):
    patch_bins = 32
    img_width, img_height = image.size
    coordinate1_x = math.floor(bbox[0] / img_width * patch_bins)
    coordinate1_y = math.floor(bbox[1] / img_height * patch_bins)
    coordinate1_index = coordinate1_y * patch_bins + coordinate1_x

    coordinate2_x = math.ceil(bbox[2] / img_width * patch_bins)
    coordinate2_y = math.ceil(bbox[3] / img_height * patch_bins)
    coordinate2_index = coordinate2_y * patch_bins + coordinate2_x

    return coordinate1_index, coordinate2_index

def make_prompt(image: Image.Image, bbox: List[float]):
    coordinate1, coordinate2 = transform_bbox2patch_index(image, bbox)
    prompt = (
        MESSAFE_FORMAT
        .replace('%(coordinate1)', f'{coordinate1:04d}')
        .replace('%(coordinate2)', f'{coordinate2:04d}')
    )
    return prompt

def generate_text_by_kosmos2(row, model, processor):
    relative_img_path = row['relative_img_path']
    image_path = f'{root_dir}/data/images/{relative_img_path}'
    image = Image.open(image_path)
    bbox = row[['left', 'top', 'right', 'bottom']].tolist()
    prompt = make_prompt(image, bbox)

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"].to(device),
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"].to(device),
        use_cache=True,
        max_new_tokens=128,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    processed_text, _ = processor.post_process_generation(generated_text)
    answer_text = processed_text.split('Answer: ')[-1]
    return answer_text
