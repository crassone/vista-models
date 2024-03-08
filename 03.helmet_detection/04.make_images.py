import copy
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from utils.class_category import class_category

font_size = 1.2 * ImageFont.load_default().size
font = ImageFont.truetype("arial.ttf", size=font_size)

pattern_date = '2023-12-24'
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f'{"/".join(current_dir.split("/")[:-1])}'
eval_result_dir = f'{root_dir}/data/outputs/03.helmet_detection/03.eval_coatnet_cls7_use_anno'
result_dir = f'{root_dir}/data/outputs/03.helmet_detection/04.make_images'

def make_images(valid_df: pd.DataFrame):
    for unique_key in tqdm(valid_df['unique_key'].unique()):
        valid_unique_df = valid_df[valid_df['unique_key'] == unique_key].reset_index(drop=True)
        relative_img_path = valid_unique_df.loc[0, "relative_img_path"]
        image_path = f'{root_dir}/data/images/{relative_img_path}'
        image = Image.open(image_path)
        width, height = image.size

        for _, img_info in valid_unique_df.iterrows():
            # Drawオブジェクトの作成
            _image = copy.deepcopy(image)
            draw = ImageDraw.Draw(_image)
            bbox_head = img_info[['left', 'top', 'right', 'bottom']].tolist()

            if bbox_head[0] > bbox_head[2]:
                tmp_bbox = bbox_head[2]
                bbox_head[2] = bbox_head[0]
                bbox_head[0] = tmp_bbox
            if bbox_head[1] > bbox_head[3]:
                tmp_bbox = bbox_head[3]
                bbox_head[3] = bbox_head[1]
                bbox_head[1] = tmp_bbox

            if bbox_head[0] < 0:
                bbox_head[0] = 0
            if bbox_head[1] < 0:
                bbox_head[1] = 0
            if bbox_head[2] > width:
                bbox_head[2] = width
            if bbox_head[3] > height:
                bbox_head[3] = height

            # バウンディングボックスの描画
            draw.rectangle(bbox_head, outline='red', width=2)

            # テキストの描画
            label_head = img_info['label']
            prob_head = img_info['pred_head_class']
            text_head = f'{label_head} x {prob_head}'
            text_height_head = font.size
            text_width_head = font.getlength(text_head)

            draw.rectangle((bbox_head[0], bbox_head[1] - text_height_head, bbox_head[0] + text_width_head, bbox_head[1]), fill='red')
            draw.text((bbox_head[0], bbox_head[1] - text_height_head), text_head, fill='white', font=font)

            for idx, class_names in reversed(list(enumerate(class_category))):
                if label_head == class_names[0]:
                    main_label_num = idx
                    break
            for idx, class_names in reversed(list(enumerate(class_category))):
                if prob_head == class_names[0]:
                    main_pred_num = idx
                    break

            output_img_path = f'{result_dir}/{main_label_num}{main_pred_num}/{relative_img_path}'
            output_img_dir = '/'.join(output_img_path.split("/")[:-1])
            os.makedirs(output_img_dir, exist_ok=True)
            _image.save(output_img_path)

if __name__ == '__main__':
    pred_head_class_df = pd.read_csv(f'{eval_result_dir}/pred_head_class-{pattern_date}.csv')
    make_images(pred_head_class_df)
