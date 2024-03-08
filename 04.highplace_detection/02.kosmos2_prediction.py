import copy
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from llm.kosmos2 import generate_text_by_kosmos2
from llm.openai import text2emb_by_gpt
from llm.utils import get_device
tqdm.pandas()

device = get_device()

pattern_date = '2023-12-24'
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f'{"/".join(current_dir.split("/")[:-1])}'
person_detection_dir = f'{root_dir}/data/outputs/04.highplace_detection/00.person_detection_add_label'
result_dir = f'{root_dir}/data/outputs/04.highplace_detection/02.kosmos2_text_and_emb'

def main(pred_person_df: pd.DataFrame):
    huggingface_model_name = 'microsoft/kosmos-2-patch14-224'
    processor = AutoProcessor.from_pretrained(huggingface_model_name)
    model = AutoModelForVision2Seq.from_pretrained(huggingface_model_name)
    model = model.to(device)

    def _generate_text_by_kosmos2(row):
        return generate_text_by_kosmos2(row, model, processor)

    valid_df = pred_person_df[
        (pred_person_df['unique_key'].str.contains('fixed-point-camera')) |
        (pred_person_df['unique_key'].str.contains('for-learning/2023-11-19-omaezaki-500')) |
        (pred_person_df['unique_key'].str.contains('for-learning/2023-11-23-mie-safetybelt')) |
        (pred_person_df['unique_key'].str.contains('evaluation'))
    ]
    valid_df['kosmos2_message'] = copy.deepcopy(valid_df).progress_apply(_generate_text_by_kosmos2, axis=1)
    valid_df['embedding'] = copy.deepcopy(valid_df)['kosmos2_message'].progress_apply(text2emb_by_gpt)

    embedding_size = 1536
    for i in range(embedding_size):
        valid_df[f'embedding_{i}'] = copy.deepcopy(valid_df)['embedding'].apply(lambda x: x[i])

    os.makedirs(result_dir, exist_ok=True)
    valid_df.to_csv(f'{result_dir}/person_eval_pred_message_and_emb-{pattern_date}.csv', index=False)


if __name__ == '__main__':
    pred_person_df = pd.read_csv(f'{person_detection_dir}/person_eval_pred_add_label-{pattern_date}.csv')
    main(pred_person_df)
