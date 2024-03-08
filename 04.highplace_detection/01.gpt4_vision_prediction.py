import copy
import os
import pandas as pd
from tqdm import tqdm
from llm.openai import generate_text_by_gpt4, text2emb_by_gpt
tqdm.pandas()

pattern_date = '2023-12-24'
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = f'{"/".join(current_dir.split("/")[:-1])}'
person_detection_dir = f'{root_dir}/data/outputs/04.highplace_detection/00.person_detection_add_label'
result_dir = f'{root_dir}/data/outputs/04.highplace_detection/01.gpt4_text_and_emb'

def main(pred_person_df: pd.DataFrame):
    valid_df = pred_person_df[
        (pred_person_df['unique_key'].str.contains('fixed-point-camera')) |
        (pred_person_df['unique_key'].str.contains('for-learning/2023-11-19-omaezaki-500')) |
        (pred_person_df['unique_key'].str.contains('for-learning/2023-11-23-mie-safetybelt')) |
        (pred_person_df['unique_key'].str.contains('evaluation'))
    ]
    valid_df['openai_message'] = copy.deepcopy(valid_df).progress_apply(generate_text_by_gpt4, axis=1)
    valid_df['embedding'] = copy.deepcopy(valid_df)['openai_message'].progress_apply(text2emb_by_gpt)

    embedding_size = 1536
    for i in range(embedding_size):
        valid_df[f'embedding_{i}'] = copy.deepcopy(valid_df)['embedding'].apply(lambda x: x[i])

    os.makedirs(result_dir, exist_ok=True)
    valid_df.to_csv(f'{result_dir}/person_eval_pred_message_and_emb-{pattern_date}.csv', index=False)


if __name__ == '__main__':
    pred_person_df = pd.read_csv(f'{person_detection_dir}/person_eval_pred_add_label-{pattern_date}.csv')
    main(pred_person_df)
