from mmdet.apis import inference_detector, init_detector
from mmdet.utils import sync_random_seed
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline
import numpy as np
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Union
import yaml
from utils.transform_pose2bbox import make_head_bbox, format_person_bbox
from utils.utils import get_device, set_seed

device = get_device()
seed = 42
set_seed(seed)
sync_random_seed(seed, device)

current_dir = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open(f'{current_dir}/config.yaml', 'r'), Loader=yaml.SafeLoader)

pattern_date = '2023-12-24'
root_dir = f'{"/".join(current_dir.split("/")[:-1])}'
format_anno_dir = f'{root_dir}/data/outputs/01.format_and_cv'
pose_output_dir = f'{root_dir}/data/outputs/02.pose_estimation'

def rtmpose_inference(data_info_df: pd.DataFrame) -> Dict[str, List[Dict[str, List[float]]]]:
    # build detector
    detector = init_detector(
        f'{pose_output_dir}/models/{config["MMDET_CONFIG_FILE"]}',
        f'{pose_output_dir}/models/{config["MMDET_CHECKPOINT_FILE"]}',
        device=device
    )
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        f'{pose_output_dir}/models/{config["MMPOSE_CONFIG_FILE"]}',
        f'{pose_output_dir}/models/{config["MMPOSE_CHECKPOINT_FILE"]}',
        device=device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
    )

    pose_result_dict = {}
    for unique_key, relative_img_path in tqdm(data_info_df[['unique_key', 'relative_img_path']].values):
        img = f'{root_dir}/data/images/{relative_img_path}'

        # predict bbox
        det_result= inference_detector(detector, img)
        pred_instance = det_result.pred_instances.cpu().numpy()

        bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > config['BBOX_THR'])]
        bboxes = bboxes[nms(bboxes, config['NMS_THR']), :4]

        # predict keypoints
        pose_results = inference_topdown(pose_estimator, img, bboxes)
        data_samples = merge_data_samples(pose_results)

        pred_instances = data_samples.get('pred_instances', None)

        pose_result_dict[unique_key] = split_instances(pred_instances)

    return pose_result_dict

def make_bbox(data_info_df: pd.DataFrame, pose_result_dict: Dict[str, List[Dict[str, List[float]]]]) -> List[List[Union[float, str]]]:
    pred_dict = {}
    for unique_key, pose_pred in pose_result_dict.items():
        relative_img_path = data_info_df[data_info_df['unique_key'] == unique_key].iloc[0, 1]
        image = Image.open(f'{root_dir}/data/images/{relative_img_path}')
        width, height = image.size
        pred_per_image = []

        for one_pose in pose_pred:
            keypoints = one_pose['keypoints']
            person_bbox = one_pose['bbox'][0]
            if all([cordinate == edge for cordinate, edge in zip(person_bbox, [0, 0, width, height])]):
                continue

            head_bbox = make_head_bbox(keypoints, width, height)
            formatted_person_bbox = format_person_bbox(person_bbox, width, height)

            if any([c == None for c in head_bbox]): continue
            if any([c == None for c in formatted_person_bbox]): continue

            pred_per_image.append({
                "person": formatted_person_bbox,
                "head": head_bbox,
            })
        pred_dict[unique_key] = pred_per_image

    pred_list = []
    for unique_key, value in pred_dict.items():
        for pred in value:
            for label, bbox in pred.items():
                pred_list.append([unique_key, label] + bbox)

    return pred_list

def main(data_info_df: pd.DataFrame):
    """_summary_

    Args:
        data_info_df (pd.DataFrame): csv file with columns [
            'unique_key', 'relative_img_path', label, left, top, right, bottom, validation]
    """

    pose_result_dict = rtmpose_inference(data_info_df)
    pred_list = make_bbox(data_info_df, pose_result_dict)
    person_head_detection_result_df = pd.DataFrame(pred_list, columns=['unique_key', 'pred', 'left', 'top', 'right', 'bottom'])
    person_head_detection_result_df = pd.merge(
        person_head_detection_result_df,
        data_info_df[['unique_key', 'relative_img_path', 'validation']].drop_duplicates(),
        on='unique_key'
    )

    os.makedirs(pose_output_dir, exist_ok=True)
    person_head_detection_result_df.to_csv(f'{pose_output_dir}/pred-{pattern_date}.csv', index=False)


if __name__ == '__main__':
    data_info_df = pd.read_csv(f'{format_anno_dir}/annotation-{pattern_date}.csv')
    main(data_info_df)
