import numpy as np
from typing import List

# 骨格推定→頭部検出
def make_head_bbox(
        person_keypoints: List[float], img_width: int, img_height: int, head_bbox_ratio: float = 0.33) -> List[float]:
    person_keypoints = np.array(person_keypoints)
    face_center_keypoints = (
        person_keypoints[0] + person_keypoints[1] + person_keypoints[2] + person_keypoints[3] + person_keypoints[4]) / 5
    hip_keypoints = (person_keypoints[11] + person_keypoints[12]) / 2
    upper_body_length = np.linalg.norm(face_center_keypoints - hip_keypoints)
    head_bbox_half_length = upper_body_length * head_bbox_ratio

    x_left = face_center_keypoints[0] - head_bbox_half_length
    y_top = face_center_keypoints[1] - head_bbox_half_length
    x_right = face_center_keypoints[0] + head_bbox_half_length
    y_bottom = face_center_keypoints[1] + head_bbox_half_length

    if x_left < 0:
        x_left = 0
    if y_top < 0:
        y_top = 0
    if x_right > img_width:
        x_right = img_width
    if y_bottom > img_height:
        y_bottom = img_height

    if x_right < 0 or y_bottom < 0 or x_left > img_width or y_top > img_height:
        return [None, None, None, None]
    return [x_left, y_top, x_right, y_bottom]

# 人検出
def format_person_bbox(person_bbox: List[float], img_width: int, img_height: int) -> List[float]:
    x_left, y_top, x_right, y_bottom = person_bbox

    if x_left < 0:
        x_left = 0
    if y_top < 0:
        y_top = 0
    if x_right > img_width:
        x_right = img_width
    if y_bottom > img_height:
        y_bottom = img_height

    if x_right < 0 or y_bottom < 0 or x_left > img_width or y_top > img_height:
        return [None, None, None, None]
    return [x_left, y_top, x_right, y_bottom]
