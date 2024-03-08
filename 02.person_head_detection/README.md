# RTMPoseによる人検出・頭部検出

## 事前準備
以下のコマンドでRTMDet・RTMPoseのモデルをダウンロード
```command
mim download mmdet --config rtmdet_m_8xb32-300e_coco --dest ../data/outputs/02.pose_estimation/models
mim download mmpose --config rtmpose-m_8xb256-420e_body8-256x192 --dest ../data/outputs/02.pose_estimation/models
```

## 推論実行
以下のコマンドで対象画像に対して、人検出・頭部検出を実行できます。
```command
python 01.rtmpose_inference.py
```
