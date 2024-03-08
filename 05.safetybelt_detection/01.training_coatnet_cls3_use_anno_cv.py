import copy
import math
import numpy as np
import os
import pandas as pd
import timm
from timm.scheduler import CosineLRScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score
import yaml
from utils.class_category import class_category, no_alarm_class_category
from utils.dataset import ClassificationDataset
from utils.pytorchtools import EarlyStopping
from utils.transforms import transform_train, transform_val
from utils.utils import get_device, set_seed

device = get_device()
set_seed()

current_dir = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open(f'{current_dir}/config.yaml', 'r'), Loader=yaml.SafeLoader)

pattern_date = '2023-12-24'
root_dir = f'{"/".join(current_dir.split("/")[:-1])}'
safetybelt_detection_label_dir = f'{root_dir}/data/outputs/05.safetybelt_detection/00.safetybelt_detection_add_label'
result_dir = f'{root_dir}/data/outputs/05.safetybelt_detection/01.training_coatnet_cls3_use_anno'

def train(train_df, valid_df, num_cv):
    class_num = [train_df['label_safetybelt'].value_counts()[class_names].sum() for class_names in class_category]
    weight_class = [round(max(class_num) / i) for i in class_num]

    train_weight_list = []
    for weight, class_names in zip(weight_class, class_category):
        for _ in range(weight):
            train_weight_list.append(
                pd.concat([train_df[train_df['label_safetybelt'] == class_name] for class_name in class_names])
            )
    train_df_weight = pd.concat(train_weight_list)

    train_data = ClassificationDataset(train_df_weight, root_dir, transform_train(config['input_img_size']), training=True)
    valid_data = ClassificationDataset(valid_df, root_dir, transform_val(config['input_img_size']))
    train_loader = DataLoader(dataset=train_data, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=config['batch_size'], shuffle=False)

    # model
    model = timm.create_model(config['huggingface_model_name'], pretrained=True, num_classes=len(class_category))
    model.to(device)

    # early stopping
    os.makedirs(f'{result_dir}/{num_cv}', exist_ok=True)
    early_stopping = EarlyStopping(verbose=True, save_dir=f'{result_dir}/{num_cv}', device=device)

    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    if config['optimizer_name'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer_name'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        assert False, '[ERROR] optimizer_name is not correct'

    # scheduler
    warm_up_epochs = math.ceil(config['epochs'] * config['warmup_ratio'])
    scheduler = CosineLRScheduler(
        optimizer, t_initial=config['epochs'],
        warmup_t=warm_up_epochs, warmup_lr_init=config['lr_min'], lr_min=config['lr_min'], warmup_prefix=True
    )

    for epoch in range(config['epochs']):
        epoch_loss = 0
        for data, label in tqdm(train_loader):

            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss / len(train_loader)

        labels_list = []
        prob_list = []
        with torch.no_grad():
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)
                epoch_val_loss += val_loss / len(valid_loader)
                probs = F.softmax(val_output, dim=1)
                labels_list.append(label.detach().cpu().numpy())
                prob_list.append(probs.detach().cpu().numpy())

        labels_np = np.concatenate(labels_list)
        prob_np = np.concatenate(prob_list)

        # loss
        print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - val_loss : {epoch_val_loss:.4f} \n")

        # PR-AUC
        for class_num, class_names in enumerate(class_category):
            if any([no_alarm_class in class_names for no_alarm_class in no_alarm_class_category]):
                continue

            use_label = [i for i in labels_np if i in [j for j in range(len(no_alarm_class_category))] + [class_num]]
            use_prob = [j for i, j in zip(labels_np, prob_np) if i in [j for j in range(len(no_alarm_class_category))] + [class_num]]
            use_label = np.array([1 if i in [class_num] else 0 for i in use_label])
            use_prob = np.array([i[class_num] for i in use_prob])
            average_precision = average_precision_score(use_label, use_prob)
            precisions, recalls, thresholds = precision_recall_curve(use_label, use_prob)
            result_precision090 = [0, 0, 0] # precision, recall, threshold で保存
            result_precision095 = [0, 0, 0] # precision, recall, threshold で保存
            for p, r, t in zip(precisions, recalls, thresholds):
                if p >= 0.9:
                    result_precision090 = [p, r, t]
                    break
            for p, r, t in zip(precisions, recalls, thresholds):
                if p >= 0.95:
                    result_precision095 = [p, r, t]
                    break

            print(f'{class_names} - PR-AUC: {average_precision:.4f}')
            print(f'precision@{result_precision090[0]:.2f} - Recall: {result_precision090[1]:.4f} - Threshold: {result_precision090[2]:.4f}')
            print(f'precision@{result_precision095[0]:.2f} - Recall: {result_precision095[1]:.4f} - Threshold: {result_precision095[2]:.4f} \n')

        scheduler.step(epoch)

        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            break
    print(f'best epoch: {early_stopping.best_epoch}')

def train_cv(annotation_safetybelt_df: pd.DataFrame, pred_safetybelt_df: pd.DataFrame):
    annotation_safetybelt_df = copy.deepcopy(annotation_safetybelt_df[annotation_safetybelt_df['label'].notnull()])
    for num_cv in range(3):
        train_df = copy.deepcopy(pd.concat([
            annotation_safetybelt_df[
                (annotation_safetybelt_df['validation'] != num_cv) &
                (annotation_safetybelt_df['validation'] != 999)
            ],
            pred_safetybelt_df[
                (pred_safetybelt_df['validation'] != num_cv) &
                (pred_safetybelt_df['validation'] != 999) &
                (pred_safetybelt_df['label'] == "detection-miss") &
                (
                    (pred_safetybelt_df['unique_key'].str.contains('fixed-point-camera')) |
                    (pred_safetybelt_df['unique_key'].str.contains('for-learning/2023-11-19-omaezaki-500')) |
                    (pred_safetybelt_df['unique_key'].str.contains('for-learning/2023-11-23-mie-safetybelt'))
                )
            ]
        ]))
        valid_df = copy.deepcopy(pred_safetybelt_df[
            (pred_safetybelt_df['validation'] == num_cv) &
            (
                (pred_safetybelt_df['unique_key'].str.contains('fixed-point-camera')) |
                (pred_safetybelt_df['unique_key'].str.contains('for-learning/2023-11-19-omaezaki-500')) |
                (pred_safetybelt_df['unique_key'].str.contains('for-learning/2023-11-23-mie-safetybelt'))
            )
        ])
        train(train_df, valid_df, num_cv)

if __name__ == '__main__':
    annotation_safetybelt_df = pd.read_csv(f'{safetybelt_detection_label_dir}/safetybelt_eval_annotation_add_label-{pattern_date}.csv')
    pred_safetybelt_df = pd.read_csv(f'{safetybelt_detection_label_dir}/safetybelt_eval_pred_add_label-{pattern_date}.csv')
    train_cv(annotation_safetybelt_df, pred_safetybelt_df)
