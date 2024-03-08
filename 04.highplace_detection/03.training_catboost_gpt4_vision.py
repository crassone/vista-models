from catboost import Pool, CatBoostClassifier
import numpy as np
import pandas as pd
import os
from sklearn.metrics import precision_recall_curve, average_precision_score
import yaml
from utils.class_category import class_category, no_alarm_class_category, classname2number

current_dir = os.path.dirname(os.path.abspath(__file__))
config = yaml.load(open(f'{current_dir}/config.yaml', 'r'), Loader=yaml.SafeLoader)

pattern_date = '2023-12-24'
root_dir = f'{"/".join(current_dir.split("/")[:-1])}'
highplace_emb_dir = f'{root_dir}/data/outputs/04.highplace_detection/01.gpt4_text_and_emb'
result_dir = f'{root_dir}/data/outputs/04.highplace_detection/03.gpt4_pred_highplace_detection'

def train(train_df, valid_df, num_cv):
    class_num = [train_df['label'].value_counts()[class_names].sum() for class_names in class_category]
    weight_class = [round(max(class_num) / i) for i in class_num]

    train_weight_list = []
    for weight, class_name in zip(weight_class, class_category):
        for _ in range(weight):
            train_weight_list.append(train_df[train_df['label'] == class_name])
    train_df_weight = pd.concat(train_weight_list)

    X_train = train_df_weight.iloc[:, 12:].values
    y_train = train_df_weight['label'].apply(classname2number).values
    X_test = valid_df.iloc[:, 12:].values
    y_test = valid_df['label'].apply(classname2number).values

    # カテゴリのカラムのみを抽出
    categorical_features_indices = np.array([])

    # データセットの作成。Poolで説明変数、目的変数、
    # カラムのデータ型を指定できる
    train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
    validate_pool = Pool(X_test, y_test, cat_features=categorical_features_indices)

    params = {
        'depth': config['depth'],
        'learning_rate': config['learning_rate'],
        'early_stopping_rounds': config['early_stopping_rounds'],
        'iterations': config['iterations'],
        'random_seed': 42
    }
    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=validate_pool,
        use_best_model=True,
        plot=False,
        verbose=False
    )
    prob_np = model.predict(X_test, prediction_type='Probability')
    labels_np = y_test

    # PR-AUC
    for class_num, class_names in enumerate(class_category):
        if class_names in no_alarm_class_category:
            continue

        use_label = [i for i in labels_np if i in [j for j in range(len(no_alarm_class_category))] + [class_num]]
        use_prob = [j for i, j in zip(labels_np, prob_np) if i in [j for j in range(len(no_alarm_class_category))] + [class_num]]
        use_label = np.array([1 if i in [class_num] else 0 for i in use_label])
        use_prob = np.array([i[class_num] for i in use_prob])
        average_precision = average_precision_score(use_label, use_prob)
        precisions, recalls, thresholds = precision_recall_curve(use_label, use_prob)
        result_precision03 = [0, 0, 0] # precision, recall, threshold で保存
        result_precision04 = [0, 0, 0] # precision, recall, threshold で保存
        result_precision05 = [0, 0, 0] # precision, recall, threshold で保存
        result_precision06 = [0, 0, 0] # precision, recall, threshold で保存
        for p, r, t in zip(precisions, recalls, thresholds):
            if p >= 0.3:
                result_precision03 = [p, r, t]
                break
        for p, r, t in zip(precisions, recalls, thresholds):
            if p >= 0.4:
                result_precision04 = [p, r, t]
                break
        for p, r, t in zip(precisions, recalls, thresholds):
            if p >= 0.5:
                result_precision05 = [p, r, t]
                break
        for p, r, t in zip(precisions, recalls, thresholds):
            if p >= 0.6:
                result_precision06 = [p, r, t]
                break

        print(f'{class_names} - PR-AUC: {average_precision:.4f}')
        print(f'precision@{result_precision03[0]:.2f} - Recall: {result_precision03[1]:.4f} - Threshold: {result_precision03[2]:.4f}')
        print(f'precision@{result_precision04[0]:.2f} - Recall: {result_precision04[1]:.4f} - Threshold: {result_precision04[2]:.4f}')
        print(f'precision@{result_precision05[0]:.2f} - Recall: {result_precision05[1]:.4f} - Threshold: {result_precision05[2]:.4f}')
        print(f'precision@{result_precision06[0]:.2f} - Recall: {result_precision06[1]:.4f} - Threshold: {result_precision06[2]:.4f} \n')

    os.makedirs(f'{result_dir}/{num_cv}', exist_ok=True)
    model.save_model(f'{result_dir}/{num_cv}/catboost.cbm')

def train_cv(pred_person_highplace_df):
    for num_cv in range(3):
        train_df = pred_person_highplace_df[
            (pred_person_highplace_df['validation'] != num_cv) &
            (pred_person_highplace_df['validation'] != 999) &
            (
                (pred_person_highplace_df['unique_key'].str.contains('fixed-point-camera')) |
                (pred_person_highplace_df['unique_key'].str.contains('for-learning/2023-11-19-omaezaki-500')) |
                (pred_person_highplace_df['unique_key'].str.contains('for-learning/2023-11-23-mie-safetybelt'))
            )
        ]
        valid_df = pred_person_highplace_df[
            (pred_person_highplace_df['validation'] == num_cv) &
            (
                (pred_person_highplace_df['unique_key'].str.contains('fixed-point-camera')) |
                (pred_person_highplace_df['unique_key'].str.contains('for-learning/2023-11-19-omaezaki-500')) |
                (pred_person_highplace_df['unique_key'].str.contains('for-learning/2023-11-23-mie-safetybelt'))
            )
        ]
        train(train_df, valid_df, num_cv)

if __name__ == '__main__':
    pred_person_highplace_df = pd.read_csv(f'{highplace_emb_dir}/person_eval_pred_message_and_emb-{pattern_date}.csv')
    train_cv(pred_person_highplace_df)
