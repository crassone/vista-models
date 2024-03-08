import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset
from .class_category import class_category

class ClassificationDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, training=False):
        self.img_paths = [f'{root_dir}/data/images/{path}' for path in df['relative_img_path'].values]
        self.labels = list(df['label'].values)
        self.bboxes = list(df[['left', 'top', 'right', 'bottom']].values)
        self.transform = transform
        self.training = training

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        bbox = self.bboxes[idx]
        if self.training:
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            img_width, img_height = image.size

            random_ratio = random.uniform(0, 0.2)
            add_bbox_width = bbox_width * random_ratio
            add_bbox_height = bbox_height * random_ratio
            bbox[0] = bbox[0] - add_bbox_width / 2 if bbox[0] - add_bbox_width / 2 > 0 else 0
            bbox[1] = bbox[1] - add_bbox_height / 2 if bbox[1] - add_bbox_height / 2 > 0 else 0
            bbox[2] = bbox[2] + add_bbox_width / 2 if bbox[2] + add_bbox_width / 2 < img_width else img_width
            bbox[3] = bbox[3] + add_bbox_height / 2 if bbox[3] + add_bbox_height / 2 < img_height else img_height

            cropped_image = image.crop(bbox)
        else:
            cropped_image = image.crop(bbox)

        img = np.array(cropped_image)
        augmented = self.transform(image=img)
        image = augmented['image']

        for i, category_names in enumerate(class_category):
            if self.labels[idx] in category_names:
                label = i
        return image, label

    def __len__(self):
        return len(self.df)
