import albumentations as A
from albumentations.pytorch import ToTensorV2

def transform_train(input_img_size: int=224):
    A_transforms = A.Compose([
        A.Resize(input_img_size, input_img_size),
        A.HorizontalFlip(p=0.3),
        A.GaussianBlur(),
        A.GaussNoise(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, interpolation=1, border_mode=4,  p=0.3),
        A.RandomGamma(gamma_limit=(85, 150), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=True, p=0.4),
        A.CoarseDropout(max_holes=4, max_height=100, max_width=100, min_holes=1, min_height=50, min_width=50, fill_value=0, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return A_transforms

def transform_val(input_img_size: int=224):
    A_transforms_val = A.Compose([
        A.Resize(input_img_size, input_img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return A_transforms_val
