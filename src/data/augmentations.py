import albumentations as A
from albumentations.pytorch import ToTensorV2

train_aug_2d = A.Compose([
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    ToTensorV2()
], additional_targets=None)

# 3D augmentations via MONAI
import monai.transforms as MT
train_aug_3d = MT.Compose([
    MT.RandFlip(prob=0.5, spatial_axis=0),
    MT.RandAffine(rotate_range=10, translate_range=10, scale_range=0.1, prob=0.5),
    MT.RandGaussianNoise(prob=0.2, mean=0.0, std=0.1),
])