import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

class MedusaDataset(Dataset):
    """
    Universal dataset for MEDUSA. Expects processed data organised as:
    data/processed/2d_1024px/xray_train/  (or xray_valid, etc.)
    data/annotations/ contains CSV files with columns 'image_path' and 'label'
    (and optionally 'class_name').
    """
    def __init__(self, root_2d, annotation_csv=None, transform=None):
        self.samples = []
        self.labels = {}
        # Load labels if provided
        if annotation_csv and Path(annotation_csv).exists():
            df = pd.read_csv(annotation_csv)
            for _, row in df.iterrows():
                self.labels[row['image_path']] = int(row['label'])
        # Gather images – now accepts .png, .jpg, .jpeg
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img_path in Path(root_2d).rglob(ext):
                self.samples.append(str(img_path))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        if self.transform:
            img = self.transform(img)
        # Label lookup (default 0 if not found)
        fname = Path(path).name
        label = self.labels.get(fname, 0)
        modality = '2D'  # all current small datasets are 2D
        return img, label, modality


class MedusaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root_2d='data/processed/2d_1024px',
        annotation_dir='data/annotations',
        batch_size=16,
        num_workers=0,
        train_folder='xray_train',
        val_folder='xray_valid',
        train_annotation_csv=None,
        val_annotation_csv=None,
    ):
        super().__init__()
        self.data_root_2d = data_root_2d
        self.annotation_dir = annotation_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.train_annotation_csv = train_annotation_csv
        self.val_annotation_csv = val_annotation_csv

    def setup(self, stage=None):
        # Use explicit CSV if provided, otherwise auto‑detect
        if self.train_annotation_csv:
            train_csv = self.train_annotation_csv
        else:
            train_csv = None
            for csv_file in Path(self.annotation_dir).glob('*.csv'):
                if 'train' in csv_file.stem.lower():
                    train_csv = str(csv_file)
                    break

        if self.val_annotation_csv:
            val_csv = self.val_annotation_csv
        else:
            val_csv = None
            for csv_file in Path(self.annotation_dir).glob('*.csv'):
                name = csv_file.stem.lower()
                if 'val' in name or 'valid' in name or 'test' in name:
                    val_csv = str(csv_file)
                    break

        self.train_dataset = MedusaDataset(
            root_2d=f'{self.data_root_2d}/{self.train_folder}',
            annotation_csv=train_csv
        )
        self.val_dataset = MedusaDataset(
            root_2d=f'{self.data_root_2d}/{self.val_folder}',
            annotation_csv=val_csv
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )