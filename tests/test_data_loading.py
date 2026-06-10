import pytest
import torch
from src.data.datamodule import MedusaDataModule, MedusaDataset

def test_dataset_creation(tmp_path):
    # Create a dummy 2D image
    d = tmp_path / "2d" / "xray"
    d.mkdir(parents=True)
    img = (torch.rand(1,224,224) * 255).byte()
    torch.save(img, d / "test.png")
    ds = MedusaDataset(root_2d=tmp_path/"2d", root_3d=None)
    assert len(ds) == 1

def test_datamodule():
    dm = MedusaDataModule(batch_size=2)
    assert dm.train_dataloader() is not None