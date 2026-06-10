import torch
import pytest
from src.models.backbone.medusa_encoder import MedusaEncoder

def test_encoder_2d():
    encoder = MedusaEncoder(use_3d=False)
    x = torch.randn(2, 1, 224, 224)
    mod_id = torch.tensor([0, 1])
    out = encoder(x, mod_id)
    assert out.shape[0] == 2
    assert out.shape[-1] == 768

def test_encoder_3d():
    encoder = MedusaEncoder(use_3d=True)
    x = torch.randn(2, 1, 128, 128, 128)
    mod_id = torch.tensor([0, 1])
    out = encoder(x, mod_id)
    assert out.shape[0] == 2