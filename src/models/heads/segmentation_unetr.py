import torch
import torch.nn as nn
from monai.networks.nets import UNETR

class SegmentationUNETR(nn.Module):
    """
    Wrapper around MONAI's UNETR for 3D segmentation.
    Input: transformer features (B, N, embed_dim) from MEDUSA encoder.
    We will use the raw volume as input and this head as a decoder,
    but since our encoder is ViT-based, we adapt the UNETR which expects
    the ViT output. For simplicity, we assume the encoder provides
    the hidden states that UNETR can consume (not just the cls token).
    In practice, you'd extract the sequence and reshape.
    """
    def __init__(self, embed_dim=768, out_channels=4, img_size=(128,128,128)):
        super().__init__()
        self.unetr = UNETR(
            in_channels=1,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=16,
            hidden_size=embed_dim,
            mlp_dim=3072,
            num_heads=12,
            pos_embed='perceptron',
            norm_name='instance',
            conv_block=True,
            res_block=True,
            dropout_rate=0.0
        )
        # For now, we bypass the encoder of UNETR and directly
        # feed the transformer features. But for simplicity,
        # we just use the full UNETR as the head (it will re‑extract patches).
        # You can later customise to inject pre‑trained features.

    def forward(self, x):
        # x: raw 3D volume (B,1,D,H,W)
        return self.unetr(x)