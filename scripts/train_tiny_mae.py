#!/usr/bin/env python
"""MEDUSA Tiny – small ViT MAE pre‑training on PneumoniaMNIST (CPU friendly)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.data.datamodule import MedusaDataModule

# -------------------------------------------------------------------
# Tiny Vision Transformer (no timm dependency, fully self‑contained)
# -------------------------------------------------------------------
class TinyViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2  # 196 for 224x224

        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)          # (B, E, H', W')
        x = x.flatten(2).transpose(1, 2) # (B, N, E)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)   # (B, N+1, E)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        return x  # (B, N+1, E)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# -------------------------------------------------------------------
# Tiny Decoder (from CLS token to full image)
# -------------------------------------------------------------------
class TinyDecoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 32 * 7 * 7)   # 128 -> 1568
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 7x7 -> 14x14
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),   # 14 -> 28
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, 4, 2, 1),    # 28 -> 56
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 2, 4, 2, 1),    # 56 -> 112
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.ConvTranspose2d(2, 1, 4, 2, 1),    # 112 -> 224
            nn.Sigmoid()
        )

    def forward(self, x):          # x: (B, embed_dim)
        x = self.fc(x)             # (B, 1568)
        x = x.view(-1, 32, 7, 7)  # (B, 32, 7, 7)
        return self.deconv(x)


# -------------------------------------------------------------------
# Lightning MAE module
# -------------------------------------------------------------------
class TinyMAE(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.encoder = TinyViT(embed_dim=128, depth=4, num_heads=4)
        self.decoder = TinyDecoder(embed_dim=128)
        self.criterion = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        features = self.encoder(x)
        cls_token = features[:, 0, :]        # (B, 128)
        return self.decoder(cls_token)

    def training_step(self, batch, batch_idx):
        x, _ = batch                 # ignore modality string, all are 2D
        pred = self(x)
        loss = self.criterion(pred, x)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# -------------------------------------------------------------------
# Main training
# -------------------------------------------------------------------
if __name__ == '__main__':
    dm = MedusaDataModule(batch_size=8, num_workers=0)
    model = TinyMAE()
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator='cpu',
        devices=1,
        log_every_n_steps=20,
    )
    trainer.fit(model, dm)
    print("Tiny MAE pre‑training complete. Model saved in lightning_logs/")