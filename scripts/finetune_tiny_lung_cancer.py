#!/usr/bin/env python
"""Fine‑tune MEDUSA Tiny on lung cancer classification (benign / malignant / normal)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, glob
import torch.nn as nn
import pytorch_lightning as pl
from src.data.datamodule import MedusaDataModule

# -------------------------------------------------------------------
# TinyViT + TransformerBlock (same as other scripts)
# -------------------------------------------------------------------
class TinyViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
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
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        return self.norm(x)

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
# Lightning Classifier
# -------------------------------------------------------------------
class TinyLungCancerClassifier(pl.LightningModule):
    def __init__(self, mae_ckpt_path=None, num_classes=3, lr=1e-3):
        super().__init__()
        self.encoder = TinyViT(embed_dim=128, depth=4, num_heads=4)

        if mae_ckpt_path:
            print(f"Loading MAE weights from {mae_ckpt_path}")
            checkpoint = torch.load(mae_ckpt_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint['state_dict']
            encoder_state = {k.replace('encoder.', ''): v
                             for k, v in state_dict.items()
                             if k.startswith('encoder.')}
            self.encoder.load_state_dict(encoder_state, strict=False)
            print("MAE encoder loaded.")

        self.head = nn.Linear(128, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        feats = self.encoder(x)
        cls_token = feats[:, 0, :]
        return self.head(cls_token)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
if __name__ == '__main__':
    # FORCE TRAIN FROM SCRATCH (bypass corrupted checkpoint)
    latest_ckpt = None
    print("Training from scratch – no pretrained encoder loaded.")

    # If you later want to use a valid MAE checkpoint, comment the two lines above
    # and uncomment the following:
    #
    # ckpt_files = glob.glob('lightning_logs/version_*/checkpoints/*.ckpt')
    # latest_ckpt = max(ckpt_files, key=os.path.getmtime) if ckpt_files else None
    # print(f"Using MAE checkpoint: {latest_ckpt}")

    dm = MedusaDataModule(
        batch_size=16,
        num_workers=0,
        train_folder='lung_train',
        val_folder='lung_valid',
        train_annotation_csv='data/annotations/lung_cancer_train_labels.csv',
        val_annotation_csv='data/annotations/lung_cancer_valid_labels.csv'
    )

    model = TinyLungCancerClassifier(mae_ckpt_path=latest_ckpt)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='cpu',
        devices=1,
        log_every_n_steps=20,
    )
    trainer.fit(model, dm)

    torch.save(model.state_dict(), 'medusa_tiny_lung_cancer.pt')
    print("Lung cancer classifier saved as medusa_tiny_lung_cancer.pt")