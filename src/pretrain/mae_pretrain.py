import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from src.models.backbone.medusa_encoder import MedusaEncoder

class MedusaMAE(pl.LightningModule):
    """
    Masked Autoencoder pre‑training for MEDUSA.
    Uses a lightweight convolutional decoder that upsamples the
    encoder's CLS token to reconstruct the full image.
    """
    def __init__(
        self,
        encoder_2d=None,
        encoder_3d=None,
        mask_ratio=0.75,
        lr=1e-4,
        img_size_2d=224,
        embed_dim=768
    ):
        super().__init__()
        self.encoder_2d = encoder_2d or MedusaEncoder(use_3d=False)
        self.encoder_3d = encoder_3d or MedusaEncoder(use_3d=True)
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.img_size_2d = img_size_2d
        self.embed_dim = embed_dim

        # Lightweight 2D decoder: CLS token → image
        self.decoder_2d = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128 * (img_size_2d // 4) * (img_size_2d // 4)),  # 128 x 56 x 56
            nn.Unflatten(1, (128, img_size_2d // 4, img_size_2d // 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 56 → 112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 112 → 224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),                       # 224 → 224 (1 channel)
            nn.Sigmoid()
        )

        # 3D decoder (dummy – same idea, but left as placeholder)
        self.decoder_3d = nn.Linear(embed_dim, 16 * 16 * 16)

        self.criterion = nn.MSELoss()

    def forward(self, x, modality_id, is_3d=False):
        encoder = self.encoder_3d if is_3d else self.encoder_2d
        # Get encoder features: (B, num_patches+1, embed_dim)
        features = encoder(x, modality_id)
        cls_token = features[:, 0, :]          # (B, embed_dim)

        if is_3d:
            # 3D placeholder
            reconstruction = self.decoder_3d(cls_token)
        else:
            reconstruction = self.decoder_2d(cls_token)   # (B, 1, 224, 224)
        return reconstruction

    def training_step(self, batch, batch_idx):
        x, modality = batch        # x: (B,1,H,W) or (B,1,D,H,W)
        is_3d = (x.dim() == 5)
        # Map modality string to ID: 0 = 2D, 1 = 3D
        mod_id = torch.tensor([1 if is_3d else 0], device=x.device)

        # For true MAE you'd mask patches, but here we simply reconstruct
        pred = self(x, mod_id, is_3d)
        loss = self.criterion(pred, x)   # input vs. reconstruction
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer