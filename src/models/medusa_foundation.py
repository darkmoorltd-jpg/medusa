import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.models.backbone.medusa_encoder import MedusaEncoder
from src.models.heads.classification_head import ClassificationHead
from src.models.heads.segmentation_unetr import SegmentationUNETR
from src.models.heads.detection_detr import DetectionDETR

class MedusaFoundation(pl.LightningModule):
    """
    Multi‑task, multi‑modal MEDUSA model.
    During fine‑tuning, you set the active task via `task` argument.
    """
    def __init__(
        self,
        encoder_2d=None,
        encoder_3d=None,
        task='classification',  # 'classification', 'segmentation', 'detection'
        num_classes=2,
        lr=1e-4,
        pretrained_encoder_path=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.task = task

        # Encoders
        if encoder_2d is None:
            self.encoder_2d = MedusaEncoder(use_3d=False)
        else:
            self.encoder_2d = encoder_2d

        if encoder_3d is None:
            self.encoder_3d = MedusaEncoder(use_3d=True)
        else:
            self.encoder_3d = encoder_3d

        # Task‑specific heads
        if task == 'classification':
            self.head = ClassificationHead(embed_dim=768, num_classes=num_classes)
        elif task == 'segmentation':
            self.head = SegmentationUNETR(embed_dim=768, out_channels=num_classes)
        elif task == 'detection':
            self.head = DetectionDETR(embed_dim=768, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown task: {task}")

        self.criterion = self._get_criterion()

        if pretrained_encoder_path:
            self.load_encoder_weights(pretrained_encoder_path)

    def load_encoder_weights(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.encoder_2d.load_state_dict(checkpoint['encoder_2d'], strict=False)
        self.encoder_3d.load_state_dict(checkpoint['encoder_3d'], strict=False)

    def _get_criterion(self):
        if self.task == 'classification':
            return nn.CrossEntropyLoss()
        elif self.task == 'segmentation':
            return nn.BCEWithLogitsLoss()  # or DiceLoss
        elif self.task == 'detection':
            return nn.L1Loss()  # placeholder
        return nn.MSELoss()

    def forward(self, x, modality_id, is_3d=False):
        encoder = self.encoder_3d if is_3d else self.encoder_2d
        features = encoder(x, modality_id)
        # For classification, use the cls token; for seg/det, use full feature map
        if self.task == 'classification':
            cls_token = features[:, 0]
            out = self.head(cls_token)
        else:
            # For segmentation/detection, we need to reshape the transformer output
            # Placeholder: pass all tokens to head (real impl would reshape)
            out = self.head(features)
        return out

    def training_step(self, batch, batch_idx):
        x, target, modality, is_3d, *meta = batch
        mod_id = torch.tensor([modality], device=x.device)
        pred = self(x, mod_id, is_3d)
        loss = self.criterion(pred, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, target, modality, is_3d, *meta = batch
        mod_id = torch.tensor([modality], device=x.device)
        pred = self(x, mod_id, is_3d)
        loss = self.criterion(pred, target)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer