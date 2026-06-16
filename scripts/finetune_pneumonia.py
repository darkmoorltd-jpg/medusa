#!/usr/bin/env python
"""
Fine‑tune MEDUSA on PneumoniaMNIST (2‑class classification).
Usage: python scripts/finetune_pneumonia.py
"""

import torch
import pytorch_lightning as pl
from src.data.datamodule import MedusaDataModule
from src.models.medusa_foundation import MedusaFoundation

def main():
    # Data
    dm = MedusaDataModule(batch_size=8, num_workers=0)

    # Model: classification with 2 classes (Normal, Pneumonia)
    model = MedusaFoundation(
        task='classification',
        num_classes=2,
        lr=1e-3
    )

    # Optional: load pre‑trained encoder weights (after Stage 1)
    # checkpoint = torch.load('lightning_logs/.../checkpoints/...ckpt')
    # model.encoder_2d.load_state_dict(checkpoint['encoder_2d'], strict=False)

    # Train
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='cpu',
        devices=1,
        log_every_n_steps=20
    )
    trainer.fit(model, dm)

    # Save final model
    torch.save(model.state_dict(), 'medusa_pneumonia_classifier.pt')
    print("Fine‑tuned model saved as medusa_pneumonia_classifier.pt")

if __name__ == '__main__':
    main()