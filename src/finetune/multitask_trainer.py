import torch
import pytorch_lightning as pl
from src.models.medusa_foundation import MedusaFoundation
from src.data.datamodule import MedusaDataModule

class MultiTaskTrainer(pl.LightningModule):
    """
    Wrapper that holds a MedusaFoundation and handles multiple
    datasets/tasks by alternating batches. Simplification: we just
    forward to the base model with task automatically set per batch.
    """
    def __init__(self, foundation_model):
        super().__init__()
        self.model = foundation_model

    def training_step(self, batch, batch_idx):
        # batch is expected to include 'task' info
        # For now we assume a single task, but you can route per batch
        loss = self.model.training_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.model.configure_optimizers()

def train_multitask():
    """
    Example usage:
    python -m src.finetune.multitask_trainer
    """
    # 1. Data: mock multi‑modal loader (replace with real data)
    dm = MedusaDataModule(batch_size=2)

    # 2. Model: pre‑trained encoder loaded
    model = MedusaFoundation(task='classification', num_classes=14)  # e.g., ChestX‑ray14
    # model.load_encoder_weights('pretrained_mae.ckpt')

    # 3. Trainer
    wrapper = MultiTaskTrainer(model)
    trainer = pl.Trainer(max_epochs=10, accelerator='cpu', devices=1)
    trainer.fit(wrapper, dm)

if __name__ == '__main__':
    train_multitask()