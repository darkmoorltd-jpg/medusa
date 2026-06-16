import torch
from src.models.medusa_foundation import MedusaFoundation
from src.data.datamodule import MedusaDataModule

BENCHMARK_CONFIGS = {
    'chexpert': {
        'task': 'classification',
        'num_classes': 14,
        'data_2d': 'data/processed/2d_1024px/xray_train',  # adjust
    },
    'rsna_pneumonia': {
        'task': 'detection',
        'num_classes': 1,
        'data_2d': 'data/processed/2d_1024px/xray_pneumonia',
    },
    'brats': {
        'task': 'segmentation',
        'num_classes': 4,
        'data_3d': 'data/processed/3d_224px/mri_brats',
    }
}

def run_benchmark(benchmark_name, checkpoint_path=None):
    config = BENCHMARK_CONFIGS[benchmark_name]
    model = MedusaFoundation(
        task=config['task'],
        num_classes=config['num_classes'],
        pretrained_encoder_path=checkpoint_path
    )
    # Dummy evaluation (in real life, you'd use the specific test set)
    dm = MedusaDataModule(
        data_root_2d=config.get('data_2d'),
        data_root_3d=config.get('data_3d'),
        batch_size=1
    )
    trainer = pl.Trainer(accelerator='cpu', devices=1, logger=False)
    # You would call trainer.test(model, dm) with a proper test set
    print(f"Benchmark {benchmark_name} ready. Run test with actual data.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        run_benchmark(sys.argv[1])
    else:
        print("Usage: python -m src.evaluation.benchmark_suite <benchmark_name>")