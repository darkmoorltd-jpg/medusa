import torch

def setup_distributed():
    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend='nccl')
    else:
        print("Running on CPU, no distributed init.")