import gc
import random
import logging
import os
import sys
import numpy as np
import torch
import torch.distributed as dist
# from loguru import logger

def init_dist():
    """Initializes distributed environment."""
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank


def set_manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_contiguous(x):
    if isinstance(x, torch.Tensor):
        return x.contiguous()
    elif isinstance(x, dict):
        return {k: make_contiguous(v) for k, v in x.items()}
    else:
        return x


class set_worker_seed_builder():
    def __init__(self, global_rank):
        self.global_rank = global_rank

    def __call__(self, worker_id):
        set_manual_seed(torch.initial_seed() % (2 ** 32 - 1))

def free_memory():
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def set_logging(local_rank):
    if local_rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)
