import os
import random

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def environment(threads):
    if threads < 1:
        raise ValueError("threads must be positive")
    for name in THREAD_VARS:
        os.environ[name] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_DISABLE_XET"] = "1"


def configure(threads):
    environment(threads)
    import faiss
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits, threadpool_info
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    faiss.omp_set_num_threads(threads)
    limits = threadpool_limits(limits=threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    info = {"environment": {n: os.environ[n] for n in THREAD_VARS},
            "torch_threads": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
            "faiss_threads": faiss.omp_get_max_threads(), "pools": threadpool_info()}
    # OpenMP stub runtimes can truthfully report one even when a larger upper
    # limit is requested. They must not be mistaken for an active four-thread
    # kernel pool; retain their actual reported count in metadata.
    if info["torch_threads"] != threads or info["faiss_threads"] != threads or any(p["num_threads"] > threads for p in info["pools"]):
        raise RuntimeError(f"Thread limit verification failed: {info}")
    return limits, info
