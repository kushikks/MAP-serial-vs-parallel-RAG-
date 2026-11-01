import numpy as np
import pytest
from src.embeddings import Embedder
from src.common import ROOT


def test_batch_cpu_cuda_parity(model_config):
    import torch
    if not (ROOT / "data/manifests/model.json").exists():
        pytest.skip("Model not cached")
    texts = ["Cells respond to a chemical signal.", "Proteins regulate transcription.", "Vitamin D affects bone health."]
    serial = Embedder(model_config).encode(texts)
    parallel = Embedder(dict(model_config, batch_size=3)).encode(texts)
    np.testing.assert_allclose(serial, parallel, atol=1e-5, rtol=0)
    if torch.cuda.is_available():
        gpu = Embedder(dict(model_config, device="cuda", batch_size=3)).encode(texts)
        np.testing.assert_allclose(serial, gpu, atol=3e-5, rtol=0)
