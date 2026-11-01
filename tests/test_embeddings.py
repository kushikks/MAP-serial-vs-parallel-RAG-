import numpy as np
import pytest
import torch
from src.embeddings import Embedder, validate_embeddings
from src.common import ROOT


def test_invalid_embeddings():
    with pytest.raises(ValueError):
        validate_embeddings(np.zeros((2, 384), dtype="float32"))
    with pytest.raises(ValueError):
        validate_embeddings(np.ones((2, 10), dtype="float32"))


def test_real_model_shape_norm(model_config):
    if not (ROOT / "data/manifests/model.json").exists():
        pytest.skip("Model not downloaded")
    e = Embedder(model_config)
    vectors = e.encode(["A scientific claim.", "Another claim."])
    assert vectors.shape == (2, 384)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5)
    with pytest.raises(ValueError):
        e.encode([""])


def test_single_query_does_not_spawn_workers(model_config, monkeypatch):
    if not (ROOT / "data/manifests/model.json").exists():
        pytest.skip("Model not downloaded")
    from src import embeddings
    original_loader = embeddings.DataLoader
    observed = []
    def loader(*args, **kwargs):
        observed.append(kwargs["num_workers"])
        assert kwargs["num_workers"] == 0
        return original_loader(*args, **kwargs)
    monkeypatch.setattr(embeddings, "DataLoader", loader)
    e = Embedder(dict(model_config, loader_workers=4))
    vector = e.encode(["A single interactive scientific claim."])
    assert vector.shape == (1, 384) and observed == [0]


def test_oom_recovery(monkeypatch, model_config):
    # Inject OOM to verify retry policy without exhausting the user's GPU.
    e = object.__new__(Embedder)
    e.device, e.batch_size, e.oom_events = "cuda", 64, []
    calls = []
    def encode(texts):
        calls.append(e.batch_size)
        if len(calls) < 3:
            raise torch.cuda.OutOfMemoryError("injected")
        return np.ones((1, 384), dtype="float32")
    monkeypatch.setattr(e, "_encode", encode)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    e.encode(["claim"])
    assert calls == [64, 32, 16] and len(e.oom_events) == 2


def test_oom_at_one_propagates(monkeypatch):
    e = object.__new__(Embedder)
    e.device, e.batch_size, e.oom_events = "cuda", 1, []
    def fail(texts):
        raise torch.cuda.OutOfMemoryError("injected")
    monkeypatch.setattr(e, "_encode", fail)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        e.encode(["claim"])
