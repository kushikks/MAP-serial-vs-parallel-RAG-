import numpy as np
import pytest
from src.indexing import ExactIndex, parity


def test_exact_ranking_ties_and_mapping():
    vectors = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    scores, ids = ExactIndex(vectors).search(vectors[:1], 10)
    assert ids.tolist() == [[0, 2, 1]]
    assert scores.tolist() == [[1, 1, 0]]
    with pytest.raises(ValueError):
        ExactIndex(vectors).search(vectors[:1], 0)


def test_random_numpy_reference():
    rng = np.random.default_rng(42)
    v = rng.normal(size=(100, 8)).astype("float32")
    q = rng.normal(size=(7, 8)).astype("float32")
    scores, ids = ExactIndex(v).search(q, 10)
    expected = np.argsort(-(q @ v.T), axis=1)[:, :10]
    assert np.array_equal(ids, expected)
    np.testing.assert_allclose(scores, np.take_along_axis(q @ v.T, expected, axis=1), atol=1e-5)
    serial_scores, serial_ids = ExactIndex(v, chunk_size=1).search(q, 10)
    assert np.array_equal(serial_ids, ids)
    np.testing.assert_allclose(serial_scores, scores, atol=1e-5)


def test_cuda_exact_parity():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    v = np.random.default_rng(42).normal(size=(100, 384)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q = v[:10]
    rs, ri = ExactIndex(v).search(q)
    ss, si = ExactIndex(v, "torch_cuda_exact", chunk_size=3).search(q)
    assert parity(rs, ri, ss, si, v, q)["passed"]
    assert torch.arange(1000, device="cuda").sum().item() == 499500
