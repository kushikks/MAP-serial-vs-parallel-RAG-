from src.runtime import configure


def test_verified_serial_native_pools():
    limits, info = configure(1)
    assert info["torch_threads"] == info["torch_interop"] == info["faiss_threads"] == 1
    assert all(pool["num_threads"] == 1 for pool in info["pools"])
