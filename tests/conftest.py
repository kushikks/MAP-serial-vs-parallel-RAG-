import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import environment
environment(1)
import faiss  # Match application native-library load order.
import pytest


@pytest.fixture
def rows():
    # Explicitly synthetic unit fixtures; never used as research observations.
    return [{"doc_id": "001", "title": "Synthetic title", "abstract": ["First.", "Second."], "structured": True},
            {"doc_id": 2, "title": "Another synthetic title", "abstract": ["Other."]}]


@pytest.fixture(scope="session")
def model_config():
    return {"name": "test", "device": "cpu", "backend": "faiss_cpu", "workers": 1, "threads": 1,
            "batch_size": 1, "loader_workers": 0, "max_length": 256}
