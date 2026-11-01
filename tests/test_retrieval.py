import pytest
from src.common import ROOT


def test_engine_rejects_invalid_input():
    from src.retrieval_engine import RetrievalEngine
    engine = object.__new__(RetrievalEngine)
    for claim, k in [("", 10), (None, 10), ("claim", 0), ("claim", True)]:
        with pytest.raises(ValueError):
            engine.retrieve(claim, k)
