import pytest
from src.common import ROOT, read


def validate_contract(results):
    assert isinstance(results, list) and results
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))
    assert len({r["doc_id"] for r in results}) == len(results)
    for row in results:
        assert set(row) == {"doc_id", "title", "abstract", "abstract_sentences", "retrieval_score", "rank"}
        assert isinstance(row["doc_id"], str)
        assert row["abstract"] == " ".join(row["abstract_sentences"])
        assert isinstance(row["retrieval_score"], float)
    assert all(a["retrieval_score"] >= b["retrieval_score"] for a, b in zip(results, results[1:]))


def test_fixed_actual_output():
    fixture = ROOT / "examples/retrieval_output.json"
    if not fixture.exists():
        pytest.skip("Actual retrieval fixture not generated yet")
    validate_contract(read(fixture))


def test_live_importable_api():
    if not (ROOT / "artifacts/serial_cpu/metadata.json").exists():
        pytest.skip("Serial index not built")
    from retrieval import retrieve
    result = retrieve("Vitamin D affects bone health.", 3)
    validate_contract(result)
    with pytest.raises(ValueError):
        retrieve("", 3)
