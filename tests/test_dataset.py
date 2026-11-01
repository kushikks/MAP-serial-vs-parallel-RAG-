import pytest
from src.preprocessing import document, preprocess, claims
from src.common import ROOT, read


def test_sentences_ids_and_join(rows):
    docs = preprocess(rows)
    assert docs[0]["doc_id"] == "001"
    assert docs[0]["abstract_sentences"] == ["First.", "Second."]
    assert docs[0]["text"] == "Synthetic title\nFirst. Second."
    assert preprocess(rows, workers=2) == docs


@pytest.mark.parametrize("row", [{}, {"doc_id": 1, "title": "", "abstract": []},
                                  {"doc_id": True, "title": "a", "abstract": ["b"]},
                                  {"doc_id": 1, "title": "a", "abstract": "bad"}])
def test_malformed(row):
    with pytest.raises((KeyError, ValueError)):
        document(row)


def test_original_and_hf_claim_schemas():
    original = [{"id": "01", "claim": "fixture", "evidence": {"001": [{"label": "SUPPORT", "sentences": [0]}]}, "cited_doc_ids": [999]}]
    expanded = [{"id": "01", "claim": "fixture", "evidence_doc_id": "001"}] * 2
    assert claims(original, "dev") == claims(expanded, "dev")
    assert claims(original, "dev")[0]["relevant_doc_ids"] == ["001"]
    assert claims([{"id": 2, "claim": "none", "evidence": {}}], "dev")[0]["relevant_doc_ids"] == []
    with pytest.raises(ValueError, match="Missing evidence schema"):
        claims([{"id": 2, "claim": "missing"}], "dev")


def test_real_dataset_validation():
    path = ROOT / "data/manifests/validation.json"
    if not path.exists():
        pytest.skip("Run prepare to validate real SciFact")
    report = read(path)
    assert report["valid"] and report["documents"] == 5183
    assert report["splits"]["dev"]["claims"] == 300
