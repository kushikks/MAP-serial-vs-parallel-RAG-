import pytest
from src.evaluation import evaluate


def test_metrics_and_unannotated_exclusion():
    claims = [{"id": "a", "relevant_doc_ids": ["1", "2"]}, {"id": "b", "relevant_doc_ids": []}]
    result = evaluate(claims, [["0", "1", "2"], ["1"]])
    m = result["metrics"]
    assert m["recall@1"] == 0 and m["recall@5"] == 1
    assert m["mrr"] == m["mrr@10"] == 0.5
    assert 0 < m["ndcg@10"] < 1
    assert result["excluded_no_evidence"] == 1


def test_empty_and_duplicates():
    assert evaluate([], [])["metrics"]["mrr"] is None
    with pytest.raises(ValueError):
        evaluate([{"id": "x", "relevant_doc_ids": ["1"]}], [["1", "1"]])


def test_full_mrr_differs_from_cutoff():
    r = evaluate([{"id": "x", "relevant_doc_ids": ["gold"]}], [[str(i) for i in range(10)] + ["gold"]])
    assert r["metrics"]["mrr"] == pytest.approx(1 / 11)
    assert r["metrics"]["mrr@10"] == 0
