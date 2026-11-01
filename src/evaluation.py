import math
import numpy as np


def evaluate(claims, rankings):
    if len(claims) != len(rankings):
        raise ValueError("One ranking required per claim")
    values = {"recall@1": [], "recall@5": [], "recall@10": [], "mrr": [], "mrr@10": [], "ndcg@10": []}
    per_claim = []
    for claim, ranking in zip(claims, rankings):
        if len(ranking) != len(set(ranking)):
            raise ValueError("Duplicate retrieved document IDs")
        relevant = set(claim["relevant_doc_ids"])
        if not relevant:
            continue
        row = {"id": claim["id"]}
        for k in (1, 5, 10):
            row[f"recall@{k}"] = len(relevant & set(ranking[:k])) / len(relevant)
        ranks = [i + 1 for i, did in enumerate(ranking[:10]) if did in relevant]
        full_ranks = [i + 1 for i, did in enumerate(ranking) if did in relevant]
        row["mrr"] = 1 / min(full_ranks) if full_ranks else 0.0
        row["mrr@10"] = 1 / min(ranks) if ranks else 0.0
        dcg = sum(1 / math.log2(rank + 1) for rank in ranks)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(relevant))))
        row["ndcg@10"] = dcg / ideal
        for name in values:
            values[name].append(row[name])
        per_claim.append(row)
    return {"metrics": {k: float(np.mean(v)) if v else None for k, v in values.items()},
            "eligible_claims": len(per_claim), "excluded_no_evidence": len(claims) - len(per_claim),
            "mrr_definition": "Full supplied ranking; benchmark supplies all corpus documents. mrr@10 also reported.",
            "per_claim": per_claim}
