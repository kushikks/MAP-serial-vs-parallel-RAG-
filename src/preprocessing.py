from concurrent.futures import ThreadPoolExecutor


def identifier(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError(f"Invalid identifier: {value!r}")
    return str(value)  # Never renumber or coerce strings to integers.


def document(row):
    doc_id = identifier(row["doc_id"])
    title = row.get("title")
    sentences = row.get("abstract")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"Missing title: {doc_id}")
    # SciFact structured abstracts are still ordered lists of sentences.
    if not isinstance(sentences, list) or not sentences or any(not isinstance(s, str) or not s.strip() for s in sentences):
        raise ValueError(f"Malformed abstract sentences: {doc_id}")
    abstract = " ".join(sentences)
    return {"doc_id": doc_id, "title": title, "abstract": abstract,
            "abstract_sentences": sentences.copy(), "text": title + "\n" + abstract}


def preprocess(rows, workers=1):
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        return [document(r) for r in rows]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(document, rows))


def claims(rows, split):
    """Accept original nested records or group the inspected HF expanded schema."""
    grouped = {}
    for row in rows:
        cid = identifier(row["id"])
        text = row.get("claim")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty claim: {cid}")
        if cid in grouped and grouped[cid]["claim"] != text:
            raise ValueError(f"Conflicting claim ID: {cid}")
        item = grouped.setdefault(cid, {"id": cid, "claim": text, "split": split, "relevant_doc_ids": []})
        relevant = []
        if split != "test" and "evidence" not in row and "evidence_doc_id" not in row:
            raise ValueError(f"Missing evidence schema for annotated split: {cid}")
        if "evidence" in row:
            if not isinstance(row["evidence"], dict):
                raise ValueError(f"Malformed evidence: {cid}")
            for did, annotations in row["evidence"].items():
                if not isinstance(annotations, list) or not annotations:
                    raise ValueError(f"Empty/malformed evidence annotations: {cid}/{did}")
                for annotation in annotations:
                    if annotation.get("label") not in ("SUPPORT", "CONTRADICT") or not annotation.get("sentences"):
                        raise ValueError(f"Malformed evidence annotation: {cid}/{did}")
                relevant.append(identifier(did))
        elif row.get("evidence_doc_id"):
            relevant.append(identifier(row["evidence_doc_id"]))
        for did in relevant:
            if did not in item["relevant_doc_ids"]:
                item["relevant_doc_ids"].append(did)
    return list(grouped.values())
