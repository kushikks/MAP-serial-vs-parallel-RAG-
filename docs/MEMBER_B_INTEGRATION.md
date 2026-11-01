# Member B integration

Member A returns retrieved documents only. Member B independently selects evidence sentences, reranks and predicts SUPPORT / REFUTE / INSUFFICIENT EVIDENCE.

```python
from retrieval import retrieve
documents = retrieve("Vitamin D affects bone health.", top_k=10)
for document in documents:
    sentences = document["abstract_sentences"]
    # Apply your own evidence extraction/reranking/verification here.
```

The convenience API lazily uses the saved full serial baseline and one-thread CPU model. It requires `prepare`, `download-model`, and the serial benchmark to have run. For a different index instantiate `src.retrieval_engine.RetrievalEngine(artifact_dir, config)` after configuring thread limits in your fresh process. Avoid changing global PyTorch interop settings after inference has already begun.

On this Windows environment, load `faiss` before `torch` in a combined application to avoid the observed intermittent DLL initialization issue. The CLI manages this import order and can be called as a separate process if Member B's application has already initialized its native libraries.

Single-claim encoding always uses zero DataLoader workers, matching measured online latency and avoiding process startup per request. Configured worker counts apply to multi-item embedding workloads.

`retrieve(claim: str, top_k: int = 10) -> list[dict]` returns:

* `doc_id`: original identifier represented as a string without renumbering.
* `title`: original document title.
* `abstract`: original sentence list joined by spaces.
* `abstract_sentences`: original ordered sentences, suitable for sentence indices.
* `retrieval_score`: cosine similarity, a float rather than a probability or verdict.
* `rank`: one-based rank.

Empty/non-string claims and nonpositive/noninteger k raise `ValueError`. k larger than the corpus is clamped to its size. Ties use original corpus order. The API verifies artifact checksums, model revision, mapping length and truncation compatibility. It does not infer evidence labels from retrieved articles.

CLI JSON input is in `examples/query.json`:

```powershell
.\.venv-rag\Scripts\python.exe retrieval.py query --input examples/query.json --output examples/retrieval_output.json
```

`examples/retrieval_output.json` is a fixed **actual** retrieval output once generated, so Member B can develop by reading it with `json.load` without installing models or waiting for inference. It is a contract fixture, not a claim label or accuracy result. `tests/test_member_b_interface.py` checks both this fixed fixture and the importable API. No fabricated scientific records are provided as evidence.
