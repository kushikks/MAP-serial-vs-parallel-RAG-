import numpy as np
import faiss
import torch


class ExactIndex:
    """All documents scored exactly; deterministic score-desc/index-asc tie policy."""
    def __init__(self, embeddings, backend="faiss_cpu", chunk_size=1024):
        self.backend = backend
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if self.embeddings.ndim != 2 or len(self.embeddings) == 0 or not np.isfinite(self.embeddings).all():
            raise ValueError("Index requires finite nonempty 2D vectors")
        self.chunk_size = chunk_size
        self.oom_events = []
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if backend == "faiss_cpu":
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(self.embeddings)
        elif backend == "torch_cuda_exact":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA search requested but CUDA unavailable")
            self.vectors = torch.from_numpy(self.embeddings).to("cuda")
        else:
            raise ValueError(f"Unsupported search backend: {backend}")

    def search(self, queries, k=10):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if type(k) is not int or k < 1 or queries.ndim != 2 or queries.shape[1] != self.embeddings.shape[1] or not np.isfinite(queries).all():
            raise ValueError("Invalid query shape, values or top_k")
        k = min(k, len(self.embeddings))
        # Score all corpus documents in bounded query blocks; sorting gives a
        # reproducible cutoff for ties, including duplicated scalability inputs.
        all_scores, all_indices = [], []
        for start in range(0, len(queries), self.chunk_size):
            q = queries[start:start + self.chunk_size]
            if self.backend == "faiss_cpu":
                scores, indices = self.index.search(q, len(self.embeddings))
                ranked = [np.lexsort((ix, -sc))[:k] for sc, ix in zip(scores, indices)]
                all_scores.extend([sc[order] for sc, order in zip(scores, ranked)])
                all_indices.extend([ix[order] for ix, order in zip(indices, ranked)])
            else:
                try:
                    with torch.inference_mode():
                        scores = torch.from_numpy(q).to("cuda") @ self.vectors.T
                        # Stable argsort specifies ties by original corpus order.
                        indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
                        values = torch.gather(scores, 1, indices)
                        all_scores.extend(values.cpu().numpy())
                        all_indices.extend(indices.cpu().numpy())
                except torch.cuda.OutOfMemoryError:
                    if self.chunk_size == 1:
                        raise
                    old = self.chunk_size
                    self.chunk_size = max(1, self.chunk_size // 2)
                    self.oom_events.append({"old_chunk_size": old, "new_chunk_size": self.chunk_size})
                    torch.cuda.empty_cache()
                    return self.search(queries, k)
        return np.asarray(all_scores, dtype=np.float32).reshape(-1, k), np.asarray(all_indices, dtype=np.int64).reshape(-1, k)

    def save(self, path):
        if self.backend != "faiss_cpu":
            raise ValueError("Only CPU FAISS serializes as a FAISS index")
        faiss.write_index(self.index, str(path))


def parity(reference_scores, reference_indices, scores, indices, embeddings, queries, atol=3e-5):
    """Permit reordered near ties but reject any materially worse replacement."""
    if reference_scores.shape != scores.shape or not np.allclose(reference_scores, scores, atol=atol, rtol=0):
        raise AssertionError("Exact-search score parity failed")
    mismatches = int(np.sum(reference_indices != indices))
    for q, ref, actual in zip(queries, reference_indices, indices):
        if set(ref) != set(actual):
            cutoff = float(np.min(embeddings[ref] @ q))
            if any(float(embeddings[i] @ q) < cutoff - atol for i in actual):
                raise AssertionError("Non-tie document replacement")
    return {"passed": True, "rank_index_differences": mismatches, "score_atol": atol}
