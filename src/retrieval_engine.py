from pathlib import Path
import numpy as np
from .common import ROOT, read, checksum, fingerprint
from .embeddings import Embedder, validate_embeddings
from .indexing import ExactIndex


class RetrievalEngine:
    def __init__(self, artifact_dir=None, config=None):
        directory = Path(artifact_dir or ROOT / "artifacts/serial_cpu")
        self.metadata = read(directory / "metadata.json")
        self.config = config or self.metadata["config"]
        if fingerprint(read(ROOT / "data/manifests/model.json")) != self.metadata["model_fingerprint"]:
            raise ValueError("Model revision differs from index")
        for name, digest in self.metadata["sha256"].items():
            if checksum(directory / name) != digest:
                raise ValueError(f"Artifact checksum mismatch: {name}")
        self.documents = read(directory / "documents.json")
        embeddings = np.load(directory / "document_embeddings.npy", allow_pickle=False)
        validate_embeddings(embeddings)
        if len(embeddings) != len(self.documents):
            raise ValueError("Document mapping length mismatch")
        if self.config["max_length"] != self.metadata["config"]["max_length"]:
            raise ValueError("Tokenizer truncation differs from document embeddings")
        self.embedder = Embedder(self.config)
        self.index = ExactIndex(embeddings, self.config["backend"])

    def retrieve(self, claim: str, top_k: int = 10) -> list[dict]:
        if not isinstance(claim, str) or not claim.strip() or type(top_k) is not int or top_k < 1:
            raise ValueError("claim must be nonempty and top_k a positive integer")
        scores, indices = self.index.search(self.embedder.encode([claim]), top_k)
        return [{**{key: self.documents[i][key] for key in ("doc_id", "title", "abstract", "abstract_sentences")},
                 "retrieval_score": float(score), "rank": rank}
                for rank, (i, score) in enumerate(zip(indices[0], scores[0]), 1)]
