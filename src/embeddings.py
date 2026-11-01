from pathlib import Path
import faiss  # Load native FAISS libraries before torch on Windows.
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import ROOT, dump, read, checksum
from .tokenization import Tokenize, seed_worker

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def acquire_model():
    from huggingface_hub import HfApi, snapshot_download
    manifest = ROOT / "data/manifests/model.json"
    target = ROOT / "artifacts/model"
    if manifest.exists():
        meta = read(manifest)
        missing = [name for name in meta["sha256"] if not (target / name).exists()]
        if missing:
            snapshot_download(MODEL, revision=meta["revision"], local_dir=str(target), allow_patterns=missing)
        for name, digest in meta["sha256"].items():
            if checksum(target / name) != digest:
                raise ValueError(f"Model checksum mismatch: {name}")
        return meta
    revision = HfApi().model_info(MODEL).sha
    snapshot_download(MODEL, revision=revision, local_dir=str(target),
                      allow_patterns=["*.json", "*.txt", "*.safetensors"],
                      ignore_patterns=["onnx/*", "openvino/*"])
    files = sorted(p for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts)
    meta = {"model": MODEL, "revision": revision, "dimension": 384, "max_length": 256,
            "pooling": "attention-mask mean pooling from original SentenceTransformer modules",
            "dtype": "float32", "normalized": True,
            "sha256": {str(p.relative_to(target)).replace("\\", "/"): checksum(p) for p in files}}
    dump(manifest, meta)
    return meta


def validate_embeddings(vectors, dimension=384):
    if vectors.ndim != 2 or vectors.shape[1] != dimension or vectors.dtype != np.float32:
        raise ValueError(f"Invalid embedding shape/dtype: {vectors.shape}/{vectors.dtype}")
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5):
        raise ValueError("Embeddings must be finite and L2 normalized")


class Embedder:
    def __init__(self, config):
        from sentence_transformers import SentenceTransformer
        self.config = config.copy()
        self.device = config["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Run `retrieval.py hardware` for diagnostics; no CPU fallback.")
        if not (ROOT / "data/manifests/model.json").exists():
            raise FileNotFoundError("Run `retrieval.py download-model` first")
        self.model = SentenceTransformer(str(ROOT / "artifacts/model"), device=self.device, local_files_only=True)
        self.model.max_seq_length = config["max_length"]
        self.model.eval()
        self.batch_size = config["batch_size"]
        self.oom_events = []
        self.worker_adjustments = []
        if self.device == "cuda" and self.config["loader_workers"]:
            import psutil
            available = psutil.virtual_memory().available
            # Windows spawn imports torch in every worker. Reserve 1 GiB for
            # the parent/OS and conservatively allow 768 MiB per worker.
            safe_workers = max(0, int((available - 2**30) // (768 * 2**20)))
            if self.config["loader_workers"] > safe_workers:
                self.worker_adjustments.append({"requested": self.config["loader_workers"],
                    "effective": safe_workers, "available_ram_bytes": available,
                    "reason": "Windows spawn memory budget: 1 GiB reserve + 768 MiB per worker"})
                self.config["loader_workers"] = safe_workers
        if self.model.get_embedding_dimension() != 384:
            raise ValueError("Unexpected model embedding dimension")
        pooling = [m for m in self.model.modules() if m.__class__.__name__ == "Pooling"]
        if len(pooling) != 1 or pooling[0].pooling_mode != "mean" or pooling[0].pooling_output_dimension != 384:
            raise ValueError("Unexpected pooling configuration")

    def encode(self, texts):
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("encode requires nonempty strings")
        while True:
            try:
                return self._encode(texts)
            except torch.cuda.OutOfMemoryError:
                if self.device != "cuda" or self.batch_size == 1:
                    raise
                old = self.batch_size
                self.batch_size = max(1, old // 2)
                self.oom_events.append({"old_batch_size": old, "new_batch_size": self.batch_size,
                                        "policy": "restart entire encode to retain order"})
                torch.cuda.empty_cache()

    def _encode(self, texts):
        loader = DataLoader(texts, batch_size=self.batch_size, shuffle=False,
                            num_workers=0 if len(texts) == 1 else self.config["loader_workers"],
                            collate_fn=Tokenize(self.model.tokenizer, self.config["max_length"]),
                            pin_memory=self.device == "cuda", worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(42))
        result = []
        completed = 0
        next_progress = 256
        with torch.inference_mode():
            for features in loader:
                features = {k: v.to(self.device, non_blocking=self.device == "cuda") for k, v in features.items()}
                output = self.model(features)["sentence_embedding"]
                output = torch.nn.functional.normalize(output.float(), p=2, dim=1)
                result.append(output.cpu().numpy())
                completed += len(output)
                if len(texts) >= 1024 and completed >= next_progress:
                    print(f"Embedding {completed}/{len(texts)} ({self.device}, batch {self.batch_size})", flush=True)
                    next_progress += 256
        vectors = np.ascontiguousarray(np.concatenate(result), dtype=np.float32)
        validate_embeddings(vectors)
        return vectors
