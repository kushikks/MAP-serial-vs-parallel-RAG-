"""Diagnostic profiles kept separate from primary benchmark statistics."""
import cProfile
import pstats
import time
import faiss  # Match the application's Windows native-library load order.
import numpy as np
import torch
from .common import ROOT, dump, read
from .data_loader import load_raw
from .preprocessing import preprocess
from .embeddings import Embedder


def profile_embedding(config, name, limit=128):
    docs = preprocess(load_raw()[:limit], config["workers"])
    texts = [d["text"] for d in docs]
    embedder = Embedder(config)
    embedder.encode(texts[:min(len(texts), config["batch_size"])])
    if config["device"] == "cuda":
        torch.cuda.synchronize()
    profile = cProfile.Profile()
    start = time.perf_counter()
    vectors = profile.runcall(embedder.encode, texts)
    if config["device"] == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    out = ROOT / "results/profiles"
    out.mkdir(parents=True, exist_ok=True)
    profile.dump_stats(str(out / f"{name}_embedding.prof"))
    with (out / f"{name}_embedding.txt").open("w", encoding="utf-8") as stream:
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(60)
    dump(out / f"{name}_embedding.json", {
        "diagnostic_only": True, "definition": "Resident-model document encode, after warmup; includes cProfile overhead",
        "limitation": "Python/call-level profiling; asynchronous CUDA kernel internals require a separate GPU profiler",
        "config": config, "effective_loader_workers": embedder.config["loader_workers"],
        "effective_batch_size": embedder.batch_size, "worker_adjustments": embedder.worker_adjustments,
        "oom_events": embedder.oom_events, "model_revision": read(ROOT / "data/manifests/model.json")["revision"],
        "document_ids": [d["doc_id"] for d in docs], "documents": len(docs), "profiled_wall_s": elapsed,
        "embedding_dimension": vectors.shape[1], "norm_min": float(np.linalg.norm(vectors, axis=1).min()),
        "norm_max": float(np.linalg.norm(vectors, axis=1).max())})
    return str(out / f"{name}_embedding.txt")
