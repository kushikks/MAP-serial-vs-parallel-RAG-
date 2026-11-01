import csv
import datetime
import gc
import time
from pathlib import Path
import faiss  # Consistent Windows native-library load order.
import numpy as np
import torch
from .common import ROOT, dump, read, checksum, fingerprint
from .data_loader import load_raw, dev_claims
from .preprocessing import preprocess
from .embeddings import Embedder
from .indexing import ExactIndex, parity
from .evaluation import evaluate
from .hardware_info import hardware, MemoryMonitor
from .runtime import configure


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def timed(function, device):
    sync(device)
    start = time.perf_counter()
    value = function()
    sync(device)
    return value, time.perf_counter() - start


def pipeline(config, embedder, limit=None, queries=None, repeat=1):
    stages = {}
    started = time.perf_counter()
    rows, stages["dataset_loading_s"] = timed(load_raw, config["device"])
    rows = rows[:limit]
    docs, stages["preprocessing_s"] = timed(lambda: preprocess(rows, config["workers"]), config["device"])
    cs, stages["claim_loading_s"] = timed(lambda: dev_claims(queries), config["device"])
    texts = [d["text"] for d in docs] * repeat
    vectors, stages["document_embedding_s"] = timed(lambda: embedder.encode(texts), config["device"])
    search_chunk = 1 if config["name"] == "serial_cpu" else 1024
    index, stages["index_build_s"] = timed(lambda: ExactIndex(vectors, config["backend"], chunk_size=search_chunk), config["device"])
    qv, stages["claim_embedding_s"] = timed(lambda: embedder.encode([c["claim"] for c in cs]), config["device"])
    (scores, indices), stages["search_s"] = timed(lambda: index.search(qv, 10), config["device"])
    stages["end_to_end_s"] = time.perf_counter() - started
    return docs, cs, vectors, qv, index, scores, indices, stages


def quality(docs, cs, vectors, qv, index, repeat):
    if repeat != 1:
        return {"disabled": "Repeated documents are a computational workload, not new evidence"}
    scores, indices = index.search(qv, len(vectors))
    result = evaluate(cs, [[docs[i]["doc_id"] for i in ranking] for ranking in indices])
    result["scope"] = "full SciFact corpus" if len(docs) == 5183 else "subset smoke test, not full-corpus quality"
    cpu = ExactIndex(vectors, "faiss_cpu")
    rs, ri = cpu.search(qv, 10)
    ss, si = index.search(qv, 10)
    result["backend_parity"] = parity(rs, ri, ss, si, vectors, qv)
    baseline_path = ROOT / "artifacts/serial_cpu"
    if (baseline_path / "document_embeddings.npy").exists():
        ref_docs = read(baseline_path / "documents.json")
        ref_meta = read(baseline_path / "metadata.json")
        if [d["doc_id"] for d in ref_docs] == [d["doc_id"] for d in docs] and ref_meta["model_fingerprint"] == fingerprint(read(ROOT / "data/manifests/model.json")):
            ref = np.load(baseline_path / "document_embeddings.npy", allow_pickle=False)
            difference = float(np.max(np.abs(ref - vectors)))
            if difference > 1e-4:
                raise AssertionError(f"Serial/parallel document embedding difference: {difference}")
            ref_q = np.load(baseline_path / "claim_embeddings.npy", allow_pickle=False)
            ref_cs = read(baseline_path / "claims.json")
            if [c["id"] for c in ref_cs] == [c["id"] for c in cs]:
                query_difference = float(np.max(np.abs(ref_q - qv)))
                if query_difference > 1e-4:
                    raise AssertionError("Serial/parallel claim embeddings differ")
                ref_index = ExactIndex(ref)
                bs, bi = ref_index.search(ref_q, 10)
                result["serial_parallel_parity"] = parity(bs, bi, ss, si, vectors, qv, atol=1e-4)
                result["serial_parallel_parity"].update(max_document_abs_diff=difference, max_claim_abs_diff=query_difference)
    return result


def save_artifacts(directory, config, docs, cs, vectors, qv):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "document_embeddings.npy", vectors, allow_pickle=False)
    np.save(directory / "claim_embeddings.npy", qv, allow_pickle=False)
    dump(directory / "documents.json", docs)
    dump(directory / "claims.json", cs)
    ExactIndex(vectors).save(directory / "index.faiss")
    dump(directory / "metadata.json", {"config": config, "model_fingerprint": fingerprint(read(ROOT / "data/manifests/model.json")),
        "dataset_fingerprint": fingerprint(read(ROOT / "data/manifests/dataset.json")),
        "document_ids": [d["doc_id"] for d in docs],
        "sha256": {n: checksum(directory / n) for n in ("document_embeddings.npy", "claim_embeddings.npy", "documents.json", "claims.json", "index.faiss")}})


def benchmark(config, trials=5, limit=None, queries=None, repeat=1, latency_queries=30, run_name=None):
    if trials < 1 or (limit is not None and limit < 1) or (queries is not None and queries < 1) or repeat < 1 or latency_queries < 1:
        raise ValueError("Workload sizes and trials must be positive")
    if config["device"] == "cpu" and config["batch_size"] == 1 and config["threads"] == 1:
        assert config["workers"] == 1 and config["loader_workers"] == 0
    limits, thread_info = configure(config["threads"])
    name = run_name or config["name"]
    out = ROOT / "results/raw" / name
    out.mkdir(parents=True, exist_ok=True)
    meta = {"config": config, "hardware_before": hardware(), "thread_limits": thread_info,
            "dataset": read(ROOT / "data/manifests/dataset.json"), "model": read(ROOT / "data/manifests/model.json"),
            "trials": trials, "limit": limit, "queries": queries, "document_repeat": repeat,
            "workload_kind": "real_unique_documents" if repeat == 1 else "repeated_document_compute_only",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "cold_definition": "First model load + first pipeline in fresh configuration process; OS disk cache not flushed",
            "warm_definition": "Resident model, repeat full document embedding and index build; no reused embeddings",
            "single_query_definition": "Sequential requests with zero DataLoader workers, same device/model; includes tokenization and transfers",
            "ram_definition": "20ms sampled process-tree RSS; shared pages may be counted more than once",
            "faiss_backend": "CPU IndexFlatIP", "search_backend": config["backend"]}
    meta["source_sha256"] = {str(p.relative_to(ROOT)): checksum(p) for p in [ROOT / "retrieval.py", *sorted((ROOT / "src").glob("*.py")), *sorted((ROOT / "configs").glob("*.yaml"))]}
    meta.update(status="incomplete", completed_trials=0)
    dump(out / "metadata.json", meta)
    records = []
    with MemoryMonitor() as monitor:
        if config["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()
        embedder, model_load = timed(lambda: Embedder(config), config["device"])
        # Imports during model initialization can load additional native pools.
        extra_limits, final_threads = configure(config["threads"])
        meta["thread_limits_after_model_load"] = final_threads
        cold = pipeline(config, embedder, limit, queries, repeat)
        docs, cs, vectors, qv, index, scores, indices, stages = cold
        stages = dict(stages, model_loading_s=model_load, cold_total_s=model_load + stages["end_to_end_s"])
        dump(out / "cold.json", stages)
        # Correctness gate BEFORE warm timings.
        effectiveness = quality(docs, cs, vectors, qv, index, repeat)
        dump(out / "quality.json", effectiveness)
        if repeat == 1:
            save_artifacts(ROOT / "artifacts" / name, config, docs, cs, vectors, qv)
        embedder.encode([d["text"] for d in docs[:min(32, len(docs))]])
        index.search(qv[:min(8, len(qv))], 10)
        for trial in range(trials):
            gc.collect()
            if config["device"] == "cuda":
                torch.cuda.reset_peak_memory_stats()
            monitor.peak = 0
            docs, cs, vectors, qv, index, scores, indices, stages = pipeline(config, embedder, limit, queries, repeat)
            row = {"trial": trial + 1, **stages, "documents": len(vectors), "queries": len(cs),
                   "document_throughput_per_s": len(vectors) / stages["document_embedding_s"],
                   "online_batched_total_s": stages["claim_embedding_s"] + stages["search_s"],
                   "index_bytes": int(vectors.nbytes + 45),
                   "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if config["device"] == "cuda" else 0,
                   "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if config["device"] == "cuda" else 0}
            row["batched_queries_per_s"] = len(cs) / row["online_batched_total_s"]
            row["batched_amortized_latency_s"] = row["online_batched_total_s"] / len(cs)
            # Measure actual single-query service latency separately from batch amortization.
            original_workers = embedder.config["loader_workers"]
            embedder.config["loader_workers"] = 0
            samples = []
            try:
                embedder.encode([cs[0]["claim"]])  # online path warm-up
                for claim in cs[:latency_queries]:
                    q, embedding_time = timed(lambda: embedder.encode([claim["claim"]]), config["device"])
                    _, search_time = timed(lambda: index.search(q, 10), config["device"])
                    samples.append({"claim_id": claim["id"], "embedding_s": embedding_time,
                                    "search_s": search_time, "total_s": embedding_time + search_time})
            finally:
                embedder.config["loader_workers"] = original_workers
            latencies = [s["total_s"] for s in samples]
            row.update(single_latency_mean_s=float(np.mean(latencies)), single_latency_median_s=float(np.median(latencies)),
                       single_latency_p95_s=float(np.percentile(latencies, 95)), single_queries_per_s=1 / float(np.mean(latencies)),
                       peak_process_tree_rss_bytes=monitor.peak, effective_batch_size=embedder.batch_size)
            dump(out / f"latency_{trial + 1}.json", samples)
            records.append(row)
            dump(out / "trials.json", records)
            meta.update(completed_trials=len(records), oom_events=embedder.oom_events,
                        effective_loader_workers=embedder.config["loader_workers"],
                        loader_worker_adjustments=embedder.worker_adjustments)
            dump(out / "metadata.json", meta)
            print(f"{name}: trial {trial + 1}/{trials}, document embedding {stages['document_embedding_s']:.3f}s, pipeline {stages['end_to_end_s']:.3f}s", flush=True)
    meta.update(status="complete", hardware_after=hardware(), oom_events=embedder.oom_events,
                loader_worker_adjustments=embedder.worker_adjustments,
                effective_loader_workers=embedder.config["loader_workers"],
                search_effective_chunk_size=index.chunk_size, search_oom_events=index.oom_events, cold=cold[-1])
    dump(out / "metadata.json", meta)
    summarize()
    return {"run": name, "quality": effectiveness.get("metrics"), "trials": len(records)}


def summary_stats(values):
    from scipy.stats import t
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else None
    half = float(t.ppf(0.975, len(a) - 1) * sd / np.sqrt(len(a))) if sd is not None else None
    return {"n": len(a), "mean": mean, "std": sd, "median": float(np.median(a)),
            "ci95_low": mean - half if half is not None else None, "ci95_high": mean + half if half is not None else None}


def summarize():
    summaries = {}
    for path in sorted((ROOT / "results/raw").glob("*/trials.json")):
        rows = read(path)
        meta_path = path.parent / "metadata.json"
        if not rows or not meta_path.exists():
            continue
        meta = read(meta_path)
        if meta.get("status", "complete") != "complete":
            continue
        summary = {"config": meta["config"], "workload": {k: rows[0][k] for k in ("documents", "queries")},
                   "effective_loader_workers": meta.get("effective_loader_workers"),
                   "workload_kind": meta["workload_kind"], "quality": read(path.parent / "quality.json"),
                   "statistics": {k: summary_stats([r[k] for r in rows]) for k in rows[0] if k != "trial"}}
        summaries[path.parent.name] = summary
    base = summaries.get("serial_cpu")
    if base:
        for name, item in summaries.items():
            if item["workload"] != base["workload"] or item["workload_kind"] != base["workload_kind"]:
                continue
            item["speedups"] = {k: base["statistics"][k]["mean"] / item["statistics"][k]["mean"]
                                for k in ("document_embedding_s", "end_to_end_s", "online_batched_total_s", "search_s")}
            item["speedup_type"] = "heterogeneous" if item["config"]["device"] == "cuda" else "CPU"
            if item["config"]["device"] == "cpu":
                # Torch/FAISS compute threads, not Python preprocessing workers.
                item["cpu_embedding_efficiency"] = item["speedups"]["document_embedding_s"] / item["config"]["threads"]
    dump(ROOT / "results/summaries/summary.json", summaries)
    flat = []
    for name, item in summaries.items():
        for metric, stat in item["statistics"].items():
            flat.append({"configuration": name, "metric": metric, **stat})
    if flat:
        with (ROOT / "results/summaries/summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    return summaries


def benchmark_search(threads=1, trials=5):
    """Isolate search using identical saved serial vectors on both backends."""
    limits, info = configure(threads)
    directory = ROOT / "artifacts/serial_cpu"
    vectors = np.load(directory / "document_embeddings.npy", allow_pickle=False)
    queries = np.load(directory / "claim_embeddings.npy", allow_pickle=False)
    reference = ExactIndex(vectors)
    rs, ri = reference.search(queries, 10)
    rows = []
    for backend, device in (("faiss_cpu", "cpu"), ("torch_cuda_exact", "cuda")):
        index, build_s = timed(lambda: ExactIndex(vectors, backend), device)
        scores, indices = index.search(queries, 10)
        check = parity(rs, ri, scores, indices, vectors, queries)
        for batch_size in (1, 32, len(queries)):
            def search_all():
                for start in range(0, len(queries), batch_size):
                    index.search(queries[start:start + batch_size], 10)
            search_all()
            for trial in range(trials):
                _, seconds = timed(search_all, device)
                rows.append({"backend": backend, "threads": threads, "batch_size": batch_size,
                             "trial": trial + 1, "queries": len(queries), "search_s": seconds,
                             "queries_per_s": len(queries) / seconds, "amortized_latency_s": seconds / len(queries),
                             "index_build_s": build_s, "parity": check})
    result = {"hardware": hardware(), "thread_limits": info, "rows": rows,
              "definition": "Same serial float32 document and claim vectors; transfer and deterministic sorting included",
              "summary": {f"{backend}_batch{b}": summary_stats([r["search_s"] for r in rows if r["backend"] == backend and r["batch_size"] == b])
                          for backend in ("faiss_cpu", "torch_cuda_exact") for b in (1, 32, len(queries))}}
    dump(ROOT / f"results/summaries/search_comparison_t{threads}.json", result)
    return result["summary"]
