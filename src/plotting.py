import os
from .common import ROOT, read
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/matplotlib-cache"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plots():
    summary = read(ROOT / "results/summaries/summary.json")
    # Smoke/subset and repeated-document workloads get separate summaries; main
    # comparative figures only use the full unique SciFact corpus.
    runs = {n: r for n, r in summary.items() if r["workload"]["documents"] == 5183 and r["workload_kind"] == "real_unique_documents"}
    preferred = ["serial_cpu", "cpu_t2", "cpu_t4", "gpu_b64_w4", "gpu_b16_w0", "gpu_b32_w0", "gpu_b64_w0", "gpu_exact_b64_w0"]
    runs = {n: runs[n] for n in preferred + sorted(set(runs) - set(preferred)) if n in runs}
    if not runs:
        raise ValueError("No full-corpus measured results available for research plots")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42})
    out = ROOT / "results/plots"
    out.mkdir(parents=True, exist_ok=True)
    hardware = read(ROOT / "results/hardware.json")
    subtitle = f"SciFact: 5,183 documents | {hardware['cpu']} | {hardware.get('gpu', 'CPU')}"
    produced = []

    def display(name):
        r = runs[name]
        c = r["config"]
        if c["device"] == "cpu":
            return "Serial CPU" if name == "serial_cpu" else f"CPU ({c['threads']} threads)"
        effective = r.get("effective_loader_workers", c["loader_workers"])
        workers = f"w{effective}"
        if effective != c["loader_workers"]:
            workers += f" (req {c['loader_workers']})"
        search = "CPU search" if c["backend"] == "faiss_cpu" else "CUDA search"
        return f"CUDA b{c['batch_size']}, {search}, {workers}"

    def save(fig, name):
        fig.text(0.02, 0.015, subtitle, fontsize=6)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        for ext in ("png", "pdf"):
            path = out / f"{name}.{ext}"
            fig.savefig(path, bbox_inches="tight")
            produced.append(str(path))
        plt.close(fig)

    def bar(metric, label, name, factor=1):
        fig, ax = plt.subplots(figsize=(8, 4.3))
        names = list(runs)
        labels = [display(n) for n in names]
        means = [runs[n]["statistics"][metric]["mean"] * factor for n in names]
        errs = [(runs[n]["statistics"][metric]["ci95_high"] - runs[n]["statistics"][metric]["mean"]) * factor if runs[n]["statistics"][metric]["ci95_high"] is not None else 0 for n in names]
        ax.bar(labels, means, yerr=errs, capsize=3, color=["#2874a6" if runs[n]["config"]["device"] == "cpu" else "#ca6f1e" for n in names])
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.grid(axis="y", alpha=0.2)
        save(fig, name)

    for metric, label, name, factor in (
        ("document_embedding_s", "Document embedding wall time (s), 95% CI", "execution_time", 1),
        ("single_latency_mean_s", "Mean single-query latency (ms), 95% CI", "gpu_cpu_latency", 1000),
        ("index_build_s", "Exact index construction time (ms), 95% CI", "index_build", 1000),
        ("batched_queries_per_s", "Batched online throughput (queries/s), 95% CI", "query_throughput", 1),
        ("peak_process_tree_rss_bytes", "Sampled peak process-tree RSS (GiB), 95% CI", "memory_ram", 1 / 2**30),
        ("gpu_peak_allocated_bytes", "Peak PyTorch CUDA allocated memory (GiB), 95% CI", "memory_gpu", 1 / 2**30)):
        bar(metric, label, name, factor)
    cpu = sorted([(n, r) for n, r in runs.items() if r["config"]["device"] == "cpu" and "speedups" in r], key=lambda x: x[1]["config"]["threads"])
    for efficiency in (False, True):
        fig, ax = plt.subplots(figsize=(6, 4))
        x = [r["config"]["threads"] for _, r in cpu]
        y = [r["cpu_embedding_efficiency"] if efficiency else r["speedups"]["document_embedding_s"] for _, r in cpu]
        ax.plot(x, y, "o-", color="#2874a6")
        ax.set(xlabel="CPU compute threads (PyTorch / FAISS)", ylabel="Embedding efficiency (speedup / threads)" if efficiency else "Embedding speedup relative to serial CPU")
        ax.set_xticks(x)
        ax.grid(alpha=0.2)
        save(fig, "cpu_efficiency" if efficiency else "cpu_speedup")
    fig, ax = plt.subplots(figsize=(6, 4))
    for backend in ("faiss_cpu", "torch_cuda_exact"):
        gpu = sorted([r for r in runs.values() if r["config"]["device"] == "cuda" and r["config"]["backend"] == backend and r["config"]["loader_workers"] == 0], key=lambda r: r["config"]["batch_size"])
        if gpu:
            ax.errorbar([r["config"]["batch_size"] for r in gpu], [r["statistics"]["document_throughput_per_s"]["mean"] for r in gpu],
                        yerr=[r["statistics"]["document_throughput_per_s"]["std"] or 0 for r in gpu], marker="o", capsize=3, label=backend)
    ax.set(xlabel="CUDA embedding batch size", ylabel="Document embedding throughput (documents/s), ±SD")
    ax.set_xticks([16, 32, 64])
    ax.legend()
    save(fig, "embedding_throughput")
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, r in runs.items():
        metrics = r["quality"].get("metrics")
        if metrics:
            ax.plot([1, 5, 10], [metrics[f"recall@{k}"] for k in (1, 5, 10)], marker="o", label=display(name))
    ax.set(xlabel="Retrieved documents (k)", ylabel="Macro Recall@k (annotated evidence claims)", ylim=(0, 1))
    ax.set_xticks([1, 5, 10])
    ax.legend(fontsize=6)
    save(fig, "recall")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(runs)
    labels = [display(n) for n in names]
    bottom = np.zeros(len(labels))
    for key, label in (("dataset_loading_s", "Data loading"), ("preprocessing_s", "Preprocessing"),
                       ("document_embedding_s", "Document embedding"), ("index_build_s", "Index build"),
                       ("claim_embedding_s", "Claim embedding"), ("search_s", "Search")):
        y = np.array([runs[n]["statistics"][key]["mean"] for n in names])
        ax.bar(labels, y, bottom=bottom, label=label)
        bottom += y
    ax.set_ylabel("Full rebuilt pipeline wall time (s)")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.legend(fontsize=6)
    save(fig, "latency_breakdown")
    search_path = ROOT / "results/summaries/search_comparison_t4.json"
    if search_path.exists():
        data = read(search_path)
        fig, ax = plt.subplots(figsize=(6, 4))
        batches = [1, 32, 300]
        for backend, label in (("faiss_cpu", "CPU FAISS (4 threads)"), ("torch_cuda_exact", "PyTorch CUDA exact")):
            stats = [data["summary"][f"{backend}_batch{b}"] for b in batches]
            means = [s["mean"] * 1000 for s in stats]
            errs = [(s["ci95_high"] - s["mean"]) * 1000 for s in stats]
            ax.errorbar(range(3), means, yerr=errs, marker="o", capsize=3, label=label)
        ax.set_xticks(range(3), batches)
        ax.set_yscale("log")
        ax.set(xlabel="Query batch size (identical saved vectors)", ylabel="Search time for 300 claims (ms, log scale), 95% CI")
        ax.legend(fontsize=7)
        save(fig, "exact_search_comparison")
    scale_names = [n for n in ("scale_unique512", "gpu_b64_w0", "scale_repeat2") if n in summary]
    if len(scale_names) == 3:
        fig, ax = plt.subplots(figsize=(6, 4))
        stats = [summary[n]["statistics"]["document_embedding_s"] for n in scale_names]
        ax.bar(["512 unique subset", "5,183 unique corpus", "10,366 inputs (repeat ×2)"],
               [s["mean"] for s in stats], yerr=[s["ci95_high"] - s["mean"] for s in stats], capsize=3,
               color=["#2874a6", "#2874a6", "#ca6f1e"])
        ax.set_ylabel("CUDA document embedding time (s), 95% CI")
        ax.tick_params(axis="x", rotation=15, labelsize=7)
        ax.set_title("Batch 64, zero loader workers; repeats are compute-only", fontsize=8)
        save(fig, "scalability_compute")
    return produced
