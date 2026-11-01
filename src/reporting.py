import csv
from .common import ROOT, read


def report():
    summary = read(ROOT / "results/summaries/summary.json")
    hardware = read(ROOT / "results/hardware.json")
    lines = ["# Executed results", "", "Generated from actual local measurements; no estimated values.", "",
             f"Hardware: {hardware['cpu']}; {hardware.get('gpu', 'CUDA unavailable')}; {hardware['os']}; torch {hardware['packages']['torch']}.", "",
             "Dataset validation: 5,183 corpus documents; 300 dev claims, of which 188 have evidence annotations and 112 are excluded from quality averages. No test-set evaluation.", "",
             "## Full-corpus warm trials", "",
             "| Configuration | Trials | Doc embedding mean ± SD (s) | Pipeline mean ± SD (s) | Embedding speedup | Pipeline speedup | Single query mean (ms) | Effective loader workers |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in summary.items():
        if r["workload"]["documents"] != 5183 or r["workload_kind"] != "real_unique_documents":
            continue
        s, speed = r["statistics"], r.get("speedups", {})
        e = s["document_embedding_s"]
        pipeline = s["end_to_end_s"]
        lines.append(f"| {name} | {e['n']} | {e['mean']:.3f} ± {(e['std'] or 0):.3f} | {pipeline['mean']:.3f} ± {(pipeline['std'] or 0):.3f} | {speed.get('document_embedding_s', 0):.3f}× | {speed.get('end_to_end_s', 0):.3f}× | {s['single_latency_mean_s']['mean'] * 1000:.3f} | {r.get('effective_loader_workers')} |")
    lines.extend(["", "Times are full-corpus rebuilt warm measurements. Cold startup and per-stage timings are saved separately. CUDA speedups are heterogeneous. CPU embedding efficiency divides by native compute threads, not preprocessing workers.", "",
                  "## Retrieval effectiveness", "", "| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR (full corpus) | nDCG@10 |", "|---|---:|---:|---:|---:|---:|"])
    for name, r in summary.items():
        if r["workload"]["documents"] != 5183:
            continue
        m = r["quality"].get("metrics")
        if m:
            lines.append("| " + name + " | " + " | ".join(f"{m[k]:.6f}" for k in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")) + " |")
    if all(n in summary for n in ("serial_cpu", "cpu_t4", "gpu_b64_w4", "gpu_b64_w0", "gpu_exact_b64_w0")):
        base = summary["serial_cpu"]["statistics"]
        fraction = 100 * base["document_embedding_s"]["mean"] / base["end_to_end_s"]["mean"]
        worker_ratio = summary["gpu_b64_w4"]["statistics"]["end_to_end_s"]["mean"] / summary["gpu_b64_w0"]["statistics"]["end_to_end_s"]["mean"]
        candidates = [(n, r) for n, r in summary.items() if r["config"]["device"] == "cuda" and r["workload"] == summary["serial_cpu"]["workload"] and r["workload_kind"] == "real_unique_documents"]
        best_name, best = min(candidates, key=lambda item: item[1]["statistics"]["end_to_end_s"]["mean"])
        cpu_ms = summary["cpu_t4"]["statistics"]["single_latency_mean_s"]["mean"] * 1000
        gpu_ms = summary["gpu_exact_b64_w0"]["statistics"]["single_latency_mean_s"]["mean"] * 1000
        lines.extend(["", "## Interpretation and optimization opportunities", "",
            f"Document embedding accounted for {fraction:.2f}% of serial rebuilt-pipeline time, making offline embedding the dominant cost. Exact flat index construction took only milliseconds.", "",
            f"The lowest observed mean full-pipeline time was `{best_name}`: {best['statistics']['end_to_end_s']['mean']:.3f} s ({best['speedups']['end_to_end_s']:.2f}× heterogeneous speedup). This is the best observed configuration in a fixed-order laptop session, not proof of a universally optimal batch size.", "",
            f"At the same requested batch size 64 and CPU-search backend, the three-effective-worker pipeline took {worker_ratio:.2f}× as long as the zero-worker pipeline. Multi-item claim embedding with workers was especially expensive. For this workload/context, select `--loader-workers 0`; a reusable tokenizer worker pool is a future optimization for larger workloads.", "",
            f"Mean single-query latency was {cpu_ms:.2f} ms for four-thread CPU and {gpu_ms:.2f} ms for CUDA exact search. These close point estimates do not establish a significant latency difference. Large offline/batched speedups must not be described as equivalent interactive-latency gains.", "",
            "Keep precomputed embeddings and indexes for deployment, batch claims when throughput matters, and investigate the documented full-ranking/tie-sort overhead before drawing conclusions about optimized native top-k search. Preserve model weights, float32 precision, truncation and quality checks during optimization."])
    search_rows = []
    for threads in (1, 4):
        path = ROOT / f"results/summaries/search_comparison_t{threads}.json"
        if not path.exists():
            continue
        data = read(path)
        if threads == 4:
            lines.extend(["", "## Isolated exact search on identical saved vectors", "",
                          "Both backends search all 300 saved serial claim vectors against the same 5,183 saved serial document vectors. Transfers and deterministic full-ranking sorting are included. These are implementation-specific results, not a GPU-FAISS measurement.", "",
                          "| Query batch | CPU FAISS, 4 threads (ms) | PyTorch CUDA exact (ms) | Search-time ratio |",
                          "|---:|---:|---:|---:|"])
            for batch in (1, 32, 300):
                cpu = data["summary"][f"faiss_cpu_batch{batch}"]["mean"]
                gpu = data["summary"][f"torch_cuda_exact_batch{batch}"]["mean"]
                lines.append(f"| {batch} | {cpu * 1000:.3f} | {gpu * 1000:.3f} | {cpu / gpu:.2f}× |")
        search_rows.extend([{**{k: v for k, v in row.items() if k != "parity"}, "parity_passed": row["parity"]["passed"]} for row in data["rows"]])
    if all(n in summary for n in ("scale_unique512", "scale_repeat2")):
        lines.extend(["", "## Scalability diagnostics", "", "| Workload | Computational inputs | Warm trials | Embedding mean (s) | Quality policy |", "|---|---:|---:|---:|---|"])
        for name, label, policy in (("scale_unique512", "Real unique subset", "Subset only; not full-corpus quality"), ("gpu_b64_w0", "Real full corpus", "Full dev effectiveness reported above"), ("scale_repeat2", "Repeated corpus ×2", "Disabled; duplicates are not new evidence")):
            r = summary[name]
            lines.append(f"| {label} | {r['workload']['documents']} | {r['statistics']['document_embedding_s']['n']} | {r['statistics']['document_embedding_s']['mean']:.3f} | {policy} |")
    test_path = ROOT / "results/tests-final.xml"
    if test_path.exists():
        import xml.etree.ElementTree as ET
        suites = list(ET.parse(test_path).getroot().iter("testsuite"))
        totals = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
        passed = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
        lines.extend(["", "## Executed validation", "", f"Final pytest: {passed} passed, {totals['failures']} failures, {totals['errors']} errors, {totals['skipped']} skipped. Report: `results/tests-final.xml`.",
                      "The actual query CLI Unicode round-trip is recorded in `results/query-cli-validation.json`. Whole-command and warmed embedding profiles are stored under `results/profiles`; profiled subsets are excluded from the full-corpus comparison table."])
    lines.extend(["", "## Files and handoff", "", "- Raw trials, cold timings, latency samples, quality and hardware metadata: `results/raw/<run>/`.",
                  "- Mean, SD, median and Student-t 95% CIs: `results/summaries/summary.json` and `.csv`.",
                  "- Figures: `results/plots/*.png` and `*.pdf`.",
                  "- Real fixed Member B output: `examples/retrieval_output.json`; import `retrieve` from `retrieval.py`.",
                  "- Dataset/model provenance: `data/manifests/`; saved index/vectors/IDs: `artifacts/<run>/`.", "",
                  "## Limitations", "", "The initial collection used Codex's workspace-write command runner; native PowerShell replication has not been performed. Runner/process-launch effects were not isolated from Windows effects. Single-laptop measurements, incomplete relevance annotations, 256-token document truncation, sampled RAM, fixed configuration order, and SciFact's small size limit generalization. Native Windows FAISS here is CPU only. CUDA search uses PyTorch, not FAISS GPU. Worker reductions and OOM changes appear in metadata. No WSL, second environment, SciFact-Open download, or Member B classifier is included. See METHODOLOGY.md for exact protocols."])
    target = ROOT / "docs/RESULTS.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    quality_rows, speed_rows = [], []
    for name, r in summary.items():
        if r["workload"]["documents"] != 5183 or r["workload_kind"] != "real_unique_documents":
            continue
        if r["quality"].get("metrics"):
            quality_rows.append({"configuration": name, "eligible_claims": r["quality"]["eligible_claims"],
                                 "excluded_no_evidence": r["quality"]["excluded_no_evidence"], **r["quality"]["metrics"]})
        if r.get("speedups"):
            speed_rows.append({"configuration": name, "comparison_type": r["speedup_type"],
                               **r["speedups"], "cpu_embedding_efficiency": r.get("cpu_embedding_efficiency")})
    for name, rows in (("quality.csv", quality_rows), ("speedups.csv", speed_rows), ("search_comparison.csv", search_rows)):
        if rows:
            with (ROOT / "results/summaries" / name).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    return str(target)
