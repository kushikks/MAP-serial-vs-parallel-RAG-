# Executed results

Generated from actual local measurements; no estimated values.

Hardware: 13th Gen Intel(R) Core(TM) i9-13900H; NVIDIA GeForce RTX 4050 Laptop GPU; Windows-10-10.0.26200-SP0; torch 2.11.0+cu128.

Dataset validation: 5,183 corpus documents; 300 dev claims, of which 188 have evidence annotations and 112 are excluded from quality averages. No test-set evaluation.

## Full-corpus warm trials

| Configuration | Trials | Doc embedding mean ± SD (s) | Pipeline mean ± SD (s) | Embedding speedup | Pipeline speedup | Single query mean (ms) | Effective loader workers |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu_t2 | 5 | 629.994 ± 74.197 | 636.905 ± 74.961 | 1.549× | 1.549× | 26.223 | 0 |
| cpu_t4 | 5 | 311.715 ± 62.125 | 315.735 ± 62.133 | 3.131× | 3.125× | 21.569 | 0 |
| gpu_b16_w0 | 5 | 24.407 ± 1.193 | 25.949 ± 1.398 | 39.989× | 38.025× | 27.772 | 0 |
| gpu_b32_w0 | 5 | 25.436 ± 0.806 | 26.933 ± 0.776 | 38.370× | 36.635× | 27.030 | 0 |
| gpu_b64_w0 | 5 | 27.472 ± 2.812 | 28.756 ± 2.871 | 35.527× | 34.313× | 24.765 | 0 |
| gpu_b64_w4 | 5 | 65.729 ± 0.982 | 120.287 ± 1.816 | 14.849× | 8.203× | 24.218 | 3 |
| gpu_exact_b64_w0 | 5 | 28.976 ± 2.606 | 29.642 ± 2.576 | 33.683× | 33.287× | 20.914 | 0 |
| serial_cpu | 5 | 976.004 ± 65.258 | 986.697 ± 66.642 | 1.000× | 1.000× | 35.336 | 0 |

Times are full-corpus rebuilt warm measurements. Cold startup and per-stage timings are saved separately. CUDA speedups are heterogeneous. CPU embedding efficiency divides by native compute threads, not preprocessing workers.

## Retrieval effectiveness

| Configuration | Recall@1 | Recall@5 | Recall@10 | MRR (full corpus) | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| cpu_t2 | 0.612589 | 0.868528 | 0.911702 | 0.735378 | 0.774275 |
| cpu_t4 | 0.612589 | 0.868528 | 0.911702 | 0.735378 | 0.774275 |
| gpu_b16_w0 | 0.612589 | 0.868528 | 0.911702 | 0.735378 | 0.774275 |
| gpu_b32_w0 | 0.612589 | 0.868528 | 0.911702 | 0.735378 | 0.774275 |
| gpu_b64_w0 | 0.612589 | 0.868528 | 0.911702 | 0.735378 | 0.774275 |
| gpu_b64_w4 | 0.612589 | 0.868528 | 0.911702 | 0.735378 | 0.774275 |
| gpu_exact_b64_w0 | 0.612589 | 0.868528 | 0.911702 | 0.735378 | 0.774275 |
| serial_cpu | 0.612589 | 0.868528 | 0.911702 | 0.735378 | 0.774275 |

## Interpretation and optimization opportunities

Document embedding accounted for 98.92% of serial rebuilt-pipeline time, making offline embedding the dominant cost. Exact flat index construction took only milliseconds.

The lowest observed mean full-pipeline time was `gpu_b16_w0`: 25.949 s (38.03× heterogeneous speedup). This is the best observed configuration in a fixed-order laptop session, not proof of a universally optimal batch size.

At the same requested batch size 64 and CPU-search backend, the three-effective-worker pipeline took 4.18× as long as the zero-worker pipeline. Multi-item claim embedding with workers was especially expensive. For this workload/context, select `--loader-workers 0`; a reusable tokenizer worker pool is a future optimization for larger workloads.

Mean single-query latency was 21.57 ms for four-thread CPU and 20.91 ms for CUDA exact search. These close point estimates do not establish a significant latency difference. Large offline/batched speedups must not be described as equivalent interactive-latency gains.

Keep precomputed embeddings and indexes for deployment, batch claims when throughput matters, and investigate the documented full-ranking/tie-sort overhead before drawing conclusions about optimized native top-k search. Preserve model weights, float32 precision, truncation and quality checks during optimization.

## Isolated exact search on identical saved vectors

Both backends search all 300 saved serial claim vectors against the same 5,183 saved serial document vectors. Transfers and deterministic full-ranking sorting are included. These are implementation-specific results, not a GPU-FAISS measurement.

| Query batch | CPU FAISS, 4 threads (ms) | PyTorch CUDA exact (ms) | Search-time ratio |
|---:|---:|---:|---:|
| 1 | 1059.144 | 316.536 | 3.35× |
| 32 | 762.037 | 11.377 | 66.98× |
| 300 | 758.392 | 3.101 | 244.58× |

## Scalability diagnostics

| Workload | Computational inputs | Warm trials | Embedding mean (s) | Quality policy |
|---|---:|---:|---:|---|
| Real unique subset | 512 | 5 | 2.113 | Subset only; not full-corpus quality |
| Real full corpus | 5183 | 5 | 27.472 | Full dev effectiveness reported above |
| Repeated corpus ×2 | 10366 | 5 | 51.226 | Disabled; duplicates are not new evidence |

## Executed validation

Final pytest: 27 passed, 0 failures, 0 errors, 0 skipped. Report: `results/tests-final.xml`.
The actual query CLI Unicode round-trip is recorded in `results/query-cli-validation.json`. Whole-command and warmed embedding profiles are stored under `results/profiles`; profiled subsets are excluded from the full-corpus comparison table.

## Files and handoff

- Raw trials, cold timings, latency samples, quality and hardware metadata: `results/raw/<run>/`.
- Mean, SD, median and Student-t 95% CIs: `results/summaries/summary.json` and `.csv`.
- Figures: `results/plots/*.png` and `*.pdf`.
- Real fixed Member B output: `examples/retrieval_output.json`; import `retrieve` from `retrieval.py`.
- Dataset/model provenance: `data/manifests/`; saved index/vectors/IDs: `artifacts/<run>/`.

## Limitations

The initial collection used Codex's workspace-write command runner; native PowerShell replication has not been performed. Runner/process-launch effects were not isolated from Windows effects. Single-laptop measurements, incomplete relevance annotations, 256-token document truncation, sampled RAM, fixed configuration order, and SciFact's small size limit generalization. Native Windows FAISS here is CPU only. CUDA search uses PyTorch, not FAISS GPU. Worker reductions and OOM changes appear in metadata. No WSL, second environment, SciFact-Open download, or Member B classifier is included. See METHODOLOGY.md for exact protocols.
