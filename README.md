# Scientific evidence retrieval — Member A

Exact semantic **document retrieval** for SciFact using `sentence-transformers/all-MiniLM-L6-v2`. Member B owns sentence extraction, reranking and classification. This project does not implement those stages.

Use PowerShell in `C:\Users\Kushi KKS\MAP` and the existing CUDA-enabled environment:

```powershell
.\.venv-rag\Scripts\python.exe retrieval.py hardware
.\.venv-rag\Scripts\python.exe retrieval.py prepare
.\.venv-rag\Scripts\python.exe retrieval.py download-model
.\.venv-rag\Scripts\python.exe -m pytest -q
.\.venv-rag\Scripts\python.exe retrieval.py benchmark --config configs/serial_cpu.yaml
.\.venv-rag\Scripts\python.exe scripts/run_experiments.py --skip-serial
.\.venv-rag\Scripts\python.exe retrieval.py query --input examples/query.json --output examples/retrieval_output.json
```

Dependencies are in `requirements.txt`; install only into `.venv-rag`. They intentionally exclude torch. Review `pip install --dry-run --report dependency-plan.json -r requirements.txt` before installation and stop if torch would change. `requirements-lock.txt` captures the actual tested environment; it is a provenance record including the existing CUDA torch, not a command to reinstall it.

Each benchmark defaults to one first-use pipeline plus **five full rebuilt warm trials**, with all 5,183 documents and 300 dev claims. This takes time on the serial CPU. A quick real-data smoke test is:

```powershell
.\.venv-rag\Scripts\python.exe retrieval.py benchmark --config configs/serial_cpu.yaml --limit 64 --queries 12 --trials 1 --latency-queries 3 --name smoke_serial
```

CPU tuning: `--threads 2 --workers 2 --batch-size 32` with `configs/parallel_cpu.yaml`. CUDA tuning: `--batch-size 16|32|64 --loader-workers 0|4` with `configs/parallel_gpu.yaml`. `--backend torch_cuda_exact` selects the optional CUDA matrix search. Default CUDA mode uses **CUDA embeddings + CPU FAISS**. CPU FAISS is never labeled GPU FAISS.

`--queries N` selects the first N dev claims deterministically; `--limit N` selects a corpus subset (smoke only). `--repeat-documents N` is an explicitly repeated-document computational workload and disables quality evaluation and API artifact saving. No synthetic or repeated records enter full-corpus quality metrics.

Outputs: dataset/model manifests under `data/manifests`; vectors, ID mappings and indexes under `artifacts/<run>`; per-trial timings and latencies under `results/raw/<run>`; CSV/JSON summaries under `results/summaries`; PNG/PDF plots under `results/plots`. `retrieval.py summarize` and `retrieval.py plots` regenerate reports from measurements. Large caches are gitignored.

Diagnostic profiles are under `results/profiles`. `benchmark --profile` saves whole-command and warmed embedding profiles. `benchmark-search --threads 1` (or `4`) compares both exact backends on identical saved serial vectors. `scripts/finalize_experiments.py` runs these checks, final tests, profiles and labeled scaling workloads without repeating the primary CPU matrix. Reusing a run name replaces its prior results.

The initial measurements were collected through Codex's command sandbox. Repeat the supplied experiments in ordinary PowerShell, with randomized/interleaved configuration order and controlled power/thermal conditions, before drawing general native-host performance conclusions. See the methodology for measurement definitions and limitations.

See [methodology](docs/METHODOLOGY.md), [experiment protocol](docs/EXPERIMENTS.md), [reproducibility](docs/REPRODUCIBILITY.md), and [Member B integration](docs/MEMBER_B_INTEGRATION.md). Actual run outcomes are recorded in `docs/RESULTS.md` after execution.
