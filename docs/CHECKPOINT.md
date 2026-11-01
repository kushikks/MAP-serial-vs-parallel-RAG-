# Paused checkpoint — 2026-11-01

## Current state: completed

The user subsequently said **continue** and all remaining work completed. The finalization process exited successfully. No benchmark job remains running.

- CLI encoding fixed; actual Unicode query round-trip passed.
- Both isolated exact-search comparisons completed (60 parity-checked timed rows).
- Final pytest: **27 passed, zero failures/errors/skips**.
- CPU/GPU whole-command and warmed embedding profiles executed and saved.
- Five warm scalability trials each for 512 unique documents and 10,366 repeated computational inputs completed. Repeated-input quality evaluation is disabled.
- Dataset/model caches revalidated; pip reported no broken requirements. Existing CUDA torch remains intact.
- Final figures reviewed; **13 PNG/PDF pairs** exported with a workspace-local Matplotlib cache.
- `docs/RESULTS.md`, summary/quality/speedup/search CSVs and JSON results generated.
- `results/final-audit.json` passed: all primary results, model/data fingerprints, cache/artifact checksums, profiles and validation records verified.

The historical pause record below explains earlier state and recovery. Its unfinished-step list is no longer pending. Do not rerun the completed primary CPU matrix unless intentionally starting a new experiment session. Native PowerShell replication and other research limitations remain documented in `METHODOLOGY.md` and `RESULTS.md`.

## Completed and saved

- Existing `.venv-rag` retained; CUDA torch remains `2.11.0+cu128`. Actual RTX 4050 CUDA tensor computation verified (sum 499500).
- SciFact downloaded from the official archive identified in the inspected Hugging Face loader. Corpus: 5,183 documents. Claims: 809 train / 300 dev / 300 unlabeled test. Dev quality uses 188 annotated claims; 112 without evidence annotations are excluded.
- Model `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; 384-dimensional normalized float32 embeddings, max length 256, mean pooling.
- Complete Member A implementation in `retrieval.py`, `src/`, configs, tests and documentation. No Member B classifier/reranker/extractor implemented.
- **All 40 primary warm trials completed**: five each for `serial_cpu`, `cpu_t2`, `cpu_t4`, `gpu_b64_w4`, `gpu_b16_w0`, `gpu_b32_w0`, `gpu_b64_w0`, `gpu_exact_b64_w0`. Each configuration also has a first-use/cold pipeline measurement and warmup. Do not rerun the long CPU experiments.
- All eight configurations matched retrieval effectiveness: Recall@1 0.6125886525, Recall@5 0.8685283688, Recall@10 0.9117021277, full MRR 0.7353778843, MRR@10 0.7313724249, nDCG@10 0.7742752652. Full-corpus serial/parallel and backend parity gates passed before warm timings.
- `gpu_b64_w4` requested four DataLoader workers; RAM preflight reduced this to **three**, with the reason and available RAM recorded. Batch size stayed 64; no OOM event occurred in that configuration. Other GPU sweep cases use zero workers.
- CPU FAISS `IndexFlatIP` and separately labeled PyTorch CUDA exact search implemented. No GPU FAISS, WSL, new environment, global pip operations or driver changes.
- Raw trial/cold/quality/latency/metadata files: `results/raw/<run>/`. Primary CSV/JSON summaries: `results/summaries/summary.csv` and `summary.json`. Full mappings, vectors and indexes: `artifacts/<run>/` (gitignored).
- Eleven figure types exported as PNG and PDF (22 files) under `results/plots/`. They have **not yet been visually inspected**.
- Actual Member B fixture: `examples/query.json`, `examples/retrieval_output.json`, `examples/provenance.json`. Query is a real dev claim: “0-dimensional biomaterials show inductive properties.” The fixture is actual exact retrieval, not fabricated evidence.
- Earlier preflight: **20 tests passed, 2 skipped** before subsequent additional tests/code changes. **Final full test suite has NOT run**; do not claim it passed. A syntax check parsed all 26 Python files that existed at that point; additional files were added later.
- Serial source archived before later worker/cache/reporting refactors: `results/source_snapshots/serial_source.zip`. Per-run source hashes are in metadata. Numerical model, tokenizer, pooling, normalization and retrieval semantics remained fixed; later edits mostly concern startup, validation, diagnostics and persistence.

## Exact stop condition

The sequential orchestration job (exec session 6471) has **exited with code 1**; no intended experiment job remains running. It completed every primary benchmark and the plot command, then failed during the query CLI's final `print(json.dumps(... ensure_ascii=False ...))`:

`UnicodeEncodeError: 'charmap' codec can't encode character '\u03b2'` on the Windows CP1252 stdout stream.

The query's `--output examples/retrieval_output.json` UTF-8 write happened **before** that print and succeeded. The runner stopped before isolated search comparisons, final pytest and report generation.

A separate nonfatal Matplotlib warning occurred: it could not write font cache under `C:\Users\Kushi KKS\.matplotlib`. Figure files were exported. On resume, set `MPLCONFIGDIR` to a directory inside the workspace before importing Matplotlib, avoiding writes outside allowed roots.

## Resume work (only after explicit “continue”)

1. Fix portable CLI JSON stdout (ASCII-escaped JSON is one safe option; retain UTF-8 in output files). Re-run the actual query CLI and verify successful JSON round-trip, including Greek text. Do not reinstall torch.
2. Configure workspace-local Matplotlib cache, inspect representative/all exported PNGs with image tools, and repair/regenerate layouts if needed.
3. Run `retrieval.py benchmark-search --threads 1 --trials 5`, then `--threads 4 --trials 5`. These isolate CPU FAISS versus CUDA exact search using the **same saved serial vectors**. They have not run yet because orchestration aborted first.
4. Run final pytest using `.venv-rag\Scripts\python.exe -m pytest -q --junitxml=results/tests-final.xml`. Added tests cover cache restoration/pinning, serial CLI mode guards and avoiding DataLoader process startup for single queries. Fix failures and recheck appropriately.
5. Execute separate named **diagnostic** profiles with the existing `benchmark --profile` flag, e.g. serial CPU 64 documents / 12 claims / one trial / three latency queries, and CUDA zero-worker 128 documents. The flag now saves both whole-command cProfile and a warmed embedding-only cProfile via `src/profiling.py`; this addition still needs execution verification. These subset profile runs are not paper performance comparisons.
6. Run labeled scalability checks with zero-worker CUDA: a real-unique subset (e.g. 512 documents) and `--repeat-documents 2` (10,366 computational document inputs). Keep five warm trials for scalability runs; repeated-document quality evaluation and artifact saving are deliberately disabled. Do not treat duplicates as independent evidence. These checks have not run yet.
7. Revalidate local caches (`prepare`, `download-model` checksum paths), run pip compatibility check, regenerate summaries, plots and `retrieval.py report`. No `docs/RESULTS.md` exists yet. The report helper also exports `quality.csv` and `speedups.csv`.
8. Write interpretation of actual measurements, bottlenecks and limitations; inspect final outputs and provide the user's requested completion summary, quality metrics, measured speedups, files, tests, figures, Member B instructions and unresolved limitations.

## Important implementation / research details

- Windows FAISS DLL initialization intermittently failed when FAISS loaded after torch. CLI/modules/tests now import FAISS first. Tokenizer collators/seeding live in `src/tokenization.py`, deliberately without FAISS imports, so spawned tokenization workers do not load FAISS after torch.
- Serial mode is CPU / batch 1 / one native inference, BLAS and FAISS thread / zero DataLoader workers / sequential preprocessing and **one query per FAISS call**. Parallel CPU batches queries. Native pool limits are checked, including after model import. OpenMP stub libraries may report one under a higher cap and are logged, not mistaken for active parallel pools.
- Single-input encoding always uses zero DataLoader workers, matching actual online-latency measurements. Configured workers apply to multi-item workloads.
- CPU embedding efficiency divides speedup by native compute threads (two/four); GPU speedups are heterogeneous, with no CPU efficiency assigned.
- CPU primary mean rebuilt pipeline: serial **986.6965 s**, two threads **636.9048 s (1.5492×)**, four threads **315.7346 s (3.1251×)**. CPU embedding efficiencies: **0.77461** and **0.78277**.
- Query latency samples are the first 30 dev claims per trial by default; batch-amortized latency is kept separate. Cold totals include initial Sentence Transformers import/model initialization, but exclude earlier core library imports/downloads.
- Search computes full rankings to make tie cutoff deterministic (score descending, corpus index ascending). This adds sorting overhead and is explicitly documented; do not claim optimized native FAISS top-k performance.
- All initial measurements ran through Codex's workspace-write command sandbox. The runner's process-launch/security effects were not isolated from Windows effects. `docs/METHODOLOGY.md` explicitly requires ordinary PowerShell replication before general native-host Windows conclusions. No native-host replication has been performed.
- Fixed configuration order, laptop thermal/power variation, incomplete relevance labels, 256-token truncation, small SciFact size and sampled process-tree RSS limit generalization. Do not fabricate clocks, speedups, native-host results, profile traces or unexecuted tests.
- Native Windows FAISS GPU prebuilt support was checked in the official installation documentation; current installed FAISS exposes no GPU resources. PyTorch CUDA exact search is the supported optional GPU-search path here. SciFact-Open was not downloaded; no multi-gigabyte download or WSL change is authorized without asking.

## Useful commands (run from project root with the existing interpreter)

```powershell
.\.venv-rag\Scripts\python.exe retrieval.py query --input examples/query.json --output examples/retrieval_output.json
.\.venv-rag\Scripts\python.exe retrieval.py benchmark-search --threads 1 --trials 5
.\.venv-rag\Scripts\python.exe retrieval.py benchmark-search --threads 4 --trials 5
.\.venv-rag\Scripts\python.exe -m pytest -q --junitxml=results/tests-final.xml
.\.venv-rag\Scripts\python.exe retrieval.py benchmark --config configs/serial_cpu.yaml --limit 64 --queries 12 --trials 1 --latency-queries 3 --name profile_serial --profile
.\.venv-rag\Scripts\python.exe retrieval.py benchmark --config configs/parallel_gpu.yaml --loader-workers 0 --limit 128 --queries 12 --trials 1 --latency-queries 3 --name profile_gpu --profile
.\.venv-rag\Scripts\python.exe retrieval.py benchmark --config configs/parallel_gpu.yaml --loader-workers 0 --limit 512 --name scale_unique512
.\.venv-rag\Scripts\python.exe retrieval.py benchmark --config configs/parallel_gpu.yaml --loader-workers 0 --repeat-documents 2 --name scale_repeat2
.\.venv-rag\Scripts\python.exe retrieval.py summarize
.\.venv-rag\Scripts\python.exe retrieval.py plots
.\.venv-rag\Scripts\python.exe retrieval.py report
```

Avoid re-running `scripts/run_experiments.py` wholesale: that would unnecessarily repeat hours of completed CPU measurements. Resume the unfinished steps explicitly.
