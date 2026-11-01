"""Resume remaining verification/diagnostics without repeating primary CPU trials."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print("Running: " + " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    query = subprocess.run([sys.executable, "retrieval.py", "query", "--input", "examples/query.json",
                            "--output", "examples/retrieval_output.json"], cwd=ROOT, check=True,
                           capture_output=True, text=True, encoding="utf-8")
    console = json.loads(query.stdout)
    saved = json.loads((ROOT / "examples/retrieval_output.json").read_text(encoding="utf-8"))
    assert console == saved and len(saved) == 3
    assert any(ord(c) > 127 for row in saved for c in row["abstract"])
    (ROOT / "results/query-cli-validation.json").write_text(json.dumps({
        "passed": True, "stdout_file_round_trip": True, "unicode_preserved": True,
        "doc_ids": [r["doc_id"] for r in saved]}, indent=2), encoding="utf-8")
    print("Query CLI Unicode round-trip passed", flush=True)
    for threads in (1, 4):
        run("retrieval.py", "benchmark-search", "--threads", str(threads), "--trials", "5")
    run("-m", "pytest", "-q", "--junitxml=results/tests-final.xml")
    run("retrieval.py", "benchmark", "--config", "configs/serial_cpu.yaml", "--limit", "64",
        "--queries", "12", "--trials", "1", "--latency-queries", "3", "--name", "profile_serial", "--profile")
    run("retrieval.py", "benchmark", "--config", "configs/parallel_gpu.yaml", "--loader-workers", "0",
        "--limit", "128", "--queries", "12", "--trials", "1", "--latency-queries", "3", "--name", "profile_gpu", "--profile")
    run("retrieval.py", "benchmark", "--config", "configs/parallel_gpu.yaml", "--loader-workers", "0",
        "--limit", "512", "--name", "scale_unique512")
    run("retrieval.py", "benchmark", "--config", "configs/parallel_gpu.yaml", "--loader-workers", "0",
        "--repeat-documents", "2", "--name", "scale_repeat2")
    for command in ("prepare", "download-model"):
        run("retrieval.py", command)
    run("-m", "pip", "check")
    for command in ("summarize", "plots", "report"):
        run("retrieval.py", command)


if __name__ == "__main__":
    main()
