"""Run sequentially to prevent competing workloads. No downloads or installs."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--skip-serial", action="store_true")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    configs = [
        ("serial_cpu", ["--config", "configs/serial_cpu.yaml"]),
        ("cpu_t2", ["--config", "configs/parallel_cpu.yaml", "--threads", "2", "--workers", "2"]),
        ("cpu_t4", ["--config", "configs/parallel_cpu.yaml", "--threads", "4", "--workers", "4"]),
        ("gpu_b64_w4", ["--config", "configs/parallel_gpu.yaml"]),
        ("gpu_b16_w0", ["--config", "configs/parallel_gpu.yaml", "--batch-size", "16", "--loader-workers", "0"]),
        ("gpu_b32_w0", ["--config", "configs/parallel_gpu.yaml", "--batch-size", "32", "--loader-workers", "0"]),
        ("gpu_b64_w0", ["--config", "configs/parallel_gpu.yaml", "--loader-workers", "0"]),
        ("gpu_exact_b64_w0", ["--config", "configs/parallel_gpu.yaml", "--loader-workers", "0", "--backend", "torch_cuda_exact"]),
    ]
    for name, options in configs:
        if args.skip_serial and name == "serial_cpu" or args.only is not None and name not in args.only:
            continue
        command = [sys.executable, "-u", "retrieval.py", "benchmark", *options, "--name", name, "--trials", str(args.trials)]
        subprocess.run(command, cwd=ROOT, check=True)
    subprocess.run([sys.executable, "retrieval.py", "plots"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "retrieval.py", "query", "--input", "examples/query.json", "--output", "examples/retrieval_output.json"], cwd=ROOT, check=True)
    for threads in (1, 4):
        subprocess.run([sys.executable, "retrieval.py", "benchmark-search", "--threads", str(threads), "--trials", str(args.trials)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "--junitxml=results/tests-final.xml"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "retrieval.py", "report"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
