import subprocess
import sys
from src.common import ROOT


def test_serial_rejects_cuda_backend_before_loading_model():
    result = subprocess.run([sys.executable, str(ROOT / "retrieval.py"), "benchmark",
                             "--config", str(ROOT / "configs/serial_cpu.yaml"),
                             "--backend", "torch_cuda_exact"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "CUDA search requires a CUDA configuration" in result.stderr


def test_serial_rejects_parallel_threads():
    result = subprocess.run([sys.executable, str(ROOT / "retrieval.py"), "benchmark",
                             "--config", str(ROOT / "configs/serial_cpu.yaml"), "--threads", "2"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "Serial baseline configuration cannot use parallel settings" in result.stderr
