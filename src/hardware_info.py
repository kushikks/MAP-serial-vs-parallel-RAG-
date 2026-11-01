import importlib.metadata
import platform
import subprocess
import sys
import psutil
import faiss
import torch


def hardware():
    cpu_model = platform.processor()
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu_model = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    packages = {}
    for name in ("torch", "sentence-transformers", "transformers", "faiss-cpu", "numpy", "psutil", "scipy", "tokenizers", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        smi = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,temperature.gpu,utilization.gpu,power.draw", "--format=csv"], text=True, timeout=15).strip()
    except Exception as exc:
        smi = str(exc)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    ac = psutil.sensors_battery()
    result = {"os": platform.platform(), "cpu": cpu_model, "physical_cores": psutil.cpu_count(logical=False),
              "logical_cores": psutil.cpu_count(), "ram_total_bytes": psutil.virtual_memory().total,
              "ram_available_bytes": psutil.virtual_memory().available, "python": sys.version,
              "python_executable": sys.executable, "packages": packages, "cuda_runtime": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "nvidia_smi": smi,
              "ac_power": ac.power_plugged if ac else None, "git_commit": commit}
    import faiss
    result["faiss_gpu_api_available"] = hasattr(faiss, "StandardGpuResources")
    result["faiss_compile_options"] = str(faiss.get_compile_options())
    if torch.cuda.is_available():
        x = torch.arange(1000, device="cuda")
        result.update(gpu=torch.cuda.get_device_name(), vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                      cuda_computation_sum=int(x.sum().item()))
        assert result["cuda_computation_sum"] == 499500
    else:
        result["cuda_diagnostic"] = "CUDA not available. CPU modes work; see docs/REPRODUCIBILITY.md for Colab fallback."
    return result


class MemoryMonitor:
    """Sample process tree RSS, including Windows DataLoader children (20 ms)."""
    def __enter__(self):
        import threading
        self.stop = threading.Event()
        self.peak = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.stop.is_set():
            try:
                process = psutil.Process()
                total = process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except psutil.Error:
                        pass
                self.peak = max(self.peak, total)
            except psutil.Error:
                pass
            self.stop.wait(0.02)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
