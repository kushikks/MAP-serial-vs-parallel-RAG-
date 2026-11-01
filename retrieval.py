"""CLI entry point and lazy Member B API. Run --help for reproducible commands."""
import argparse
import json
from pathlib import Path
from src.common import ROOT, read, dump
from src.runtime import environment

_engine = None
_thread_limits = None


def retrieve(claim: str, top_k: int = 10) -> list[dict]:
    global _engine, _thread_limits
    if not isinstance(claim, str) or not claim.strip() or type(top_k) is not int or top_k < 1:
        raise ValueError("claim must be nonempty and top_k a positive integer")
    if _engine is None:
        environment(1)
        from src.runtime import configure
        _thread_limits, _ = configure(1)
        from src.retrieval_engine import RetrievalEngine
        _engine = RetrievalEngine()
    return _engine.retrieve(claim, top_k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Download/checksum/validate official SciFact")
    sub.add_parser("download-model", help="Pin and cache all-MiniLM-L6-v2 weights")
    sub.add_parser("hardware")
    sub.add_parser("summarize")
    sub.add_parser("plots")
    sub.add_parser("report")
    search = sub.add_parser("benchmark-search", help="Isolate CPU/CUDA exact search on identical saved serial vectors")
    search.add_argument("--threads", type=int, default=1)
    search.add_argument("--trials", type=int, default=5)
    for command in ("benchmark", "query"):
        p = sub.add_parser(command)
        p.add_argument("--config", default="configs/serial_cpu.yaml")
        p.add_argument("--threads", type=int)
        p.add_argument("--workers", type=int)
        p.add_argument("--batch-size", type=int)
        p.add_argument("--loader-workers", type=int)
        p.add_argument("--backend", choices=["faiss_cpu", "torch_cuda_exact"])
        if command == "benchmark":
            p.add_argument("--profile", action="store_true", help="Save Python cProfile stats; use a separate named diagnostic run")
            p.add_argument("--trials", type=int, default=5)
            p.add_argument("--limit", type=int)
            p.add_argument("--queries", type=int)
            p.add_argument("--repeat-documents", type=int, default=1)
            p.add_argument("--latency-queries", type=int, default=30)
            p.add_argument("--name")
        else:
            p.add_argument("--claim")
            p.add_argument("--input", type=Path, help='JSON {"claim": "...", "top_k": 10}')
            p.add_argument("--top-k", type=int, default=10)
            p.add_argument("--artifact-dir", default="artifacts/serial_cpu")
            p.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = None
    if hasattr(args, "config"):
        import yaml
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        for key in ("threads", "workers", "batch_size", "loader_workers", "backend"):
            value = getattr(args, key)
            if value is not None:
                config[key] = value
        if config["device"] not in ("cpu", "cuda"):
            parser.error("device must be cpu or cuda")
        if config["backend"] not in ("faiss_cpu", "torch_cuda_exact"):
            parser.error("Unsupported search backend")
        if config["backend"] == "torch_cuda_exact" and config["device"] != "cuda":
            parser.error("CUDA search requires a CUDA configuration; use benchmark-search for isolated backend comparisons")
        for key in ("threads", "workers", "batch_size", "max_length"):
            if config[key] < 1:
                parser.error(f"{key} must be positive")
        if config["loader_workers"] < 0:
            parser.error("loader_workers cannot be negative")
        if config["name"] == "serial_cpu" and (config["device"] != "cpu" or config["backend"] != "faiss_cpu" or config["threads"] != 1 or config["batch_size"] != 1 or config["workers"] != 1 or config["loader_workers"] != 0):
            parser.error("Serial baseline configuration cannot use parallel settings")
    environment(config["threads"] if config else getattr(args, "threads", 1))
    if args.command == "prepare":
        from src.data_loader import prepare
        result = prepare()
    elif args.command == "download-model":
        from src.embeddings import acquire_model
        result = acquire_model()
    elif args.command == "hardware":
        from src.hardware_info import hardware
        result = hardware()
        dump(ROOT / "results/hardware.json", result)
    elif args.command == "benchmark":
        from src.benchmarking import benchmark
        if args.profile:
            import cProfile
            import pstats
            profile = cProfile.Profile()
            result = profile.runcall(benchmark, config, args.trials, args.limit, args.queries, args.repeat_documents, args.latency_queries, args.name)
            destination = ROOT / "results/profiles"
            destination.mkdir(parents=True, exist_ok=True)
            profile.dump_stats(str(destination / f"{args.name or config['name']}.prof"))
            with (destination / f"{args.name or config['name']}.txt").open("w", encoding="utf-8") as stream:
                pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(50)
            from src.profiling import profile_embedding
            result["steady_embedding_profile"] = profile_embedding(config, args.name or config["name"], min(args.limit or 128, 128))
        else:
            result = benchmark(config, args.trials, args.limit, args.queries, args.repeat_documents, args.latency_queries, args.name)
    elif args.command == "query":
        from src.runtime import configure
        limits, info = configure(config["threads"])
        from src.retrieval_engine import RetrievalEngine
        engine = RetrievalEngine(args.artifact_dir, config)
        request = read(args.input) if args.input else {"claim": args.claim, "top_k": args.top_k}
        result = engine.retrieve(request["claim"], request.get("top_k", 10))
        if args.output:
            dump(args.output, result)
    elif args.command == "summarize":
        from src.benchmarking import summarize
        result = summarize()
    elif args.command == "benchmark-search":
        from src.benchmarking import benchmark_search
        result = benchmark_search(args.threads, args.trials)
    elif args.command == "report":
        from src.reporting import report
        result = report()
    else:
        from src.plotting import plots
        result = plots()
    # JSON escapes preserve Unicode semantics on legacy Windows stdout codecs.
    # Files written by dump() remain readable UTF-8.
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
