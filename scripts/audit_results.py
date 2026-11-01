"""Audit saved experimental outputs without repeating inference or benchmarks."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    names = ["serial_cpu", "cpu_t2", "cpu_t4", "gpu_b64_w4", "gpu_b16_w0", "gpu_b32_w0", "gpu_b64_w0", "gpu_exact_b64_w0"]
    summary = read(ROOT / "results/summaries/summary.json")
    baseline = summary["serial_cpu"]["quality"]["metrics"]
    model_fingerprints, dataset_fingerprints = set(), set()
    for name in names:
        run = ROOT / "results/raw" / name
        rows, meta = read(run / "trials.json"), read(run / "metadata.json")
        assert len(rows) == 5 and meta.get("status", "complete") == "complete", name
        assert all(r["documents"] == 5183 and r["queries"] == 300 for r in rows), name
        assert summary[name]["quality"]["metrics"] == baseline, name
        assert summary[name]["quality"]["eligible_claims"] == 188, name
        assert summary[name]["quality"]["excluded_no_evidence"] == 112, name
        assert summary[name]["quality"]["backend_parity"]["passed"], name
        if name != "serial_cpu":
            assert summary[name]["quality"]["serial_parallel_parity"]["passed"], name
        artifact = ROOT / "artifacts" / name
        artifact_meta = read(artifact / "metadata.json")
        for file, digest in artifact_meta["sha256"].items():
            assert sha(artifact / file) == digest, (name, file)
        model_fingerprints.add(artifact_meta["model_fingerprint"])
        dataset_fingerprints.add(artifact_meta["dataset_fingerprint"])
    assert len(model_fingerprints) == len(dataset_fingerprints) == 1
    for kind, directory in (("dataset", ROOT / "data/raw"), ("model", ROOT / "artifacts/model")):
        for file, digest in read(ROOT / f"data/manifests/{kind}.json")["sha256"].items():
            assert sha(directory / file) == digest, (kind, file)
    for threads in (1, 4):
        search = read(ROOT / f"results/summaries/search_comparison_t{threads}.json")
        assert len(search["rows"]) == 30 and all(r["parity"]["passed"] for r in search["rows"])
    for name in ("profile_serial", "profile_gpu"):
        profile = read(ROOT / f"results/profiles/{name}_embedding.json")
        assert profile["diagnostic_only"] and profile["embedding_dimension"] == 384
        assert abs(profile["norm_min"] - 1) < 2e-5 and abs(profile["norm_max"] - 1) < 2e-5
        assert (ROOT / f"results/profiles/{name}.prof").stat().st_size > 0
    repeated = read(ROOT / "results/raw/scale_repeat2/quality.json")
    assert "disabled" in repeated and "metrics" not in repeated
    assert summary["scale_repeat2"]["workload"]["documents"] == 10366
    assert summary["scale_repeat2"]["workload_kind"] == "repeated_document_compute_only"
    assert summary["scale_unique512"]["workload"]["documents"] == 512
    assert not (ROOT / "artifacts/scale_repeat2").exists()
    suites = list(ET.parse(ROOT / "results/tests-final.xml").getroot().iter("testsuite"))
    tests = sum(int(s.get("tests", 0)) for s in suites)
    assert all(int(s.get(k, 0)) == 0 for s in suites for k in ("errors", "failures", "skipped"))
    assert read(ROOT / "results/query-cli-validation.json")["passed"]
    assert (ROOT / "docs/RESULTS.md").exists()
    pngs = list((ROOT / "results/plots").glob("*.png"))
    assert len(pngs) >= 13 and all(p.with_suffix(".pdf").exists() for p in pngs)
    result = {"passed": True, "primary_configurations": len(names), "primary_warm_trials": 40,
              "scalability_warm_trials": 10, "pytest_passed": tests, "figures_png_pdf_pairs": len(pngs),
              "shared_model_and_dataset_fingerprints": True, "all_saved_artifact_and_cache_checksums_verified": True,
              "all_primary_effectiveness_metrics_identical": True, "isolated_search_parity_rows": 60,
              "diagnostic_profiles": 2, "repeated_workload_excluded_from_quality": True}
    (ROOT / "results/final-audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
