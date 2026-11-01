"""Pinned HF schema inspection and its official source archive, without remote code execution."""
import datetime
import json
import tarfile
import urllib.request
from pathlib import Path
from .common import ROOT, checksum, dump, read
from .preprocessing import preprocess, claims

HF = "https://huggingface.co"
DATASET = "allenai/scifact"
ARCHIVE = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DATA_FILES = ["corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"]


def extract_files(archive, raw, names):
    with tarfile.open(archive) as tar:
        for name in names:
            member = tar.getmember("data/" + name)
            if not member.isfile() or member.size > 100_000_000:
                raise ValueError("Unexpected archive member")
            with tar.extractfile(member) as f:
                (raw / name).write_bytes(f.read())


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".tmp")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as f:
            while chunk := response.read(1024 * 1024):
                f.write(chunk)
        temporary.replace(target)
    return target


def prepare():
    raw = ROOT / "data/raw"
    manifest_path = ROOT / "data/manifests/dataset.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        for name in ("scifact.py", "dataset_infos.json", "README.md"):
            download(f"{HF}/datasets/{DATASET}/resolve/{manifest['hf_revision']}/{name}", raw / name)
        archive = download(manifest["source"], raw / "data.tar.gz")
        if checksum(archive) != manifest["sha256"]["data.tar.gz"]:
            raise ValueError("Official latest archive changed since this manifest. Restore the archived cache; refusing to substitute a new dataset version.")
        missing = [name for name in DATA_FILES if not (raw / name).exists()]
        if missing:
            extract_files(archive, raw, missing)
        for name, digest in manifest["sha256"].items():
            # HF API metadata includes mutable download/like counts. The original
            # API snapshot is archival provenance, not required dataset content;
            # never manufacture a replacement that claims the original checksum.
            if name == "hf_repository.json" and not (raw / name).exists():
                continue
            if checksum(raw / name) != digest:
                raise ValueError(f"Cached dataset checksum mismatch: {name}")
    else:
        api = download(f"{HF}/api/datasets/{DATASET}", raw / "hf_repository.json")
        revision = read(api)["sha"]
        for name in ("scifact.py", "dataset_infos.json", "README.md"):
            download(f"{HF}/datasets/{DATASET}/resolve/{revision}/{name}", raw / name)
        # HF is a legacy Python dataset script, not Parquet. Its _URL is the same
        # official archive. Read JSONL directly rather than executing remote code.
        script = (raw / "scifact.py").read_text(encoding="utf-8")
        if ARCHIVE not in script or 'name="corpus"' not in script or 'name="claims"' not in script:
            raise ValueError("SciFact upstream schema/source changed: inspect before loading")
        archive = download(ARCHIVE, raw / "data.tar.gz")
        expected = DATA_FILES
        extract_files(archive, raw, expected)
        names = expected + ["data.tar.gz", "scifact.py", "dataset_infos.json", "hf_repository.json", "README.md"]
        manifest = {"dataset": DATASET, "hf_revision": revision,
                    "hf_config_version": "1.0.0", "source": ARCHIVE,
                    "download_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "loader": "official JSONL source used by inspected HF legacy script",
                    "configurations": {"corpus": ["train"], "claims": ["train", "validation", "test"]},
                    "sha256": {name: checksum(raw / name) for name in names}}
        dump(manifest_path, manifest)
    try:
        rows = load_raw()
        docs = preprocess(rows)
        ids = {d["doc_id"] for d in docs}
        if len(ids) != len(docs):
            raise ValueError("Duplicate corpus IDs")
        splits = {s: claims(load_jsonl(raw / f"claims_{s}.jsonl"), s) for s in ("train", "dev", "test")}
        split_ids = {s: {c["id"] for c in cs} for s, cs in splits.items()}
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
            if split_ids[a] & split_ids[b]:
                raise ValueError(f"Claim ID leakage between {a}/{b}")
        for split, cs in splits.items():
            for c in cs:
                if set(c["relevant_doc_ids"]) - ids:
                    raise ValueError(f"Evidence document absent: {c['id']}")
            dump(ROOT / f"data/processed/claims_{split}.json", cs)
        # Validate rationale sentence indices without using them for retrieval.
        counts = {d["doc_id"]: len(d["abstract_sentences"]) for d in docs}
        for s in ("train", "dev"):
            for c in load_jsonl(raw / f"claims_{s}.jsonl"):
                for did, anns in c["evidence"].items():
                    for ann in anns:
                        if any(type(i) is not int or not 0 <= i < counts[did] for i in ann["sentences"]):
                            raise ValueError(f"Invalid rationale sentence index: {c['id']}")
        dump(ROOT / "data/processed/documents.json", docs)
        report = {"valid": True, "documents": len(docs), "splits": {
            s: {"claims": len(cs), "with_evidence": sum(bool(c["relevant_doc_ids"]) for c in cs),
                "without_evidence": sum(not c["relevant_doc_ids"] for c in cs)} for s, cs in splits.items()},
            "structured_abstracts": sum(bool(r.get("structured")) for r in rows),
            "sentence_count": sum(counts.values()), "raw_schema": {"corpus": list(rows[0]),
                **{s: list(load_jsonl(raw / f"claims_{s}.jsonl")[0]) for s in splits}},
            "missing_archival_api_snapshot": not (raw / "hf_repository.json").exists(),
            "test_annotation_status": "Unreleased/unknown; evidence counts describe available annotations, not confirmed absence",
            "policy": "Fail explicitly on malformed records; no silent dropping. Test labels never evaluated."}
        dump(ROOT / "data/manifests/validation.json", report)
        return report
    except Exception as exc:
        dump(ROOT / "data/manifests/validation.json", {"valid": False, "error": str(exc)})
        raise


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"Empty JSONL line: {path}:{lineno}")
            rows.append(json.loads(line))
    return rows


def load_raw():
    return load_jsonl(ROOT / "data/raw/corpus.jsonl")


def dev_claims(limit=None):
    return read(ROOT / "data/processed/claims_dev.json")[:limit]
