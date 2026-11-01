"""Isolated synthetic fixtures; no network access or research data mutation."""
import hashlib
import io
import json
import tarfile
import pytest
from src.common import dump, checksum


def test_missing_model_cache_uses_recorded_revision(tmp_path, monkeypatch):
    from src import embeddings
    import huggingface_hub
    monkeypatch.setattr(embeddings, "ROOT", tmp_path)
    content = b"unit-test-only mocked weights, not a scientific model"
    meta = {"revision": "recorded-unit-revision", "sha256": {"model.safetensors": hashlib.sha256(content).hexdigest()}}
    dump(tmp_path / "data/manifests/model.json", meta)
    calls = []
    def fetch(repo_id, revision, local_dir, allow_patterns):
        calls.append(revision)
        target = tmp_path / "artifacts/model"
        target.mkdir(parents=True)
        (target / "model.safetensors").write_bytes(content)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fetch)
    assert embeddings.acquire_model() == meta
    assert calls == ["recorded-unit-revision"]
    (tmp_path / "artifacts/model/model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        embeddings.acquire_model()
    assert len(calls) == 1


def test_dataset_files_restore_from_checksummed_archive(tmp_path, monkeypatch):
    from src import data_loader
    monkeypatch.setattr(data_loader, "ROOT", tmp_path)
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    records = {"corpus.jsonl": [{"doc_id": "01", "title": "Synthetic unit fixture", "abstract": ["Synthetic sentence."], "structured": False}]}
    for split in ("train", "dev", "test"):
        record = {"id": split, "claim": f"Synthetic {split} unit claim."}
        if split != "test":
            record["evidence"] = {"01": [{"label": "SUPPORT", "sentences": [0]}]}
        records[f"claims_{split}.jsonl"] = [record]
    digests = {}
    with tarfile.open(raw / "data.tar.gz", "w:gz") as archive:
        for name, rows in records.items():
            content = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
            item = tarfile.TarInfo("data/" + name)
            item.size = len(content)
            archive.addfile(item, io.BytesIO(content))
            digests[name] = hashlib.sha256(content).hexdigest()
    digests["data.tar.gz"] = checksum(raw / "data.tar.gz")
    for name in ("scifact.py", "dataset_infos.json", "README.md"):
        (raw / name).write_text("Synthetic unit metadata only", encoding="utf-8")
        digests[name] = checksum(raw / name)
    digests["hf_repository.json"] = "historical-api-snapshot-not-present-in-fixture"
    dump(tmp_path / "data/manifests/dataset.json", {"hf_revision": "unit-only", "source": "https://example.invalid/unit-fixture", "sha256": digests})
    def no_network(*args, **kwargs):
        raise AssertionError("Cache restoration must not contact network for this fixture")
    monkeypatch.setattr(data_loader.urllib.request, "urlopen", no_network)
    report = data_loader.prepare()
    assert report["valid"] and report["documents"] == 1
    assert report["missing_archival_api_snapshot"]
    for name in records:
        assert checksum(raw / name) == digests[name]
    (raw / "data.tar.gz").write_bytes(b"wrong archive")
    with pytest.raises(ValueError, match="refusing to substitute"):
        data_loader.prepare()
