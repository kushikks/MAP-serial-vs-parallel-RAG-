"""Package small reproducibility sources; exclude datasets, weights and environments."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = [ROOT / n for n in ("retrieval.py", "README.md", "requirements.txt", "requirements-lock.txt", ".gitignore")]
    for directory, pattern in (("src", "*.py"), ("tests", "*.py"), ("scripts", "*.py"),
                               ("configs", "*.yaml"), ("docs", "*.md"), ("examples", "*.json"),
                               ("data/manifests", "*.json")):
        files.extend((ROOT / directory).glob(pattern))
    files = sorted(set(files))
    destination = ROOT / "results/source_snapshots"
    destination.mkdir(parents=True, exist_ok=True)
    digests = {}
    with zipfile.ZipFile(destination / "member_a_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            digests[relative] = hashlib.sha256(content).hexdigest()
            archive.writestr(relative, content)
    (destination / "final_source_manifest.json").write_text(json.dumps(digests, indent=2), encoding="utf-8")
    print(json.dumps({"source_files": len(files), "archive": str(destination / "member_a_source.zip")}))


if __name__ == "__main__":
    main()
