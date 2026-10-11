from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def stage(repository: Path, code: Path, manifest: Path) -> None:
    repository, code = repository.resolve(), code.resolve()
    if code == repository or code in repository.parents or repository in code.parents:
        raise ValueError("Test staging must not overwrite the repository")
    metadata = [Path("data/processed/records.csv"), Path("data/processed/splits/protocol_a_it0.csv")]
    metadata += [Path(f"data/raw/sidtd/kfold_split_templates/split_kfold/{role}_split_SIDTD_it_{fold}.csv")
                 for fold in range(10) for role in ("train", "val", "test")]
    image_relative = Path("data/raw/sidtd/templates/Images")
    image_root = repository / image_relative
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    payloads = [(relative, (repository / relative).read_bytes()) for relative in metadata]
    for relative in metadata + [image_relative]:
        destination = code / relative
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
    if manifest.exists():
        raise FileExistsError(manifest)
    entries = []
    for relative, payload in payloads:
        destination = code / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        checksum = hashlib.sha256(payload).hexdigest()
        if hashlib.sha256(destination.read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f"Staged metadata changed: {relative}")
        entries.append(dict(relative_path=str(relative), sha256=checksum))
    image_link = code / image_relative
    image_link.parent.mkdir(parents=True, exist_ok=True)
    image_link.symlink_to(image_root, target_is_directory=True)
    manifest.write_text(json.dumps(dict(
        schema_version="full_suite_metadata_v1", repository=str(repository), code=str(code),
        metadata_files=entries, image_existence_link=str(image_root),
        image_scope="Legacy scorer checks file existence only; no corpus image bytes staged or decoded",
    ), indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    stage(args.repository, args.code, args.manifest)


if __name__ == "__main__":
    main()
