"""Shared allocated-host preliminary CPU timing. Never train or select on timing."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from refer_or_clear.anchor_contract import contract_for
from refer_or_clear.data.query import AnchorQueryDataset, load_role_rows
from refer_or_clear.pilots import (
    build_initialized_model, file_sha256, json_sha256, read_pilot_checkpoint, source_sha256,
)
from report_anchor_pilots import inspect_run, read_json, require

CONTRACT = {"name": "anchor_cpu_v1", "track_order": ["R", "D", "N"], "role": "train", "fold": 0,
            "n_documents": 200, "id_selection": "first_200_lexicographic_train_ids", "warmups": 20,
            "timed_passes": 1, "batch_size": 1, "precision": "float32", "device": "cpu",
            "execution": "eager_inference_mode", "torch_intraop": 4, "torch_interop": 1,
            "opencv_threads": 1, "OMP_NUM_THREADS": 4, "MKL_NUM_THREADS": 4,
            "concurrent_benchmark_workload": False,
            "network_boundary": "ready_tensor_forward_through_attack_score_materialization",
            "preparation_boundary": "source_read_decode_declared_transform_no_upstream_geometry",
            "interpretation": "compound_recipe_preliminary_not_officer_device_end_to_end"}


def host_info():
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    versions = {"python": sys.version.split()[0], "torch": str(torch.__version__), "numpy": np.__version__,
                "opencv": cv2.__version__}
    for package in ("torchvision", "mmcv", "mmengine", "mmsegmentation", "opencd"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return {"hostname": platform.node(), "cpu_model": cpu_model,
            "boot_id": boot_path.read_text().strip() if boot_path.is_file() else None,
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
            "allocated_cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", "0")),
            "job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME"),
            "os": platform.platform(), "versions": versions,
            "environment_prefix": os.environ.get("CONDA_PREFIX"), "python_executable": sys.executable}


def validate_host(host):
    require(bool(host.get("job_id")) and bool(host.get("hostname")) and bool(host.get("boot_id"))
            and bool(host.get("cpu_model")), "common physical allocated host is unavailable")
    require(not any(word in (host.get("node") or host["hostname"]).lower() for word in ("login", "head", "frontend", "lgn")),
            "login host is forbidden")
    require(host.get("allocated_cpus", 0) >= 4 and len(host.get("affinity", [])) >= 4, "at least four allocated CPUs required")


def configure_threads():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    require(torch.get_num_threads() == 4 and torch.get_num_interop_threads() == 1 and cv2.getNumThreads() == 1,
            "thread contract could not be established")


def make_manifest(ids, host, tracks):
    validate_host(host)
    require(len(ids) == 200 and len(set(ids)) == 200, "exactly 200 unique train IDs required")
    require(ids == sorted(ids), "train IDs must be lexicographic")
    manifest = {"schema_version": "anchor_cpu_v1", "contract": CONTRACT, "host": host,
                "image_ids": ids, "image_ids_sha256": json_sha256(ids), "tracks": tracks}
    manifest["tool_sha256"] = {name: file_sha256(Path(__file__).parent / name)
                              for name in ("bench_anchor_pilots.py", "report_anchor_pilots.py")}
    return dict(manifest, manifest_sha256=json_sha256(manifest))


def freeze_manifest(path, manifest):
    path = Path(path)
    if path.exists():
        require(read_json(path) == manifest, "runtime manifest/host/source mismatch")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")


def stats(values):
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}


def measure_track(model, dataset, ids, *, clock=time.perf_counter):
    require(len(ids) == 200 and len(set(ids)) == 200, "exactly 200 timed documents required")
    def score(index, timed):
        before = clock() if timed else None
        sample = dataset[index]
        require(sample["meta"]["image_id"] == ids[index], "timed document identity mismatch")
        query = sample["query"].unsqueeze(0)
        require(query.device.type == "cpu" and query.dtype == torch.float32, "batch-one FP32 CPU tensor required")
        prepared = clock() if timed else None
        attack_score = model(query)["attack_logit"].reshape(-1)[0].item()
        finished = clock() if timed else None
        require(np.isfinite(attack_score), "non-finite materialized score")
        if timed:
            require(before <= prepared <= finished, "non-monotonic timing clock")
            return {"image_id": ids[index], "preparation_ms": 1000 * (prepared - before),
                    "network_ms": 1000 * (finished - prepared), "prepared_input_total_ms": 1000 * (finished - before)}
    with torch.inference_mode():
        for index in range(20):
            score(index, False)
        durations = [score(index, True) for index in range(200)]
    return {"warmups": 20, "n_documents": 200, "per_image": durations,
            "timing": {key: stats([row[key] for row in durations])
                       for key in ("preparation_ms", "network_ms", "prepared_input_total_ms")}}


def resolve_weights(run_dir, info):
    path = Path(info["paths"]["weights"])
    if not path.is_file():
        parent_manifest = Path(run_dir).parent / "input_manifest.json"
        path = Path(read_json(parent_manifest)["paths"]["weights"])
    require(file_sha256(path) == info["identities"]["weights_sha256"], "original offline weights SHA256 mismatch")
    return path


def resolve_metadata(run_dir, info, key):
    path = Path(info["paths"][key])
    if not path.is_file():
        path = Path(read_json(Path(run_dir).parent / "input_manifest.json")["paths"][key])
    require(file_sha256(path) == info["identities"][f"{key}_sha256"], f"{key} SHA256 mismatch")
    return path


def source_paths(row, track, raw_root, legacy_root):
    if track == "R":
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "raw path must remain within root")
        return [Path(raw_root) / relative]
    else:
        image_id, design_id = row["image_id"], row["design_id"].replace("/", "_")
        require(all(".." not in Path(value).parts and not Path(value).is_absolute() for value in (image_id, design_id)),
                "legacy paths must remain within root")
        return [Path(legacy_root) / "canvases" / f"{image_id}.png", Path(legacy_root) / "zones" / f"{design_id}.png"]


def source_artifacts(row, track, raw_root, legacy_root):
    paths = source_paths(row, track, raw_root, legacy_root)
    artifacts = []
    for index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(str(path))
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if index == 1 else cv2.IMREAD_COLOR)
        require(image is not None, f"source cannot decode {path}")
        if track != "R":
            require(image.shape[:2] == (512, 800), f"source shape mismatch {path}")
        artifacts.append({"path": str(path.resolve()), "sha256": file_sha256(path), "shape": list(image.shape)})
    return artifacts


def verify_source_artifacts(metadata):
    for source in metadata["sources"]:
        for artifact in source["artifacts"]:
            require(file_sha256(artifact["path"]) == artifact["sha256"], "source artifact SHA256 changed")


def runtime_train_rows(info):
    rows = info["membership"]["train"]
    ids = [row["image_id"] for row in rows]
    require(len(rows) == 1778 and len(set(ids)) == 1778, "runtime candidate train IDs incomplete/duplicate")
    return sorted(rows, key=lambda row: row["image_id"])[:200]


def validate_source_mapping(frozen, result, rows, track):
    require(result["source_roots"] == frozen["source_roots"], "runtime source roots mismatch")
    roots = frozen["source_roots"]
    require(all(Path(roots[key]).is_absolute() for key in ("raw", "legacy")), "runtime source roots must be absolute")
    sources = frozen["sources"]
    require(result["sources"] == sources and result["sources_sha256"] == frozen["sources_sha256"] == json_sha256(sources),
            "runtime source manifest mismatch")
    require([source["image_id"] for source in sources] == [row["image_id"] for row in rows], "runtime source/train IDs mismatch")
    for source, row in zip(sources, rows):
        expected_paths = [str(path.resolve()) for path in source_paths(row, track, roots["raw"], roots["legacy"])]
        require([artifact["path"] for artifact in source["artifacts"]] == expected_paths,
                "runtime source mapping differs from candidate training ledger")


def prepare_runs(runs, raw_root, legacy_root):
    prepared, metadata, ids = {}, {}, None
    for track, run in zip("RDN", runs):
        evidence = inspect_run(run, track)
        require(evidence["status"] == "complete", f"{track} selected validation evidence incomplete/invalid ({evidence['reasons']})")
        info = read_json(Path(run) / "train.json")
        require(info["identities"]["source_sha256"] == source_sha256(), "immutable runtime source hash mismatch")
        records, split = (resolve_metadata(run, info, key) for key in ("records", "split"))
        rows = load_role_rows(records, split, "train").iloc[:200].copy()
        train_rows = runtime_train_rows(info)
        require(all(row[key] == trained[key] for row, trained in zip(rows.to_dict("records"), train_rows)
                    for key in ("image_id", "path", "design_id")), "runtime source/train IDs differ from candidate ledger")
        current_ids = rows.image_id.tolist()
        require(len(current_ids) == 200, "exactly 200 train documents required")
        if ids is None:
            ids = current_ids
        require(ids == current_ids, "tracks do not use the same first 200 train IDs")
        sources = [{"image_id": row["image_id"], "artifacts": source_artifacts(row, track, raw_root, legacy_root)}
                   for row in rows.to_dict("records")]
        weights = resolve_weights(run, info)
        checkpoint = Path(run) / "checkpoints/best.pt"
        metadata[track] = {"run_dir": str(Path(run).resolve()), "identities": info["identities"],
                           "checkpoint_sha256": file_sha256(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
                           "initialization": info["initialization"], "weights_path": str(weights.resolve()),
                           "input_shape": [1, *contract_for(track).input_shape], "sources": sources,
                           "source_roots": {"raw": str(Path(raw_root).resolve()), "legacy": str(Path(legacy_root).resolve())},
                           "sources_sha256": json_sha256(sources), "n_active_parameters": info["n_active_parameters"],
                           "n_total_parameters": info["n_total_parameters"],
                           "preparation_scope": "raw_input" if track == "R" else "processed_input_only"}
        rows["role"] = "val"
        dataset = AnchorQueryDataset(rows, raw_root, legacy_root, track, False, {"enabled": False})
        prepared[track] = (info, weights, checkpoint, dataset)
    require(all(metadata[track]["identities"][key] == metadata["R"]["identities"][key]
                for track in "DN" for key in ("records_sha256", "split_sha256", "source_sha256")),
            "mixed source/membership identities")
    return ids, metadata, prepared


def validate_measurement(measurement, candidates):
    manifest = measurement["manifest"]
    expected = dict(manifest)
    digest = expected.pop("manifest_sha256")
    require(json_sha256(expected) == digest == measurement["manifest_sha256"], "runtime manifest hash mismatch")
    require(manifest["contract"] == CONTRACT, "runtime contract mismatch")
    validate_host(manifest["host"])
    ids = manifest["image_ids"]
    require(len(ids) == 200 and ids == sorted(set(ids)) and json_sha256(ids) == manifest["image_ids_sha256"], "runtime IDs mismatch")
    require(measurement["status"] == "measured" and set(measurement["tracks"]) == set("RDN"), "runtime incomplete")
    for track in "RDN":
        result, frozen, candidate = measurement["tracks"][track], manifest["tracks"][track], candidates[track]
        require(candidate["status"] == "complete" and frozen["identities"] == candidate["identities"], "runtime candidate mismatch")
        ledger_path = Path(candidate["run_dir"]) / "train.json"
        ledger_hash = candidate.get("artifact_sha256", {}).get("train.json")
        if ledger_hash is not None:
            require(file_sha256(ledger_path) == ledger_hash, "runtime candidate source ledger changed")
        train_rows = runtime_train_rows(read_json(ledger_path))
        require(ids == [row["image_id"] for row in train_rows], "runtime IDs differ from candidate first 200 train IDs")
        validate_source_mapping(frozen, result, train_rows, track)
        checkpoint = Path(candidate["run_dir"]) / "checkpoints/best.pt"
        require(frozen["checkpoint_sha256"] == result["checkpoint_sha256"] == file_sha256(checkpoint), "runtime checkpoint mismatch")
        require(result["input_shape"] == frozen["input_shape"] == [1, *contract_for(track).input_shape], "runtime input shape mismatch")
        require(result["preparation_scope"] == frozen["preparation_scope"]
                == ("raw_input" if track == "R" else "processed_input_only"), "runtime preparation scope mismatch")
        for key in ("n_active_parameters", "n_total_parameters"):
            require(result[key] == frozen[key] == candidate[key], "runtime parameter count mismatch")
        require(result["host"] == manifest["host"] and result["image_ids_sha256"] == manifest["image_ids_sha256"], "runtime common host/IDs mismatch")
        require(result["warmups"] == 20 and result["n_documents"] == 200 and len(result["per_image"]) == 200,
                "runtime counts mismatch")
        require([row["image_id"] for row in result["per_image"]] == ids, "runtime timed IDs mismatch")
        for row in result["per_image"]:
            require(all(math_value >= 0 and np.isfinite(math_value) for math_value in
                        (row["preparation_ms"], row["network_ms"], row["prepared_input_total_ms"])), "invalid durations")
            require(np.isclose(row["preparation_ms"] + row["network_ms"], row["prepared_input_total_ms"]), "timing accounting mismatch")
        for key in ("preparation_ms", "network_ms", "prepared_input_total_ms"):
            require(result["timing"][key] == stats([row[key] for row in result["per_image"]]), "timing summary mismatch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=3, type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = {"schema_version": "anchor_cpu_v1", "status": "not-comparable", "tracks": {}}
    try:
        host = host_info()
        validate_host(host)
        configure_threads()
        ids, metadata, prepared = prepare_runs(args.runs, args.raw_root, args.legacy_root)
        manifest = make_manifest(ids, host, metadata)
        freeze_manifest(args.runtime_manifest, manifest)
        result.update(manifest=manifest, manifest_sha256=manifest["manifest_sha256"])
        for track in "RDN":
            require(host_info() == host, "allocated host/affinity changed before timing")
            verify_source_artifacts(metadata[track])
            info, weights, checkpoint_path, dataset = prepared[track]
            require(file_sha256(checkpoint_path) == metadata[track]["checkpoint_sha256"], "selected checkpoint SHA256 changed")
            model = build_initialized_model(info["config"], weights, info["identities"]["weights_sha256"])
            saved = read_pilot_checkpoint(checkpoint_path, info["config"], info["identities"])
            require(model.initialization_provenance["pretrained_tensor_sha256"]
                    == saved["initialization_provenance"]["pretrained_tensor_sha256"], "initialized tensor hash mismatch")
            model.load_state_dict(saved["state_dict"], strict=True)
            model.cpu().float().eval()
            require(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                    == metadata[track]["n_active_parameters"] and sum(parameter.numel() for parameter in model.parameters())
                    == metadata[track]["n_total_parameters"], "actual model parameter counts mismatch")
            result["tracks"][track] = dict(measure_track(model, dataset, ids), host=host,
                                           image_ids_sha256=manifest["image_ids_sha256"], **metadata[track])
            del model, saved
            require(host_info() == host, "allocated host/affinity changed during timing")
            verify_source_artifacts(metadata[track])
            require(file_sha256(checkpoint_path) == metadata[track]["checkpoint_sha256"], "selected checkpoint SHA256 changed")
        result["status"] = "measured"
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        result["reason"] = str(error)
        result["status"] = "not-comparable"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(0 if main()["status"] == "measured" else 1)
