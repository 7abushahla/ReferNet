"""Allocated-only isolated candidate inspection, preparation, and CPU timing."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load authenticated module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scientific_origins(core, candidate):
    root = core.absolute_path(candidate["scientific_source"]["directory"], "scientific source")
    parent = core.authenticated_json(candidate["parent_manifest"], "scientific parent")
    inventory = {str(root / item["path"]) for item in parent["source_files"]}
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name != "refer_or_clear" and not name.startswith("refer_or_clear."):
            continue
        origin = getattr(module, "__file__", None)
        core.require(origin is not None, f"scientific import origin unavailable: {name}")
        path = core.absolute_path(origin, "scientific import origin")
        core.require(path.is_relative_to(root / "src") and str(path) in inventory,
                     f"scientific import origin drift: {name}")
        for search_path in getattr(module, "__path__", ()):
            actual = core.absolute_path(search_path, "scientific package search path")
            core.require(actual.is_relative_to(root / "src"), f"scientific package path drift: {name}")
        origins[name] = str(path)
    core.require("refer_or_clear" in origins, "scientific package not preloaded")
    return origins


def load_scientific_helpers(core, catalogue, candidate):
    core.require(not any(name == "refer_or_clear" or name.startswith("refer_or_clear.") for name in sys.modules),
                 "scientific package already imported before source isolation")
    core.authenticate_scientific_source(candidate)
    source = Path(candidate["scientific_source"]["directory"])
    tools = catalogue["measurement_tools"]
    for name in ("bench_anchor_candidates.py", "bench_anchor_candidate_worker.py"):
        core.require(name in tools["files"], "authenticated measurement coordinator/worker required")
    core.require(Path(tools["directory"]) / "bench_anchor_candidate_worker.py" == Path(__file__).resolve()
                 and Path(tools["directory"]) / "bench_anchor_candidates.py" == Path(core.__file__).resolve(),
                 "measurement tool origin mismatch")
    os.environ.pop("PYTHONPATH", None)
    sys.path.insert(0, str(source / "src"))
    importlib.invalidate_caches()
    package = importlib.import_module("refer_or_clear")
    contract = importlib.import_module("refer_or_clear.anchor_contract")
    pilots = importlib.import_module("refer_or_clear.pilots")
    query = importlib.import_module("refer_or_clear.data.query")
    scientific_origins(core, candidate)
    core.require(pilots.source_sha256() == candidate["identities"]["source_sha256"], "scientific helper runtime source mismatch")
    reporter = load_file_module("report_anchor_pilots", Path(tools["directory"]) / "report_anchor_pilots.py")
    bench = load_file_module("anchor_candidate_measurement_helpers", Path(tools["directory"]) / "bench_anchor_pilots.py")
    scientific_origins(core, candidate)
    core.require(reporter.read_pilot_checkpoint is pilots.read_pilot_checkpoint
                 and bench.read_pilot_checkpoint is pilots.read_pilot_checkpoint
                 and bench.build_initialized_model is pilots.build_initialized_model
                 and bench.load_role_rows is query.load_role_rows and bench.AnchorQueryDataset is query.AnchorQueryDataset
                 and bench.contract_for is contract.contract_for, "measurement helpers replaced scientific APIs")
    named = contract.contract_for(candidate["track"])
    core.require(named.architecture_id == candidate["architecture_id"] and named.bn_policy == "running",
                 "unknown architecture or normalization policy")
    expected_shape = (3, 299, 299) if candidate["track"] == "R" else (4, 512, 800)
    core.require(tuple(named.input_shape) == expected_shape, "scientific input shape mismatch")
    return package, contract, pilots, query, reporter, bench


def checked_output(core, requested, catalogue, request_binding, request):
    path = Path(requested).absolute()
    core.reject_symlinks(path)
    actual = path.resolve()
    core.require(actual.suffix == ".json" and not actual.exists(), "new JSON worker output required")
    protections = core.protected_paths(catalogue) + [request_binding["path"], request["catalogue"]["path"]]
    for binding in (request.get("inspections") or {}).values():
        protections.append(binding["path"])
    if request.get("runtime_manifest"):
        protections.append(request["runtime_manifest"]["path"])
    for value in protections:
        root = Path(value)
        core.require(not actual.is_relative_to(root) and not root.is_relative_to(actual),
                     f"worker output overlaps protected input: {root}")
    core.require(actual.parent.is_dir(), "worker output parent must already exist")
    return actual


def inspect_candidate(core, candidate, reporter):
    inspection = reporter.inspect_run(Path(candidate["run_dir"]), candidate["track"])
    core.require(inspection.get("status") == "complete" and inspection.get("eligible_candidate") is True,
                 f"selected candidate inspection failed: {inspection.get('reasons', [])}")
    train = core.authenticated_json(candidate["train_json"], "train ledger")
    core.require(inspection.get("track") == candidate["track"] and inspection.get("identities") == candidate["identities"]
                 and inspection.get("selected_epoch") == train["selected_epoch"], "selected inspection identity/epoch mismatch")
    expected = {"train.json": candidate["train_json"]["sha256"],
                **{name: binding["sha256"] for name, binding in candidate["selected_evidence"].items()}}
    core.require(inspection.get("artifact_sha256") == expected, "selected inspection evidence mismatch")
    for name in ("n_active_parameters", "n_total_parameters"):
        core.require(type(inspection.get(name)) is int and inspection[name] == train[name], "selected parameter counts mismatch")
    core.require(0 < inspection["n_active_parameters"] <= inspection["n_total_parameters"], "invalid selected parameter counts")
    checkpoint = Path(candidate["selected_checkpoint"]["path"])
    core.require(inspection.get("checkpoint_bytes") == checkpoint.stat().st_size, "selected checkpoint byte count mismatch")
    return inspection


def prepare_candidate(core, catalogue, candidate, request, inspections, contract, query, bench):
    train = core.authenticated_json(candidate["train_json"], "train ledger")
    run = Path(candidate["run_dir"])
    resolved = {}
    for kind in ("records", "split"):
        if kind in candidate.get("relocations", {}):
            path = Path(candidate[kind]["path"])
        else:
            path = bench.resolve_metadata(run, train, kind)
        path = core.absolute_path(str(path), f"resolved {kind}")
        core.require(str(path) in (candidate[kind]["path"], train["paths"][kind]), "unbound metadata fallback")
        core.require(core.file_sha256(path) == candidate["identities"][f"{kind}_sha256"], f"resolved {kind} SHA256 mismatch")
        resolved[kind] = path
    weights = core.absolute_path(str(bench.resolve_weights(run, train)), "resolved offline weights")
    core.require(str(weights) in (candidate["weights"]["path"], train["paths"]["weights"]), "unbound weight fallback")
    core.require(core.file_sha256(weights) == candidate["identities"]["weights_sha256"], "resolved offline weights SHA256 mismatch")
    rows = query.load_role_rows(resolved["records"], resolved["split"], "train")
    core.require(len(rows) == 1778, "full training metadata required")
    rows = rows.iloc[:200].copy()
    records = rows.to_dict("records")
    trained = bench.runtime_train_rows(train)
    core.require([row["image_id"] for row in records] == catalogue["image_ids"], "first 200 train IDs changed")
    core.require(all(row[field] == saved[field] for row, saved in zip(records, trained) for field in core.ROW_FIELDS)
                 and len(trained) == 200 and all(row["role"] == "train" for row in records), "training source metadata changed")
    raw = Path(candidate["raw_root"])
    legacy = Path(candidate["legacy_root"])
    sources = []
    for row in records:
        expected = [core.absolute_path(str(path), "authorized training input")
                    for path in bench.source_paths(row, candidate["track"], raw, legacy)]
        root = raw if candidate["track"] == "R" else legacy
        core.require(all(path.is_relative_to(root) for path in expected), "training source path escape")
        artifacts = bench.source_artifacts(row, candidate["track"], raw, legacy)
        core.require(isinstance(artifacts, list) and len(artifacts) == len(expected), "training artifact count mismatch")
        for index, (artifact, path) in enumerate(zip(artifacts, expected)):
            core.require(artifact["path"] == str(path) and artifact["sha256"] == core.file_sha256(path), "training artifact SHA256/path mismatch")
            shape = artifact["shape"]
            core.require(isinstance(shape, list) and all(type(value) is int and value > 0 for value in shape), "training artifact shape invalid")
            if candidate["track"] != "R":
                core.require(shape == ([512, 800, 3] if index == 0 else [512, 800]), "legacy artifact shape mismatch")
            else:
                core.require(len(shape) == 3 and shape[2] == 3, "raw RGB source shape mismatch")
        sources.append({"image_id": row["image_id"], "artifacts": artifacts})
    rows["role"] = "val"
    query.AnchorQueryDataset(rows, raw, legacy, candidate["track"], False, {"enabled": False})
    selected = inspections[candidate["track"]]["inspection"]
    return {"image_ids": catalogue["image_ids"], "image_ids_sha256": core.json_sha256(catalogue["image_ids"]),
            "sources": sources, "sources_sha256": core.json_sha256(sources),
            "source_roots": {"raw": str(raw), "legacy": str(legacy)}, "weights_path": str(weights),
            "resolved_metadata": {name: str(path) for name, path in resolved.items()},
            "input_shape": [1, *contract.contract_for(candidate["track"]).input_shape],
            "membership_role": "train", "query_view_role": "val", "augment": {"enabled": False},
            "checkpoint_sha256": candidate["selected_checkpoint"]["sha256"], "checkpoint_bytes": selected["checkpoint_bytes"],
            "selected_epoch": train["selected_epoch"], "selection_key": train["selection_key"],
            "n_active_parameters": selected["n_active_parameters"], "n_total_parameters": selected["n_total_parameters"],
            "initialization": train["initialization"], "preparation_scope": "raw_input" if candidate["track"] == "R" else "processed_input_only",
            "inspections": request["inspections"], "full_source_to_decision_latency": "not_measured"}


def model_fingerprints(model, pilots):
    return {"parameters_sha256": pilots.tensor_sha256(dict(model.named_parameters())),
            "buffers_sha256": pilots.tensor_sha256(dict(model.named_buffers())),
            "state_sha256": pilots.tensor_sha256(model.state_dict())}


def execution_snapshot(model, core, bench, torch, expected_host):
    host = bench.host_info()
    core.require(host == expected_host, "timing host identity changed")
    threads = {"torch_intraop": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
               "opencv_threads": bench.cv2.getNumThreads(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
               "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")}
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        core.require(threads[name] == str(core.CONTRACT[name]), "timing thread/environment drift")
        threads[name] = int(threads[name])
    core.require(threads == {name: core.CONTRACT[name] for name in threads}, "timing thread/environment drift")
    core.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only allocation required")
    core.require(torch.is_inference_mode_enabled() and not torch.is_autocast_cpu_enabled()
                 and not torch.is_autocast_enabled(), "inference/AMP contract drift")
    core.require(torch.get_default_dtype() == torch.float32, "default precision drift")
    core.require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32, "TF32 must be disabled")
    core.require(all(not module.training for module in model.modules()), "model must remain in eval mode")
    core.require(all(tensor.device.type == "cpu" and tensor.dtype == torch.float32 for tensor in model.parameters()),
                 "CPU FP32 parameters required")
    core.require(all(tensor.device.type == "cpu" and (not tensor.is_floating_point() or tensor.dtype == torch.float32)
                     for name, tensor in model.named_buffers()), "CPU FP32 floating buffers required")
    return {"host": host, "thread_contract": threads,
            "precision_contract": {"device": "cpu", "precision": "float32", "batch_size": 1, "amp": False, "tf32": False},
            "inference_mode": True, "eval_mode": True, "CUDA_VISIBLE_DEVICES": ""}


def measure_model(model, dataset, image_ids, *, core, pilots, bench, torch, expected_shape, expected_host, clock=None):
    class QueryOnly:
        def __getitem__(self, index):
            sample = dataset[index]
            core.require(set(sample) <= {"query", "label", "meta"} and {"query", "meta"} <= set(sample),
                         "query-only timing forbids dense/reference targets")
            tensor = sample["query"]
            core.require(list(tensor.shape) == expected_shape[1:] and tensor.device.type == "cpu"
                         and tensor.dtype == torch.float32, "timing query shape/precision mismatch")
            return sample

    with torch.inference_mode(), torch.autocast(device_type="cpu", enabled=False):
        before = execution_snapshot(model, core, bench, torch, expected_host)
        fingerprints = model_fingerprints(model, pilots)
        options = {} if clock is None else {"clock": clock}
        result = bench.measure_track(model, QueryOnly(), image_ids, **options)
        after = execution_snapshot(model, core, bench, torch, expected_host)
        final = model_fingerprints(model, pilots)
        core.require(before == after and fingerprints == final, "model state or execution environment mutated during timing")
    return dict(result, model_before=fingerprints, model_after=final, runtime_before=before, runtime_after=after)


def measure_candidate(core, catalogue, candidate, manifest, pilots, query, bench, torch=None):
    if torch is None:
        torch = importlib.import_module("torch")
    prepared = manifest["tracks"][candidate["track"]]
    bench.verify_source_artifacts(prepared)
    train = core.authenticated_json(candidate["train_json"], "selected train ledger")
    config = train["config"]
    model = pilots.build_initialized_model(config, Path(prepared["weights_path"]), candidate["identities"]["weights_sha256"])
    saved = pilots.read_pilot_checkpoint(Path(candidate["selected_checkpoint"]["path"]), config, candidate["identities"])
    core.require(saved["phase"] == "train" and saved["epoch"] == train["selected_epoch"]
                 and list(saved["selection_key"]) == train["selection_key"], "selected checkpoint phase/epoch/key mismatch")
    core.require(model.initialization_provenance["pretrained_tensor_sha256"]
                 == saved["initialization_provenance"]["pretrained_tensor_sha256"], "initialized pretrained tensors mismatch")
    model.load_state_dict(saved["state_dict"], strict=True)
    model.cpu().float().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    counts = {"n_active_parameters": sum(tensor.numel() for tensor in model.parameters() if tensor.requires_grad),
              "n_total_parameters": sum(tensor.numel() for tensor in model.parameters())}
    core.require(all(counts[name] == train[name] for name in counts), "actual parameter counts mismatch")
    checkpoint_state = pilots.tensor_sha256(saved["state_dict"])
    core.require(model_fingerprints(model, pilots)["state_sha256"] == checkpoint_state, "loaded selected state mismatch")
    rows = query.load_role_rows(Path(prepared["resolved_metadata"]["records"]), Path(prepared["resolved_metadata"]["split"]), "train")
    core.require(len(rows) == 1778, "full train membership required")
    rows = rows.iloc[:200].copy()
    records = rows.to_dict("records")
    trained = bench.runtime_train_rows(train)
    core.require(len(trained) == 200 and [row["image_id"] for row in records] == manifest["image_ids"]
                 and all(row[field] == saved_row[field] for row, saved_row in zip(records, trained) for field in core.ROW_FIELDS),
                 "timed training source metadata drift")
    rows["role"] = "val"
    dataset = query.AnchorQueryDataset(rows, Path(candidate["raw_root"]), Path(candidate["legacy_root"]),
                                     candidate["track"], False, {"enabled": False})
    timing = measure_model(model, dataset, manifest["image_ids"], core=core, pilots=pilots, bench=bench, torch=torch,
                           expected_shape=prepared["input_shape"], expected_host=manifest["host"])
    bench.verify_source_artifacts(prepared)
    return dict(timing, checkpoint_state_sha256=checkpoint_state, **counts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("inspect", "prepare", "measure"), required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    try:
        core = load_file_module("anchor_candidate_coordinator", Path(__file__).with_name("bench_anchor_candidates.py"))
        core.require(sys.flags.isolated, "worker requires an isolated fresh interpreter (-I)")
        binding = {"path": str(Path(args.request).absolute()), "sha256": args.request_sha256}
        request = core.authenticated_json(binding, "worker request")
        core.require(request.get("schema_version") == "anchor_candidate_worker_request_v1"
                     and request.get("operation") == args.operation and request.get("track") in core.TRACK_ORDER,
                     "worker request schema/operation/track mismatch")
        if args.operation == "measure":
            core.require(isinstance(request.get("runtime_manifest"), dict), "frozen runtime manifest required before timing")
        catalogue = core.authenticated_json(request["catalogue"], "candidate catalogue")
        readiness = core.prepare_bindings(catalogue)
        core.require(readiness["status"] == "metadata-ready", f"all completed selected reviews required: {readiness['status']} {readiness['errors']}")
        core.validate_allocation()
        output = checked_output(core, args.out, catalogue, binding, request)
        candidate = next(item for item in catalogue["candidates"] if item["track"] == request["track"])
        inspections = None
        if args.operation in ("prepare", "measure"):
            inspections = core.validate_inspection_bundle(request.get("inspections"), catalogue, request["catalogue"]["sha256"])
        manifest = None
        if args.operation == "measure":
            manifest = core.validate_runtime_manifest(request["runtime_manifest"], request["catalogue"])
            core.require(request["inspections"] == manifest["inspections"], "timing inspection authority mismatch")
        before = core.candidate_fingerprints(catalogue, candidate)
        package, contract, pilots, query, reporter, bench = load_scientific_helpers(core, catalogue, candidate)
        bench.configure_threads()
        host = bench.host_info()
        bench.validate_host(host)
        result = {"schema_version": "anchor_candidate_worker_result_v1", "operation": args.operation,
                  "status": {"inspect": "inspected", "prepare": "prepared", "measure": "measured"}[args.operation], "track": candidate["track"],
                  "catalogue_sha256": request["catalogue"]["sha256"], "request_sha256": binding["sha256"],
                  "identities": candidate["identities"], "scientific_source": candidate["scientific_source"],
                  "measurement_tools": catalogue["measurement_tools"], "host": host, "comparable": False,
                  "execution_authorized": False, "thread_contract": {name: core.CONTRACT[name] for name in
                      ("torch_intraop", "torch_interop", "opencv_threads", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                  "precision_contract": {"device": "cpu", "precision": "float32", "batch_size": 1, "amp": False, "tf32": False}}
        if args.operation == "inspect":
            result["inspection"] = inspect_candidate(core, candidate, reporter)
        elif args.operation == "prepare":
            core.require(all(item["host"] == host for item in inspections.values()), "preparation host differs from inspection host")
            result.update(prepare_candidate(core, catalogue, candidate, request, inspections, contract, query, bench))
        else:
            core.require(host == manifest["host"], "timing host differs from frozen manifest")
            result.update({name: manifest["tracks"][candidate["track"]][name] for name in
                           ("image_ids", "image_ids_sha256", "sources", "sources_sha256", "input_shape", "preparation_scope",
                            "checkpoint_sha256", "checkpoint_bytes", "selected_epoch", "selection_key", "initialization")})
            result.update(measure_candidate(core, catalogue, candidate, manifest, pilots, query, bench))
            result["runtime_manifest"] = request["runtime_manifest"]
            result["manifest_sha256"] = manifest["manifest_sha256"]
            core.validate_runtime_manifest(request["runtime_manifest"], request["catalogue"])
        core.require(bench.host_info() == host, "worker host identity changed")
        result["scientific_module_origins"] = scientific_origins(core, candidate)
        after = core.candidate_fingerprints(catalogue, candidate)
        core.require(before == after, "immutable scientific/runtime authorities changed")
        result["fingerprints"] = after
        with output.open("x") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, AttributeError, ImportError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
