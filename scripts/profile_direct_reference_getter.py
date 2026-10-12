"""External 38088-only probe: 16 warmup and 32 profiled TRAIN getters, six OFF GPU steps."""
import argparse
import cProfile
import hashlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import pstats
import sys
import time

STAGE = Path("/scratch/b00090279/direct-reference-throughput-38088")
SOURCE_SHA = "c1c59cefd2dce9a1815b235f81f763eb8781667853a8d80bf36666a71074f815"
REQUEST_SHA = "36247e1ac7ef13dd9944dd5fd6aa6d191765ce28e4e2362648cc69171a6a8ba8"
REPORT_SHA = "70dda430267086ceb884013e2a70a3e0aa88f5cf78e419b08b597f785088766f"
MANIFEST_SHA = "2d00467a322f5fd7c1c2b2a99a0cc20522d17adb582294d024822b839e7db627"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(value):
    target = Path(value)
    require(target.is_absolute() and ".." not in target.parts, "absolute nontraversing path required")
    require(not any(parent.is_symlink() for parent in (target, *target.parents)), "symlink path forbidden")
    return target


def read(binding):
    target = path(binding["path"])
    require(target.is_file(), f"existing authority required: {target}")
    payload = target.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == binding["sha256"], f"SHA256 mismatch: {target}")
    return payload


def load_script(target):
    spec = importlib.util.spec_from_file_location("_frozen_profile_" + target.stem, target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap(args):
    require(os.environ.get("SLURMD_NODENAME") == "gpu-dy-g5-0-17", "original node 17 required")
    require(args.request_sha256 == REQUEST_SHA and args.report_sha256 == REPORT_SHA, "38088 pinned hashes required")
    source, run, output = path(args.source_dir), path(args.run_dir), path(args.output_root)
    require(source == STAGE / "code" and source.is_dir(), "existing original scratch code required")
    require(not path(str(Path(__file__).absolute())).is_relative_to(source), "instrumentation must be external")
    manifest = json.loads(read(dict(path=str(run / "input_manifest.json"), sha256=MANIFEST_SHA)))
    report = json.loads(read(dict(path=args.report, sha256=REPORT_SHA)))
    request = json.loads(read(dict(path=args.request, sha256=REQUEST_SHA)))
    require(manifest["paths"]["source_dir"] == str(source) and manifest["paths"]["run_dir"] == str(run)
            and Path(args.report) == run / "throughput/report.json", "original source/run/report paths required")
    require(manifest["request"] == report["request"] == dict(path=args.request, sha256=REQUEST_SHA)
            and Path(args.request) == STAGE / "inputs/request.json", "original request binding required")
    require(report["completed"] and report["no_val_test_open"] and report["execution_probe_only"]
            and not report["scientific_pilot"] and report["arm"] == "off" and report["recipe_batch_size"] == 8,
            "completed OFF throughput authority required")
    entries = manifest["source_files"]
    require(hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            == manifest["source_sha256"] == report["identities"]["source_sha256"] == SOURCE_SHA, "source inventory mismatch")
    for entry in entries:
        relative = Path(entry["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "invalid source inventory path")
        require(len(read(dict(path=str(source / relative), sha256=entry["sha256"]))) == entry["size"], "source size mismatch")
    observed = list(source.rglob("*"))
    require(not any(value.is_symlink() for value in observed)
            and {value.relative_to(source).as_posix() for value in observed if value.is_file()} == {entry["path"] for entry in entries},
            "undeclared source file or symlink")
    protected = list(manifest["paths"].values()) + [args.request, args.report]
    protected += [value["path"] for value in manifest["authorities"].values()]
    protected += [value["path"] for value in request.values() if isinstance(value, dict) and "path" in value]
    protected += [value for name, value in request.items() if name.endswith("_root")]
    require(not output.exists(), "exclusive new output directory required")
    for value in protected:
        root = path(value)
        require(not output.is_relative_to(root) and not root.is_relative_to(output), "output overlaps immutable inputs")
    require(not any(name == "refer_or_clear" or name.startswith("refer_or_clear.") for name in sys.modules), "scientific package already imported")
    require(os.environ.get("OMP_NUM_THREADS") == os.environ.get("MKL_NUM_THREADS") == "4"
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "original thread/determinism environment required")
    sys.dont_write_bytecode = True
    benchmark = load_script(source / "scripts/benchmark_direct_reference_throughput.py")
    benchmark.allocation_guard()
    staging = load_script(source / "scripts/direct_reference_job.py")
    require(staging.source_digest(source) == SOURCE_SHA, "frozen source byte mismatch")
    api = benchmark.runtime()
    context = api.diagnostic.load_context(args.request, REQUEST_SHA, run, "cuda")
    require(context["identities"] == report["identities"] and context["manifest"] == manifest, "context/report binding mismatch")
    return api, context, report


def measure(context, prepared, selected, api):
    require(len(selected) == 48 and selected.role.eq("train").all() and not selected.image_id.duplicated().any(), "48 unique TRAIN IDs required")
    model = api.diagnostic._model(context, False)
    optimizer = api.pilots.make_pilot_optimizer(model, {"track": "D_running_pilot"}, "train")
    scheduler = api.pilots.pilot_scheduler(optimizer, "D_running_pilot", 22200)
    started = time.perf_counter()
    dataset = api.diagnostic._dataset(context, prepared, selected, False, augmented=True)
    result = dict(constructor_seconds=time.perf_counter() - started, getter_seconds=0.0, gpu_step_seconds=0.0,
                  end_to_end_seconds=0.0, warmup_images=16, timed_images=32, optimizer_steps=0)
    profile = cProfile.Profile()
    for batch_index in range(6):
        api.torch.cuda.synchronize()
        started = getter_started = time.perf_counter()
        samples = []
        for index in range(batch_index * 8, (batch_index + 1) * 8):
            if batch_index >= 2: profile.enable()
            try:
                samples.append(dataset[index])
            finally:
                profile.disable()
        getter_seconds = time.perf_counter() - getter_started
        batch = api.torch.utils.data.default_collate(samples)
        require(batch["meta"]["image_id"] == selected.image_id.iloc[batch_index * 8:(batch_index + 1) * 8].tolist()
                and batch["meta"]["role"] == ["train"] * 8 and list(batch["query"].shape) == [8, 4, 512, 800], "TRAIN batch identity/shape mismatch")
        gpu_started = time.perf_counter()
        stats = api.core.train_direct_reference_epoch(model, (batch,), optimizer, scheduler, context["device"])
        api.torch.cuda.synchronize()
        ended = time.perf_counter()
        require(stats["n_images"] == 8 and stats["optimizer_steps"] == 1, "one original batch-8 update required")
        result["optimizer_steps"] += 1
        if batch_index >= 2:
            result["getter_seconds"] += getter_seconds
            result["gpu_step_seconds"] += ended - gpu_started
            result["end_to_end_seconds"] += ended - started
    result.update(input_shape=[8, 4, 512, 800], rng_final_sha256={name: hashlib.sha256(repr(getattr(dataset, name).getstate()).encode()).hexdigest()
                                                             for name in ("geometry_rng", "photometry_rng")})
    return result, profile


def run(args):
    script = dict(path=str(Path(__file__).absolute()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    api, context, report = bootstrap(args)
    require(os.environ.get("OMP_NUM_THREADS") == os.environ.get("MKL_NUM_THREADS") == "4"
            and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "original thread/determinism environment required")
    api.torch.set_num_threads(4)
    api.torch.set_num_interop_threads(1)
    api.cv2.setNumThreads(4)
    authorities = context["manifest"]["authorities"]
    rows = api.load_role_rows(BytesIO(api.bridge._bytes(authorities["records"])), BytesIO(api.bridge._bytes(authorities["split"])), "train").sort_values("image_id").reset_index(drop=True)
    require(len(rows) == 1778 and rows.role.eq("train").all() and not rows.image_id.duplicated().any()
            and report["selected_train_ids"] == rows.image_id.iloc[:128].tolist(), "original official TRAIN selection required")
    preprocess = api.diagnostic._preprocess(context, report["phase_authority"])
    enrollment = api.bridge._json(preprocess["enrollment_authority"])
    started = time.perf_counter()
    table = api.bridge.load_registration_table(report["registration"]["path"], report["registration"]["sha256"], rows=rows,
                                              enrollment_manifest=enrollment, preprocess=preprocess)
    table_seconds = time.perf_counter() - started
    prepared = dict(preprocess=preprocess, enrollment=enrollment, table=table, table_binding=report["registration"])
    selected = rows.iloc[:48].copy()
    output = path(args.output_root)
    output.mkdir(parents=True, exist_ok=False)
    with api.diagnostic.pixel_audit(context, selected) as opened:
        result, profile = measure(context, prepared, selected, api)
    require([entry["image_id"] for entry in opened] == selected.image_id.tolist()
            and all(entry["role"] == "train" for entry in opened), "getter decode ledger mismatch")
    refreshed = api.diagnostic.load_context(args.request, REQUEST_SHA, args.run_dir, "cuda")
    require(refreshed["identities"] == context["identities"] and refreshed["manifest"] == context["manifest"], "authority drift during profile")
    for authority in (script, dict(path=args.report, sha256=REPORT_SHA), dict(path=str(context["run_dir"] / "input_manifest.json"), sha256=MANIFEST_SHA), report["registration"], report["phase_authority"]): read(authority)
    stats = pstats.Stats(profile)
    functions = [dict(file=key[0], line=key[1], function=key[2], primitive_calls=value[0], calls=value[1], self_seconds=value[2], cumulative_seconds=value[3]) for key, value in stats.stats.items()]
    result.update(schema_version="external_direct_reference_getter_profile_v1", completed=True, execution_probe_only=True, scientific_pilot=False,
                  arm="off", instrumentation=script, identities=context["identities"], request=context["request"],
                  original_report=dict(path=args.report, sha256=REPORT_SHA), input_manifest_sha256=MANIFEST_SHA,
                  registration=report["registration"], phase_authority=report["phase_authority"], table_load_seconds=table_seconds,
                  registration_population=1778, query_reads=opened, no_val_test_open=True, selected_train_ids=selected.image_id.tolist(),
                  allocation={name: os.environ.get(name) for name in ("SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK")},
                  query_reads_scope="query_decode_helper_reads, not staging bytes or syscall audit",
                  top_cumulative=sorted(functions, key=lambda value: value["cumulative_seconds"], reverse=True)[:30],
                  top_self=sorted(functions, key=lambda value: value["self_seconds"], reverse=True)[:30],
                  path_checks=[value for value in functions if value["function"] in ("_fingerprint", "_path", "resolve", "is_symlink", "stat", "lstat")],
                  timing_scope="getter-only cProfile; GPU step includes transfer/host guards/backward/clip/optimizer/scheduler; end-to-end includes profiler overhead; no checkpoint or replay")
    profile.dump_stats(str(output / "raw.pstats"))
    result["raw_pstats"] = dict(path=str(output / "raw.pstats"), sha256=hashlib.sha256((output / "raw.pstats").read_bytes()).hexdigest())
    api.diagnostic.write_receipt(output / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source-dir", "request", "request-sha256", "run-dir", "report", "report-sha256", "output-root"):
        parser.add_argument("--" + name, required=True)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
