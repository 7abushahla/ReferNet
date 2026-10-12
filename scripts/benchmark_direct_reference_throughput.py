"""Bounded OFF execution probes, never a pilot, validation, or batch selection."""
from __future__ import annotations

import argparse
import gc
from io import BytesIO
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
from types import SimpleNamespace


ENVIRONMENT = "/shared/b00090279/conda/envs/refer-or-clear"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def runtime():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import cv2
    import torch
    from refer_or_clear import direct_reference_diagnostic as diagnostic
    from refer_or_clear import direct_reference_pilot as core
    from refer_or_clear import pilots
    from refer_or_clear.data import direct_reference as bridge
    from refer_or_clear.data.query import load_role_rows
    return SimpleNamespace(torch=torch, cv2=cv2, diagnostic=diagnostic, core=core,
                           pilots=pilots, bridge=bridge, load_role_rows=load_role_rows)


def allocation_guard():
    node = os.environ.get("SLURMD_NODENAME", "").lower()
    require(node and not any(name in node for name in ("login", "head", "frontend", "lgn")), "allocated compute node required")
    require(os.environ.get("SLURM_JOB_PARTITION") == "gpu", "GPU partition required")
    job = os.environ.get("SLURM_JOB_ID", "")
    require(re.fullmatch(r"[0-9]+", job) and int(job) > 0, "allocated Slurm job required")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "")
    require(cpus.isdigit() and int(cpus) >= 4, "four allocated CPUs required")
    counts = [os.environ.get(name, "") for name in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE")]
    devices = [os.environ.get(name, "") for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")]
    require(any(value.isdigit() and int(value) > 0 for value in counts)
            or any(value not in ("", "-1", "NoDevFiles") for value in devices), "allocated GPU required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1", "NoDevFiles"), "visible allocated GPU required")
    require(os.environ.get("CONDA_PREFIX") == ENVIRONMENT and sys.prefix == ENVIRONMENT
            and Path(sys.executable).parent == Path(ENVIRONMENT) / "bin", "exact project interpreter required")


def prepare(context, directory, api):
    records = api.bridge._bytes(context["manifest"]["authorities"]["records"])
    split = api.bridge._bytes(context["manifest"]["authorities"]["split"])
    rows = api.load_role_rows(BytesIO(records), BytesIO(split), "train").sort_values("image_id").reset_index(drop=True)
    require(len(rows) == 1778 and rows.role.eq("train").all() and not rows.image_id.duplicated().any(), "official 1778 TRAIN rows required")
    phase = dict(schema_version="direct_reference_phase_v1", phase="overfit", **context["identities"],
                 authorized_image_ids=dict(train=rows.image_id.tolist(), val=[]))
    directory.mkdir(parents=True, exist_ok=False)
    phase_binding = api.diagnostic.write_receipt(directory / "phase-authority.json", phase)
    preprocess = api.diagnostic._preprocess(context, phase_binding)
    enrollment = api.bridge._json(preprocess["enrollment_authority"])
    table = api.bridge.build_registration_table(rows, Path(context["manifest"]["paths"]["raw_root"]),
                                               enrollment, preprocess=preprocess)
    table_binding = api.diagnostic.write_receipt(directory / "registration.json", table)
    return dict(preprocess=preprocess, enrollment=enrollment, table=table, table_binding=table_binding), rows


def _loader(context, prepared, rows, batch_size, batches, api):
    dataset = api.diagnostic._dataset(context, prepared, rows, False, augmented=True)
    indices = [index % len(rows) for index in range(batch_size * batches)]
    selected = api.torch.utils.data.Subset(dataset, indices)
    return api.torch.utils.data.DataLoader(selected, batch_size=batch_size, shuffle=False, drop_last=True,
                                         num_workers=0, pin_memory=False,
                                         generator=api.torch.Generator().manual_seed(context["config"]["seed"] + 2001))


def _check_batch(batch, rows, batch_size, context):
    identifiers = batch["meta"]["image_id"]
    require(len(identifiers) == batch_size and all(identifier in set(rows.image_id) for identifier in identifiers), "batch IDs outside selected TRAIN rows")
    require(batch["meta"]["role"] == ["train"] * batch_size, "only TRAIN batches permitted")
    require(list(batch["query"].shape) == [batch_size, *context["config"]["input_shape"]], "effective input shape mismatch")
    return list(identifiers)


def measure_candidate(context, prepared, rows, batch_size, warmup, steps, api):
    require(rows.role.eq("train").all() and not rows.image_id.duplicated().any(), "only unique TRAIN rows permitted")
    require(batch_size in (8, 16, 32) and len(rows) >= batch_size, "bounded candidate batch required")
    require(type(warmup) is int and 1 <= warmup <= 5 and type(steps) is int and 1 <= steps <= 16, "bounded warmup/timing required")
    result = dict(batch_size=batch_size, execution_probe_only=True, status="completed", modes={})
    initial_hash = None
    for mode in ("loader_only", "gpu_only", "end_to_end"):
        model = optimizer = scheduler = gpu_batch = iterator = batch = first = loader = None
        try:
            api.diagnostic._seed(context["config"]["seed"])
            loader = _loader(context, prepared, rows, batch_size, warmup + steps, api)
            iterator = iter(loader)
            if mode != "loader_only":
                model = api.diagnostic._model(context, False)
                state_hash = api.pilots.tensor_sha256(model.state_dict())
                require(initial_hash is None or initial_hash == state_hash, "candidate modes did not reset fresh model")
                initial_hash = state_hash
                optimizer = api.pilots.make_pilot_optimizer(model, {"track": "D_running_pilot"}, "train")
                require(not optimizer.state, "fresh optimizer required")
                scheduler = api.pilots.pilot_scheduler(optimizer, "D_running_pilot", 22200)
            if mode == "gpu_only":
                first = next(iterator)
                _check_batch(first, rows, batch_size, context)
                gpu_batch = {key: value.to(context["device"]) if isinstance(value, api.torch.Tensor) else value
                             for key, value in first.items()}
            api.torch.cuda.synchronize()
            api.torch.cuda.reset_peak_memory_stats()
            warmup_ids, timed_ids, updates = [], [], 0
            started = None
            for index in range(warmup + steps):
                if index == warmup:
                    api.torch.cuda.synchronize()
                    started = time.perf_counter()
                batch = gpu_batch if mode == "gpu_only" else next(iterator)
                identifiers = _check_batch(batch, rows, batch_size, context)
                (warmup_ids if index < warmup else timed_ids).extend(identifiers)
                if mode != "loader_only":
                    stats = api.core.train_direct_reference_epoch(model, (batch,), optimizer, scheduler, context["device"])
                    require(stats["n_images"] == batch_size and stats["optimizer_steps"] == 1, "update/clip accounting mismatch")
                    updates += stats["optimizer_steps"]
            api.torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            require(math.isfinite(seconds) and seconds > 0, "finite positive timing required")
            measured = dict(seconds=seconds, images_per_second=batch_size * steps / seconds,
                            warmup_batches=warmup, timed_batches=steps, timed_images=batch_size * steps,
                            warmup_image_ids=warmup_ids, timed_image_ids=timed_ids, optimizer_steps=updates,
                            clip_norm=None if mode == "loader_only" else 5.0,
                            input_shape=[batch_size, *context["config"]["input_shape"]],
                            peak_allocated_bytes=api.torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=api.torch.cuda.max_memory_reserved(),
                            timing_scope="synchronized wall time; GPU mode excludes loading and transfer but retains host guards")
            if mode != "loader_only":
                measured.update(initial_state_sha256=state_hash, scheduler_steps=scheduler.last_epoch)
            result["modes"][mode] = measured
        except api.torch.cuda.OutOfMemoryError as error:
            result.update(status="cuda_oom", oom_mode=mode, error=str(error)[:500])
            break
        finally:
            model = optimizer = scheduler = gpu_batch = iterator = batch = first = loader = None
            gc.collect()
            api.torch.cuda.empty_cache()
    return result


def run(args):
    allocation_guard()
    api = runtime()
    api.torch.set_num_threads(args.torch_threads)
    api.torch.set_num_interop_threads(1)
    context = api.diagnostic.load_context(args.request, args.request_sha256, args.run_dir, args.device)
    precision = api.pilots.configure_precision(context["device"], "D_running_pilot")
    output = api.bridge._path(str(context["run_dir"] / "throughput"))
    require(not output.exists(), "new throughput directory required")
    manifest = context["manifest"]
    protected = [manifest["paths"][name] for name in ("source_dir", "raw_root", "product_root", "stage_dir")]
    protected += [binding["path"] for binding in manifest["authorities"].values()]
    for path in protected:
        root = api.bridge._path(path)
        require(not output.is_relative_to(root) and not root.is_relative_to(output), "throughput output overlaps authenticated inputs")
    records = api.bridge._bytes(manifest["authorities"]["records"])
    split = api.bridge._bytes(manifest["authorities"]["split"])
    allowed = api.load_role_rows(BytesIO(records), BytesIO(split), "train")
    require(len(allowed) == 1778 and allowed.role.eq("train").all(), "official TRAIN population required")
    report = dict(schema_version="direct_reference_throughput_v1", execution_probe_only=True,
                  scientific_pilot=False, arm="off", recipe_batch_size=8, num_workers=0, pin_memory=False,
                  request=context["request"], identities=context["identities"],
                  precision=precision, candidates=[],
                  environment=dict(python=sys.version, interpreter=sys.executable, prefix=sys.prefix,
                                   torch=api.torch.__version__, cuda=api.torch.version.cuda,
                                   cudnn=api.torch.backends.cudnn.version(), opencv=api.cv2.__version__,
                                   torch_threads=api.torch.get_num_threads(), interop_threads=api.torch.get_num_interop_threads(),
                                   opencv_threads=api.cv2.getNumThreads(), host=platform.node(),
                                   gpu=api.torch.cuda.get_device_name(0), gpu_total_bytes=api.torch.cuda.get_device_properties(0).total_memory,
                                   allocation={name: os.environ.get(name) for name in
                                               ("SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK")},
                                   thread_environment={name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}))
    with api.diagnostic.pixel_audit(context, allowed) as opened:
        started = time.perf_counter()
        prepared, rows = prepare(context, output, api)
        report.update(registration=prepared["table_binding"], phase_authority=prepared["preprocess"]["phase_authority"],
                      registration_seconds=time.perf_counter() - started, registration_population=len(rows))
        require(len(opened) == 1778 and {entry["image_id"] for entry in opened} == set(rows.image_id)
                and all(entry["role"] == "train" for entry in opened), "registration decode audit must contain all TRAIN queries only")
        selected = rows.iloc[:args.sample_count].copy()
        report["selected_train_ids"] = selected.image_id.tolist()
        oom = False
        for batch_size in args.batch_sizes:
            if oom:
                report["candidates"].append(dict(batch_size=batch_size, status="skipped_after_cuda_oom", execution_probe_only=True))
                continue
            candidate = measure_candidate(context, prepared, selected, batch_size, args.warmup_batches, args.timed_batches, api)
            report["candidates"].append(candidate)
            oom = candidate["status"] == "cuda_oom"
        require(all(entry["role"] == "train" and entry["image_id"] in set(rows.image_id) for entry in opened), "forbidden timing query read")
        report.update(query_reads=opened, query_reads_scope="query_decode_helper_reads", no_val_test_open=True)
    refreshed = api.diagnostic.load_context(args.request, args.request_sha256, args.run_dir, args.device)
    require(refreshed["identities"] == context["identities"], "source/input identity changed during benchmark")
    api.bridge._bytes(prepared["table_binding"])
    api.bridge._bytes(prepared["preprocess"]["phase_authority"])
    baseline = next((candidate for candidate in report["candidates"] if candidate["batch_size"] == 8), None)
    report["completed"] = baseline is not None and baseline["status"] == "completed"
    if report["completed"]:
        speed = baseline["modes"]["end_to_end"]["images_per_second"]
        report["estimator"] = dict(fixed_batch8_training_epoch_seconds=1776 / speed,
                                   excludes=["startup_registration", "validation", "checkpoint_io", "selected_replay"],
                                   scope="bounded TRAIN sample extrapolation, not a completion promise or batch recommendation")
    api.diagnostic.write_receipt(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--timed-batches", type=int, default=4)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, choices=(1, 4), default=4)
    args = parser.parse_args(argv)
    if args.batch_sizes != sorted(set(args.batch_sizes)) or 8 not in args.batch_sizes or any(value not in (8, 16, 32) for value in args.batch_sizes):
        parser.error("ordered unique candidates from 8/16/32 must include baseline 8")
    if not 1 <= args.warmup_batches <= 5 or not 1 <= args.timed_batches <= 16 or not 64 <= args.sample_count <= 256:
        parser.error("bounded warmup 1..5, timing 1..16, TRAIN sample count 64..256 required")
    if not re.fullmatch(r"[0-9a-f]{64}", args.request_sha256):
        parser.error("lowercase request SHA256 required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(0 if main()["completed"] else 1)
