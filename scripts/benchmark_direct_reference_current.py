"""Uncached, validation-200, original DirectReference OFF/ON CPU inference.

Externally hash-bound direct_reference_current_cpu_v1 requests declare
source_root, evaluation_source_sha256, scientific_source_hashes (all 34 original
scientific Python files), overlay_source_hashes (every additional src Python
file), source_selection (records/split/enrollment/weights/config/arms), image_ids,
and output_root. Embedded original initialization, registration, phase, product
and receipt paths must be restored unchanged. No A-test authority is fabricated.
Both completed selected parents are validated before opening query pixels.
The exact first 200 sorted official validation IDs use their existing full-bank
entries and original path/source-bound RANSAC seeds. Live fits must reproduce
those saved entries. No augmentation, application reference cache or learning.
Neural timing is nested in each contiguous end-to-end observation and includes
input validation and scalar/referral materialization, not just model.forward.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = Path('/shared/b00090279/conda/envs/refer-or-clear')
FIXED = dict(schema_version='direct_reference_current_cpu_v1', protocol='A', fold=0,
             role='val', bank_route='enrollment', id_selection='first_200_sorted_validation_ids',
             arm_order=['off', 'on'], reference_cache='disabled', precision='float32',
             batch_size=1, threads=1, warmups=20, timed_documents=200, timed_passes=1,
             device='cpu', workers=0, execution='eager_inference_mode', augmentation='disabled')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute(value):
    require(isinstance(value, (str, Path)), 'absolute path required')
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts
            and not any(item.is_symlink() for item in (path, *path.parents)), 'absolute non-symlink path required')
    return path


def file_sha(path):
    with absolute(path).open('rb') as stream:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
        return checksum.hexdigest()


def content(binding):
    require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}, 'path/SHA256 binding required')
    checksum = binding['sha256']
    require(isinstance(checksum, str) and len(checksum) == 64
            and all(character in '0123456789abcdef' for character in checksum), 'lowercase SHA256 required')
    payload = absolute(binding['path']).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == checksum, 'artifact SHA256 mismatch')
    return payload


def document(binding):
    def pairs(items):
        value = {}
        for name, item in items:
            require(name not in value, 'duplicate JSON key')
            value[name] = item
        return value
    value = json.loads(content(binding), object_pairs_hook=pairs)
    json.dumps(value, allow_nan=False)
    require(isinstance(value, dict), 'JSON object required')
    return value


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, absolute(path))
    require(spec is not None and spec.loader is not None, 'script import unavailable')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_scientific_files(source, request):
    source = absolute(source)
    require(not any(path for directory in ('src', 'scripts') for path in (source / directory).rglob('*.pyc')),
            'unbound scientific/tool bytecode forbidden')
    original = request['scientific_source_hashes']
    overlays = request['overlay_source_hashes']
    require(isinstance(original, dict) and len(original) == 34, 'all 34 original scientific files required')
    require(isinstance(overlays, dict) and not set(original) & set(overlays), 'disjoint source overlays required')
    expected = dict(original, **overlays)
    observed = {path.relative_to(source).as_posix() for path in (source / 'src').rglob('*.py')}
    require(observed == set(expected), 'unbound or missing scientific Python file')
    for relative, checksum in expected.items():
        require(relative.startswith('src/refer_or_clear/') and relative.endswith('.py')
                and '..' not in Path(relative).parts, 'invalid scientific source path')
        require(file_sha(source / relative) == checksum, 'scientific file SHA256 mismatch')


def verify_imports(source, request):
    expected = dict(request['scientific_source_hashes'], **request['overlay_source_hashes'])
    for name, module in tuple(sys.modules.items()):
        if name == 'refer_or_clear' or name.startswith('refer_or_clear.'):
            location = getattr(module, '__file__', None)
            require(location is not None, 'scientific import origin unavailable')
            path = absolute(location)
            require(path.is_relative_to(source), 'foreign scientific import')
            relative = path.relative_to(source).as_posix()
            require(relative in expected and file_sha(path) == expected[relative], 'unbound scientific import')
            for search in getattr(module, '__path__', ()):
                require(absolute(search).is_relative_to(source / 'src'), 'foreign scientific package path')


def load_runtime_request(binding):
    request = document(binding)
    fields = {'source_root', 'evaluation_source_sha256', 'scientific_source_hashes',
              'overlay_source_hashes', 'source_selection', 'image_ids', 'output_root'}
    require(set(request) == set(FIXED) | fields, 'strict runtime request fields required')
    require(all(type(request[name]) is type(value) and request[name] == value for name, value in FIXED.items()),
            'fixed validation-200 uncached CPU contract mismatch')
    require(absolute(request['source_root']) == ROOT, 'runner must be in the authenticated isolated source tree')
    verify_scientific_files(ROOT, request)
    require(not any(name == 'refer_or_clear' or name.startswith('refer_or_clear.') for name in sys.modules),
            'scientific package imported before source authentication')
    sys.path.insert(0, str(ROOT / 'src'))
    sys.path.insert(0, str(ROOT / 'scripts'))
    api = load_script('_direct_reference_cpu_a_authorities', ROOT / 'scripts/score_protocol_a_current.py')
    api.validate_source(dict(evaluation_source_sha256=request['evaluation_source_sha256'],
                             critical_source_hashes=request['scientific_source_hashes']))
    verify_imports(ROOT, request)
    selection = request['source_selection']
    require(isinstance(selection, dict) and set(selection) == {'records', 'split', 'enrollment', 'weights', 'config', 'arms'},
            'ordinary original A source-selection bundle required')
    for name, checksum in (('records', api.RECORDS_SHA256), ('split', api.SPLIT_SHA256),
                           ('weights', api.WEIGHTS_SHA256), ('enrollment', api.consumer.ENROLLMENT_SHA256)):
        require(selection[name].get('sha256') == checksum, 'original A metadata/weights identity mismatch')
        api.authenticate(selection[name])
    api.authenticate(selection['config'])
    require(set(selection['arms']) == {'off', 'on'}, 'both selected parents required')
    for authorities in selection['arms'].values():
        require(set(authorities) == {'train_result', 'completion', 'fresh_replay'}, 'complete selected parent bindings required')
    config = api.yaml.safe_load(api.content(selection['config']))
    return request, api, config


def measurement_rows(rows):
    require(len(rows) == 222 and rows.role.eq('val').all() and rows.image_id.is_unique,
            'complete 222 source validation population required')
    return rows.sort_values('image_id').iloc[:200].reset_index(drop=True)


def validate_parents(api, selection, config, val_rows, train_rows):
    selected = {arm: api.validate_arm(selection, config, arm, val_rows, train_rows) for arm in ('off', 'on')}
    for name in ('source_sha256', 'weights_sha256', 'records_sha256', 'split_sha256', 'enrollment_sha256',
                 'registration_sha256', 'phase_authority_sha256', 'preprocess_sha256'):
        require(selected['off']['identities'][name] == selected['on']['identities'][name], 'matched arm identity mismatch')
    require(selected['off']['result']['fresh_initial_state_sha256'] == selected['on']['result']['fresh_initial_state_sha256'],
            'matched initial state required')
    require(selected['off']['authority']['diagnostic_request_sha256'] == selected['on']['authority']['diagnostic_request_sha256']
            and selected['off']['result']['diagnostic_gate'] == selected['on']['result']['diagnostic_gate'],
            'matched diagnostic authority required')
    return selected


def prepare_live(direct, row, entry, raw_root, enabled):
    import cv2
    import numpy as np
    import torch
    products = entry['products']
    target, masks = direct._common(products)
    raw = direct._image(dict(path=str(raw_root / row['path']), sha256=entry['query_sha256']), cv2.IMREAD_COLOR)
    query, valid, _ = direct.to_canvas(raw)
    inliers = 0
    try:
        matrix, inliers = direct._fit(direct._features(query), direct._features(target), int(entry['key'][:8], 16) & 0x7fffffff)
        require(matrix is not None and type(inliers) is int and inliers >= direct.MIN_INLIERS, 'registration failed')
        matrix = direct._similarity(matrix)
        _, fitted_overlap = direct._overlap(valid, masks, matrix)
        reason = 'fit' if direct._available(fitted_overlap) else 'empty_overlap'
    except (ValueError, cv2.error, np.linalg.LinAlgError):
        matrix, reason = None, 'registration_failed'
        inliers = inliers if type(inliers) is int and inliers >= 0 else 0
    status = 'fit' if reason == 'fit' else 'failed'
    require(status == entry['status'] and reason == entry['reason'] and inliers == entry['inliers'],
            'live fit differs from saved registration')
    if status == 'fit':
        require(np.array_equal(matrix, np.asarray(entry['query_canvas_to_bank'])), 'live transform differs from saved registration')
    overlap, zone = np.zeros((512, 800), np.uint8), np.zeros((512, 800), np.uint8)
    reference = None
    if status == 'fit':
        inverse, overlap = direct._overlap(valid, masks, matrix)
        zone = (direct._warp(masks['zone'], inverse, True) > 0).astype(np.uint8) * overlap
        if enabled and direct._available(overlap):
            median = direct._image(products['median'], cv2.IMREAD_COLOR)
            require(median.shape == (512, 800, 3), 'median shape mismatch')
            reference = direct._warp(median, inverse)
    available = direct._available(overlap)
    if not available:
        zone.fill(0)
        overlap.fill(0)
    zone *= overlap
    batch = dict(query=torch.from_numpy(np.ascontiguousarray(np.concatenate([
        direct._normalize(query), zone[None].astype(np.float32)]))).unsqueeze(0),
        overlap=torch.from_numpy(overlap[None].astype(np.float32)).unsqueeze(0),
        context_available=torch.tensor([available], dtype=torch.bool), label=torch.tensor([int(row['label'])]))
    if enabled:
        batch['reference'] = (torch.from_numpy(np.ascontiguousarray(direct._normalize(reference))).unsqueeze(0)
                              if available else torch.zeros((1, 3, 512, 800), dtype=torch.float32))
    return batch, dict(status=status, reason=reason, inliers=inliers, registration_key=entry['key'],
                       context_available=available, referral_required=not available)


def infer(core, torch, model, batch):
    require(not model.training, 'stored-statistics eval mode required')
    core._strict_fp32(model, 'cpu')
    with torch.inference_mode():
        inputs, _ = core._batch_inputs(model, batch, 'cpu')
        outputs = core._forward(model, inputs)
        score = float((1 - torch.sigmoid(outputs['attack_logit']))[0].item())
        available = bool(batch['context_available'][0].item())
    require(math.isfinite(score) and 0 <= score <= 1, 'invalid materialized score')
    decision = ('genuine' if score >= .5 else 'attack') if available else 'refer'
    return dict(bonafide_score=score, decision=decision, referral_required=not available)


def measure_cell(ids, prepare, predict, *, clock=time.perf_counter):
    require(len(ids) == len(set(ids)) == 200 and ids == sorted(ids), 'exact sorted 200 IDs required')
    for identifier in ids[:20]:
        batch, _ = prepare(identifier)
        predict(batch)
    observations = []
    for identifier in ids:
        start = clock()
        batch, geometry = prepare(identifier)
        prepared = clock()
        score = predict(batch)
        end = clock()
        require(all(math.isfinite(value) for value in (start, prepared, end))
                and start <= prepared <= end, 'nonfinite or nonmonotonic clock')
        observations.append(dict(image_id=identifier, preparation_seconds=prepared - start,
            neural_seconds=end - prepared, end_to_end_seconds=end - start, geometry=geometry, **score))
    return observations


def timing_summary(rows):
    def summarize(items):
        values = sorted(items)
        return dict(n=len(values), median_seconds=statistics.median(values),
                    p95_seconds=values[math.ceil(.95 * len(values)) - 1], p95_method='nearest_rank')
    summary = {name: summarize([row[name] for row in rows]) for name in
               ('preparation_seconds', 'neural_seconds', 'end_to_end_seconds')}
    summary['availability_strata'] = {}
    for available in (True, False):
        subset = [row['end_to_end_seconds'] for row in rows if row['geometry']['context_available'] is available]
        summary['availability_strata']['available' if available else 'unavailable'] = (
            summarize(subset) if subset else dict(n=0, status='unmeasured'))
    return summary


def fingerprint(path):
    stat = absolute(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class IntegrityGuard:
    def __init__(self, bindings):
        self.bindings = {}
        for item in bindings:
            path = item['path']
            require(path not in self.bindings or self.bindings[path] == item, 'conflicting artifact bindings')
            self.bindings[path] = dict(item)
        self.fingerprints = {}
        for path, item in self.bindings.items():
            before = fingerprint(path)
            require(file_sha(path) == item['sha256'], 'protected artifact SHA256 mismatch')
            require(before == fingerprint(path), 'artifact changed during authentication')
            self.fingerprints[path] = before

    def check(self):
        for path, before in self.fingerprints.items():
            require(fingerprint(path) == before, 'protected artifact changed during benchmark')

    def finish(self):
        self.check()
        for path, item in self.bindings.items():
            require(file_sha(path) == item['sha256'], 'protected artifact SHA256 drift')
        self.check()


def allocation():
    node = os.environ.get('SLURMD_NODENAME', '').lower()
    job = os.environ.get('SLURM_JOB_ID', '')
    require(job.isdecimal() and int(job) > 0 and node and os.environ.get('SLURM_JOB_PARTITION') == 'cpu'
            and not any(token in node for token in ('login', 'head', 'frontend', 'lgn')), 'allocated CPU compute node required')
    require(all(os.environ.get(name, '0') == '0' for name in ('SLURM_GPUS', 'SLURM_GPUS_ON_NODE'))
            and not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('SLURM_STEP_GPUS')
            and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'GPU allocation/access forbidden')
    require(Path(sys.prefix).resolve() == ENVIRONMENT.resolve()
            and Path(os.environ.get('CONDA_PREFIX', '')).resolve() == ENVIRONMENT.resolve(), 'exact interpreter/environment prefix required')
    require(os.environ.get('OMP_NUM_THREADS') == os.environ.get('MKL_NUM_THREADS') == '1', 'one-thread environment required')
    count = os.environ.get('SLURM_CPUS_PER_TASK', '')
    require(count.isdecimal() and int(count) >= 4, 'four allocated CPUs required')


def host_info(torch, cv2):
    cpuinfo = Path('/proc/cpuinfo').read_text()
    model = next((line.split(':', 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith('model name')), '')
    require(bool(model), 'named CPU hardware required')
    return dict(hostname=platform.node(), cpu_model=model,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        affinity=sorted(os.sched_getaffinity(0)), job_id=os.environ['SLURM_JOB_ID'],
        node=os.environ['SLURMD_NODENAME'], allocated_cpus=int(os.environ['SLURM_CPUS_PER_TASK']),
        python=sys.version, executable=sys.executable, prefix=sys.prefix, os=platform.platform(),
        torch=str(torch.__version__), opencv=cv2.__version__, torch_intraop=torch.get_num_threads(),
        torch_interop=torch.get_num_interop_threads(), opencv_threads=cv2.getNumThreads(),
        environment_threads={name: os.environ.get(name) for name in
                             ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})


def rng_sha(torch, np):
    import pickle
    return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().tolist()))).hexdigest()


def run(binding, output_path):
    allocation()
    started = time.perf_counter()
    request, api, config = load_runtime_request(binding)
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from refer_or_clear.data import direct_reference as direct
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    host = host_info(torch, cv2)
    require(host['torch_intraop'] == host['torch_interop'] == host['opencv_threads'] == 1, 'one-thread execution required')
    selection = request['source_selection']
    roles = {role: api.load_role_rows(Path(selection['records']['path']), Path(selection['split']['path']), role)
             for role in ('train', 'val')}
    rows = measurement_rows(roles['val'])
    ids = rows.image_id.tolist()
    require(request['image_ids'] == ids, 'request must freeze first 200 sorted validation IDs')
    selected = validate_parents(api, selection, config, roles['val'], roles['train'])
    protected = [binding, *api.all_bindings(selection)]
    for arm in ('off', 'on'):
        protected.extend(api.all_bindings(selected[arm]['result']))
        protected.extend(api.all_bindings(selected[arm]['authority']))
        for name in ('completion', 'fresh_replay'):
            protected.extend(api.all_bindings(document(selection['arms'][arm][name])))
    registration = selected['off']['result']['registration']
    saved_document = document(registration)
    preprocess = saved_document['preprocess']
    require(preprocess['enrollment_authority'] == selection['enrollment']
            and preprocess['records_authority']['sha256'] == selection['records']['sha256']
            and preprocess['split_authority']['sha256'] == selection['split']['sha256'], 'original registration metadata mismatch')
    protected.extend(api.all_bindings(preprocess))
    manifest = document(selection['enrollment'])
    raw_root = absolute(preprocess['raw_root'])
    table = direct.load_registration_table(Path(registration['path']), registration['sha256'], rows=rows,
                                         enrollment_manifest=manifest, preprocess=preprocess)
    for identifier in ids:
        entry = saved_document['entries'][identifier]
        require(entry['role'] == 'val' and entry['bank'] == 'enrollment', 'validation full-bank entries required')
        protected.extend(entry['products'].values())
    output = absolute(output_path)
    require(str(output) == request['output_root'] and not output.exists(), 'new bound output directory required')
    roots = (ROOT, raw_root, absolute(manifest['output_root']), absolute(direct._json(preprocess['product_root_authority'])['relocated_root']))
    require(all(not output.is_relative_to(path) and not path.is_relative_to(output) for path in roots), 'output overlaps input root')
    require(all(not output.is_relative_to(absolute(item['path'])) and not absolute(item['path']).is_relative_to(output)
                for item in protected), 'output overlaps protected artifact')
    guard = IntegrityGuard(protected)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version='direct_reference_current_cpu_result_v1', status='incomplete', comparable=False,
                  contract=FIXED, request=binding, host=host, image_ids=ids,
                  image_ids_sha256=direct._digest(ids), arms={})
    try:
        records = {row['image_id']: row for row in rows.to_dict('records')}
        models = {arm: api.fresh_model(selection, selected[arm], arm, device='cpu').eval() for arm in ('off', 'on')}
        before = {arm: api.pilots.tensor_sha256(model.state_dict()) for arm, model in models.items()}
        rng = rng_sha(torch, np)
        for arm in ('off', 'on'):
            enabled = arm == 'on'
            dataset = direct.DirectReferenceDataset(rows, raw_root, manifest, table,
                                                    reference_enabled=enabled, train=False, augment={})
            for index, identifier in enumerate(ids):
                guard.check()
                live, _ = prepare_live(direct, records[identifier], saved_document['entries'][identifier], raw_root, enabled)
                saved = default_collate([dataset[index]])
                for key in ('query', 'overlap', 'context_available', 'label', *(['reference'] if enabled else [])):
                    require(torch.equal(live[key], saved[key]), 'live/saved tensor parity failed')
                require(infer(api.core, torch, models[arm], live) == infer(api.core, torch, models[arm], saved),
                        'live/saved scalar parity failed')
        report['startup_seconds'] = time.perf_counter() - started
        report['parity'] = dict(passed=True, n_images=200, arms=['off', 'on'], tensors_exact=True, scores_exact=True)
        for arm in ('off', 'on'):
            def prepare(identifier):
                guard.check()
                direct._bytes(registration)
                return prepare_live(direct, records[identifier], saved_document['entries'][identifier], raw_root, arm == 'on')
            observations = measure_cell(ids, prepare, lambda batch: infer(api.core, torch, models[arm], batch))
            model = models[arm]
            report['arms'][arm] = dict(observations=observations, summary=timing_summary(observations),
                checkpoint=selected[arm]['result']['checkpoint'], selected_epoch=selected[arm]['result']['selected_epoch'],
                training_identities=selected[arm]['identities'], initialization_provenance=selected[arm]['provenance'],
                n_total_parameters=sum(parameter.numel() for parameter in model.parameters()),
                n_trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                input_shapes=dict(query=[1, 4, 512, 800], overlap=[1, 1, 512, 800],
                                  reference=[1, 3, 512, 800] if arm == 'on' else None),
                checkpoint_bytes=Path(selected[arm]['result']['checkpoint']['path']).stat().st_size,
                state_sha256_before=before[arm], state_sha256_after=api.pilots.tensor_sha256(model.state_dict()))
            require(report['arms'][arm]['state_sha256_before'] == report['arms'][arm]['state_sha256_after'], 'model/buffer drift')
            require(host_info(torch, cv2) == host, 'host/thread drift')
        require([item['geometry'] for item in report['arms']['off']['observations']]
                == [item['geometry'] for item in report['arms']['on']['observations']], 'OFF/ON geometry mismatch')
        require(rng_sha(torch, np) == rng, 'Python/NumPy/Torch RNG drift')
        guard.finish()
        verify_scientific_files(ROOT, request)
        verify_imports(ROOT, request)
        api.validate_source(dict(evaluation_source_sha256=request['evaluation_source_sha256'],
                                 critical_source_hashes=request['scientific_source_hashes']))
        report.update(status='complete', comparable=True, rng_unchanged=True,
                      interpretation='single_host_validation_uncached_not_officer_device_or_test_outcome',
                      neural_boundary='nested_input_validation_forward_sigmoid_scalar_referral',
                      end_to_end_boundary='integrity_read_decode_letterbox_live_fit_inverse_warp_normalize_neural_decision',
                      storage_cache='OS_pages_warmed_no_cold_disk_claim')
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        with (output / 'runtime.json').open('x') as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(dict(path=args.request, sha256=args.request_sha256), args.out)


if __name__ == '__main__':
    main()
