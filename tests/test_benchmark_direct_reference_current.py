import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / 'scripts/benchmark_direct_reference_current.py'
spec = importlib.util.spec_from_file_location('benchmark_current_test', PATH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def binding(path, value):
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_binding_mutation_and_duplicate_keys_rejected(tmp_path):
    authority = binding(tmp_path / 'request.json', {'passed': True})
    assert bench.document(authority) == {'passed': True}
    Path(authority['path']).write_text('{}')
    with pytest.raises(ValueError, match='SHA256'):
        bench.document(authority)
    path = tmp_path / 'duplicate.json'
    path.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError, match='duplicate'):
        bench.document({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})


def test_membership_is_first_200_sorted_validation_not_outcomes():
    import pandas as pd
    rows = pd.DataFrame({'image_id': [f'val/{index:03}' for index in reversed(range(222))],
                         'role': 'val', 'label': [index % 2 for index in range(222)]})
    selected = bench.measurement_rows(rows)
    assert selected.image_id.tolist() == [f'val/{index:03}' for index in range(200)]
    rows.loc[0, 'role'] = 'train'
    with pytest.raises(ValueError, match='validation'):
        bench.measurement_rows(rows)


def test_source_authentication_requires_all_34_original_files(tmp_path):
    source = tmp_path / 'source'
    files = {}
    for index in range(34):
        name = f'src/refer_or_clear/file{index}.py'
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'value = {index}\n')
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    request = {'scientific_source_hashes': files, 'overlay_source_hashes': {}}
    bench.verify_scientific_files(source, request)
    bytecode = source / 'src/refer_or_clear/unbound.pyc'
    bytecode.write_bytes(b'untrusted bytecode')
    with pytest.raises(ValueError, match='bytecode'):
        bench.verify_scientific_files(source, request)
    bytecode.unlink()
    next(source.rglob('*.py')).write_text('changed')
    with pytest.raises(ValueError, match='SHA256'):
        bench.verify_scientific_files(source, request)
    request['scientific_source_hashes'] = dict(list(files.items())[:33])
    with pytest.raises(ValueError, match='34'):
        bench.verify_scientific_files(source, request)


def test_parent_failure_before_any_pixel_preparation():
    events = []
    def validate(*args):
        events.append('authority')
        raise ValueError('full completion missing')
    api = SimpleNamespace(validate_arm=validate)
    with pytest.raises(ValueError, match='completion'):
        bench.validate_parents(api, {}, {}, None, None)
    assert events == ['authority']


@pytest.fixture
def geometry():
    sys.path.insert(0, str(PATH.parents[1] / 'src'))
    from refer_or_clear.data import direct_reference as actual
    products = {name: {'path': '/' + name, 'sha256': 'a' * 64}
                for name in ('target', 'median', 'zone', 'valid', 'target_valid', 'geometry', 'manifest')}
    image = np.full((512, 800, 3), 120, np.uint8)
    masks = {name: np.ones((512, 800), np.uint8) for name in ('zone', 'valid', 'target_valid')}
    calls = []
    def decode(item, flags):
        calls.append(item['path'])
        return image.copy()
    direct = SimpleNamespace(_image=decode, _common=lambda *args: (image.copy(), masks),
        to_canvas=lambda raw: (raw, np.ones((512, 800), np.uint8), None),
        _features=lambda plane: None, _fit=lambda *args: (np.eye(3), actual.MIN_INLIERS),
        _similarity=actual._similarity, _overlap=actual._overlap, _available=actual._available,
        _warp=actual._warp, _normalize=actual._normalize, MIN_INLIERS=actual.MIN_INLIERS,
        _digest=actual._digest)
    entry = {'image_id': 'val/000', 'query_sha256': 'b' * 64, 'products': products,
             'key': 'a' * 64, 'status': 'fit', 'reason': 'fit', 'inliers': actual.MIN_INLIERS,
             'query_canvas_to_bank': np.eye(3).tolist()}
    return direct, entry, calls


def test_live_geometry_same_tensors_off_on_and_no_off_median(geometry):
    direct, entry, calls = geometry
    row = {'image_id': 'val/000', 'path': 'raw.png', 'label': 0}
    off, off_geometry = bench.prepare_live(direct, row, entry, Path('/raw'), False)
    assert '/median' not in calls
    on, on_geometry = bench.prepare_live(direct, row, entry, Path('/raw'), True)
    assert '/median' in calls
    for key in ('query', 'overlap', 'context_available'):
        assert torch.equal(off[key], on[key])
    assert off_geometry == on_geometry
    assert 'reference' not in off and on['reference'].shape == (1, 3, 512, 800)
    assert off['query'].shape == (1, 4, 512, 800)


def test_geometry_drift_rejected_before_inference(geometry):
    direct, entry, calls = geometry
    direct._fit = lambda *args: (np.array([[1., 0., 2.], [0., 1., 0.], [0., 0., 1.]]), direct.MIN_INLIERS)
    with pytest.raises(ValueError, match='saved registration'):
        bench.prepare_live(direct, {'image_id': 'val/000', 'path': 'raw.png', 'label': 0}, entry, Path('/raw'), True)
    assert '/median' not in calls


def test_natural_failure_kept_referral_and_no_median(geometry):
    direct, entry, calls = geometry
    entry.update(status='failed', reason='registration_failed', inliers=0, query_canvas_to_bank=None)
    direct._fit = lambda *args: (None, 0)
    sample, details = bench.prepare_live(direct, {'image_id': 'val/000', 'path': 'raw.png', 'label': 1}, entry, Path('/raw'), True)
    assert not sample['context_available'].item()
    assert torch.count_nonzero(sample['reference']) == 0
    assert torch.count_nonzero(sample['query'][:, 3]) == 0
    assert details['referral_required'] and '/median' not in calls


def test_fixed_timing_counts_and_contiguous_boundary():
    events = []
    clock_values = iter(range(2000))
    def prepare(identifier):
        events.append(('prepare', identifier))
        return identifier, {'referral_required': False}
    def infer(identifier):
        events.append(('infer', identifier))
        return {'bonafide_score': .5, 'decision': 'genuine', 'referral_required': False}
    observed = bench.measure_cell([f'{index:03}' for index in range(200)], prepare, infer,
                                  clock=lambda: next(clock_values))
    assert len(events) == 440 and len(observed) == 200
    assert [item['image_id'] for item in observed] == [f'{index:03}' for index in range(200)]
    assert all(item['end_to_end_seconds'] == 2 and item['neural_seconds'] == 1 for item in observed)
    assert all(item['preparation_seconds'] == 1 for item in observed)
    with pytest.raises(ValueError, match='200'):
        bench.measure_cell(['one'], prepare, infer)


def test_infer_four_inputs_scalar_tie_and_no_grad():
    class Model(torch.nn.Module):
        def forward(self, *inputs):
            assert len(inputs) == 4 and not torch.is_grad_enabled()
            return {'attack_logit': torch.zeros(1)}
    model = Model().eval()
    api = SimpleNamespace(_strict_fp32=lambda *args: None,
                          _batch_inputs=lambda *args: ((1, 2, 3, 4), None),
                          _forward=lambda instance, inputs: instance(*inputs))
    value = bench.infer(api, torch, model, {'context_available': torch.tensor([True])})
    assert value == {'bonafide_score': .5, 'decision': 'genuine', 'referral_required': False}


def test_wrapper_cpu_environment_and_no_gpu():
    wrapper = PATH.parent / 'cluster/benchmark_direct_reference_current.sh'
    text = wrapper.read_text()
    for expected in ('--partition=cpu', '/opt/miniconda3/etc/profile.d/conda.sh',
                     '/shared/b00090279/conda/envs/refer-or-clear', 'sys.prefix',
                     'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'set -euo pipefail',
                     'CUDA_VISIBLE_DEVICES=""', '--request-sha256'):
        assert expected in text


def test_integrity_guard_rejects_changed_file_before_preparation(tmp_path):
    authority = binding(tmp_path / 'product.json', {'value': 1})
    guard = bench.IntegrityGuard([authority])
    guard.check()
    Path(authority['path']).write_text('{"value":2}')
    with pytest.raises(ValueError, match='changed'):
        guard.check()


def test_integrity_guard_rejects_replaced_symlink_ancestor(tmp_path):
    directory = tmp_path / 'bank'
    directory.mkdir()
    authority = binding(directory / 'product.json', {'value': 1})
    guard = bench.IntegrityGuard([authority])
    directory.rename(tmp_path / 'original')
    directory.symlink_to(tmp_path / 'original', target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        guard.check()


def test_strata_no_failures_is_unmeasured_not_zero():
    rows = [dict(preparation_seconds=1., neural_seconds=2., end_to_end_seconds=3.,
                 geometry={'context_available': True}) for _ in range(200)]
    summary = bench.timing_summary(rows)
    assert summary['end_to_end_seconds']['p95_seconds'] == 3
    assert summary['availability_strata']['unavailable'] == {'n': 0, 'status': 'unmeasured'}


def test_empty_overlap_failure_preserves_reason_and_no_median(geometry):
    direct, entry, calls = geometry
    direct._overlap = lambda *args: (np.eye(3), np.zeros((512, 800), np.uint8))
    entry.update(status='failed', reason='empty_overlap', query_canvas_to_bank=None)
    batch, details = bench.prepare_live(direct, {'image_id': 'val/000', 'path': 'raw.png', 'label': 0}, entry, Path('/raw'), True)
    assert details['reason'] == 'empty_overlap' and not batch['context_available'].item()
    assert '/median' not in calls


def test_original_seed_and_no_annotation_or_component_routing(geometry):
    direct, entry, _ = geometry
    seeds = []
    def fit(source, target, seed):
        seeds.append(seed)
        return np.eye(3), direct.MIN_INLIERS
    direct._fit = fit
    row = {'image_id': 'val/000', 'path': 'raw.png', 'label': 1,
           'annotation': 'do not read', 'component_id': 'do not route'}
    bench.prepare_live(direct, row, entry, Path('/raw'), False)
    assert seeds == [int(entry['key'][:8], 16) & 0x7fffffff]


def test_wrong_interpreter_prefix_rejected(monkeypatch):
    for name, value in dict(SLURMD_NODENAME='cpu001', SLURM_JOB_ID='123', SLURM_JOB_PARTITION='cpu',
                            CUDA_VISIBLE_DEVICES='', CONDA_PREFIX=str(bench.ENVIRONMENT),
                            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1').items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, 'prefix', '/wrong/interpreter')
    with pytest.raises(ValueError, match='interpreter'):
        bench.allocation()


def test_real_core_four_input_cpu_inference_and_retained_failure(geometry):
    from refer_or_clear import direct_reference_pilot as core
    direct, entry, _ = geometry
    seen = []
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.reference_enabled = True
        def forward(self, query, *, overlap, reference, context_available):
            assert not torch.is_grad_enabled()
            seen.append(bool(context_available.item()))
            value = torch.zeros(1)
            return dict(attack_logit=value, query_logit=value, correction_logit=value)
    model = Model().eval()
    row = {'image_id': 'val/000', 'path': 'raw.png', 'label': 0}
    batch, _ = bench.prepare_live(direct, row, entry, Path('/raw'), True)
    assert bench.infer(core, torch, model, batch)['decision'] == 'genuine'
    entry.update(status='failed', reason='registration_failed', inliers=0, query_canvas_to_bank=None)
    direct._fit = lambda *args: (None, 0)
    batch, _ = bench.prepare_live(direct, row, entry, Path('/raw'), True)
    assert bench.infer(core, torch, model, batch)['decision'] == 'refer'
    assert seen == [True, False]


@pytest.mark.parametrize('field,value', [('role', 'train'), ('threads', 4), ('warmups', 0),
                                       ('reference_cache', 'enabled'), ('timed_documents', 199)])
def test_strict_request_rejects_other_contracts_before_imports(tmp_path, field, value):
    request = dict(bench.FIXED, source_root=str(bench.ROOT), evaluation_source_sha256='a' * 64,
                   scientific_source_hashes={}, overlay_source_hashes={}, source_selection={},
                   image_ids=[], output_root=str(tmp_path / 'out'))
    request[field] = value
    authority = binding(tmp_path / 'request.json', request)
    with pytest.raises(ValueError, match='fixed'):
        bench.load_runtime_request(authority)


def test_conflicting_bindings_cannot_be_silently_deduplicated(tmp_path):
    authority = binding(tmp_path / 'product.json', {'value': 1})
    with pytest.raises(ValueError, match='conflicting'):
        bench.IntegrityGuard([dict(authority, sha256='0' * 64), authority])


def test_unavailable_scalar_is_referral_not_clearance():
    model = torch.nn.Linear(1, 1).eval()
    api = SimpleNamespace(_strict_fp32=lambda *args: None,
        _batch_inputs=lambda *args: ({}, None),
        _forward=lambda *args: {'attack_logit': torch.zeros(1)})
    result = bench.infer(api, torch, model, {'context_available': torch.tensor([False])})
    assert result['decision'] == 'refer' and result['referral_required']


@pytest.fixture
def run_harness(tmp_path, monkeypatch):
    import pandas as pd
    sys.path.insert(0, str(PATH.parents[1] / 'src'))
    from refer_or_clear.data import direct_reference as direct
    from refer_or_clear import pilots
    ids = [f'val/{index:03}' for index in range(222)]
    rows = pd.DataFrame({'image_id': ids, 'path': ['query.png'] * 222, 'role': 'val', 'label': 0})
    enrollment = binding(tmp_path / 'enrollment.json', {'output_root': str(tmp_path / 'original')})
    root = binding(tmp_path / 'root.json', {'relocated_root': str(tmp_path / 'relocated')})
    records = binding(tmp_path / 'records.json', {})
    split = binding(tmp_path / 'split.json', {})
    phase = binding(tmp_path / 'phase.json', {})
    registration = binding(tmp_path / 'table.json', {'preprocess': dict(enrollment_authority=enrollment,
        records_authority=records, split_authority=split, phase_authority=phase,
        product_root_authority=root, raw_root=str(tmp_path / 'raw')),
        'entries': {identifier: dict(role='val', bank='enrollment', products={}) for identifier in ids}})
    checkpoint = binding(tmp_path / 'synthetic-checkpoint.json', {})
    receipt = binding(tmp_path / 'receipt.json', {})
    result = dict(registration=registration, checkpoint=checkpoint, selected_epoch=1,
                  fresh_initial_state_sha256='same', diagnostic_gate='same')
    identities = {key: 'same' for key in ('source_sha256', 'weights_sha256', 'records_sha256', 'split_sha256',
                 'enrollment_sha256', 'registration_sha256', 'phase_authority_sha256', 'preprocess_sha256')}
    selected = dict(result=result, identities=identities, authority={'diagnostic_request_sha256': 'same'}, provenance={})
    selection = dict(records=records, split=split, enrollment=enrollment,
        arms={arm: {'completion': receipt, 'fresh_replay': receipt} for arm in ('off', 'on')})
    request = dict(source_selection=selection, image_ids=ids[:200], output_root=str(tmp_path / 'out'),
                   evaluation_source_sha256='same', scientific_source_hashes={}, overlay_source_hashes={})
    request_binding = binding(tmp_path / 'request.json', request)
    events = []
    state = {'drift': False, 'fail_parent': False}
    def validate(*args):
        events.append('parent')
        if state['fail_parent']:
            raise ValueError('incomplete parent')
        return selected
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('running', torch.zeros(1))
    def predict(model, inputs):
        if state['drift']:
            model.running.add_(1)
        return {'attack_logit': torch.zeros(1)}
    core = SimpleNamespace(_strict_fp32=lambda *args: None,
                          _batch_inputs=lambda model, batch, device: ({}, None), _forward=predict)
    def all_bindings(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'sha256'}:
                yield value
            else:
                for item in value.values():
                    yield from all_bindings(item)
    api = SimpleNamespace(validate_arm=validate, load_role_rows=lambda *args: rows,
        all_bindings=all_bindings, core=core, pilots=pilots, fresh_model=lambda *args, **kwargs: Model(),
        validate_source=lambda *args: None)
    batch = dict(query=torch.zeros(1, 4, 2, 3), overlap=torch.ones(1, 1, 2, 3),
                 context_available=torch.ones(1, dtype=torch.bool), label=torch.zeros(1, dtype=torch.int64))
    def prepare(*args):
        events.append('pixels')
        return dict(batch, **({'reference': torch.zeros(1, 3, 2, 3)} if args[-1] else {})), {'context_available': True}
    class Dataset:
        def __init__(self, *args, reference_enabled, train, augment):
            assert not train
            self.enabled = reference_enabled
        def __getitem__(self, index):
            return {key: value[0] for key, value in prepare(None, None, None, None, self.enabled)[0].items()}
    monkeypatch.setattr(bench, 'allocation', lambda: None)
    monkeypatch.setattr(bench, 'load_runtime_request', lambda *args: (request, api, {}))
    monkeypatch.setattr(bench, 'host_info', lambda *args: {'torch_intraop': 1, 'torch_interop': 1, 'opencv_threads': 1})
    monkeypatch.setattr(bench, 'verify_scientific_files', lambda *args: None)
    monkeypatch.setattr(bench, 'verify_imports', lambda *args: None)
    monkeypatch.setattr(bench, 'prepare_live', prepare)
    monkeypatch.setattr(direct, 'load_registration_table', lambda *args, **kwargs: {})
    monkeypatch.setattr(direct, 'DirectReferenceDataset', Dataset)
    return request_binding, request, events, state


def test_whole_dispatch_synthetic_only_and_same_200_both_arms(run_harness):
    authority, request, events, _ = run_harness
    report = bench.run(authority, request['output_root'])
    assert report['status'] == 'complete' and report['comparable']
    assert events[:2] == ['parent', 'parent']
    for arm in ('off', 'on'):
        assert len(report['arms'][arm]['observations']) == 200
        assert report['arms'][arm]['state_sha256_before'] == report['arms'][arm]['state_sha256_after']
    saved = json.loads((Path(request['output_root']) / 'runtime.json').read_text())
    assert saved['parity']['passed'] and saved['rng_unchanged']


def test_whole_dispatch_parent_failure_blocks_pixels_and_output(run_harness):
    authority, request, events, state = run_harness
    state['fail_parent'] = True
    with pytest.raises(ValueError, match='parent'):
        bench.run(authority, request['output_root'])
    assert 'pixels' not in events and not Path(request['output_root']).exists()


def test_whole_dispatch_state_drift_produces_incomplete_not_comparable(run_harness):
    authority, request, _, state = run_harness
    state['drift'] = True
    with pytest.raises(ValueError, match='buffer drift'):
        bench.run(authority, request['output_root'])
    saved = json.loads((Path(request['output_root']) / 'runtime.json').read_text())
    assert saved['status'] == 'incomplete' and not saved['comparable']
