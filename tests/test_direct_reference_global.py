import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

from test_direct_reference_model import assert_tensors_equal, buffers, build, checkpoint, inputs


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/train_direct_reference_global.py'


def launcher():
    assert SCRIPT.is_file(), 'Global-only launcher is not implemented'
    spec = importlib.util.spec_from_file_location('global_launcher_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conversion_preserves_exact_state_identity_and_rng(checkpoint):
    module = launcher()
    original = build(checkpoint, False)
    original_class = type(original)
    state = {name: value.clone() for name, value in original.state_dict().items()}
    identities = {name: id(value) for name, value in original.named_parameters()}
    policy = original.normalization_policy
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    model = module.global_model(original)
    assert model is original and type(model) is not original_class
    assert isinstance(model, original_class)
    assert_tensors_equal(model.state_dict(), state)
    assert {name: id(value) for name, value in model.named_parameters()} == identities
    assert model.reference_enabled is False
    assert model.architecture_id == 'anchor_direct_global_ablation_v1'
    assert model.normalization_policy == policy
    assert all(not parameter.requires_grad for parameter in model.adapter.parameters())
    assert all(parameter.requires_grad for parameter in model.base.parameters())
    assert random.getstate() == python_state
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == numpy_state[0]
    np.testing.assert_array_equal(actual_numpy[1], numpy_state[1])
    assert actual_numpy[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_state, atol=0, rtol=0)


@pytest.mark.parametrize('training', [False, True])
def test_direct_base_forward_once_without_adapter(checkpoint, training):
    model = launcher().global_model(build(checkpoint, False)).train(training)
    expected_base = copy.deepcopy(model.base)
    batch = inputs()
    batch['reference'] = object()
    with torch.no_grad():
        model.adapter.correction_head.weight.fill_(1)
        model.adapter.correction_head.bias.fill_(9)
    calls = []
    handles = [model.base.register_forward_pre_hook(lambda module, arguments: calls.append(arguments[0]))]
    handles.extend(child.register_forward_pre_hook(lambda module, arguments: pytest.fail('Frozen adapter executed'))
                   for child in model.adapter.modules())
    try:
        expected = expected_base(batch['query'])['attack_logit']
        actual = model(**batch)
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) == 1 and calls[0] is batch['query']
    assert set(actual) == {'attack_logit', 'query_logit', 'correction_logit'}
    torch.testing.assert_close(actual['attack_logit'], expected, atol=0, rtol=0)
    torch.testing.assert_close(actual['query_logit'], expected, atol=0, rtol=0)
    assert torch.count_nonzero(actual['correction_logit']) == 0
    assert_tensors_equal(buffers(model.base), buffers(expected_base))


@pytest.mark.parametrize('available', [True, False])
def test_base_updates_and_bn_advance_with_no_adapter_gradients(checkpoint, available):
    model = launcher().global_model(build(checkpoint, False)).train()
    batch = inputs()
    batch['context_available'].fill_(available)
    batch['reference'].requires_grad_()
    if not available:
        batch['overlap'].zero_()
    pretrained, new = model.parameter_groups()
    assert len(pretrained + new) == len(list(model.base.parameters()))
    assert {id(parameter) for parameter in pretrained} == {id(parameter) for parameter in model.base.backbone.parameters()}
    assert {id(parameter) for parameter in new} == {id(parameter) for parameter in model.base.classifier.parameters()}
    assert len(pretrained + new) == len({id(parameter) for parameter in pretrained + new})
    assert {id(parameter) for parameter in pretrained + new} == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    optimizer = torch.optim.SGD([{'params': pretrained, 'lr': .01}, {'params': new, 'lr': .1}])
    before = {name: value.clone() for name, value in model.named_parameters()}
    before_buffers = buffers(model)
    output = model(**batch)
    functional.binary_cross_entropy_with_logits(output['attack_logit'], torch.tensor([0., 1.])).backward()
    assert batch['reference'].grad is None
    assert all(parameter.grad is None for parameter in model.adapter.parameters())
    for group in (pretrained, new):
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in group)
        assert any(parameter.grad.abs().sum() > 0 for parameter in group)
    optimizer.step()
    for prefix in ('base.backbone.', 'base.classifier.'):
        assert any(not torch.equal(value, before[name]) for name, value in model.named_parameters() if name.startswith(prefix))
    for name, value in model.named_parameters():
        if name.startswith('adapter.'):
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    for name, value in model.named_buffers():
        if name.endswith('num_batches_tracked'):
            assert value == before_buffers[name] + 1


@pytest.mark.parametrize('failure', ['query_channels', 'query_rank', 'query_dtype', 'query_nan',
    'query_empty', 'query_grid', 'query_zone_low', 'query_zone_high', 'query_device',
    'overlap_shape', 'overlap_dtype', 'overlap_nonbinary', 'overlap_nan', 'overlap_device',
    'availability_shape', 'availability_dtype', 'availability_device'])
def test_invalid_inputs_rejected_before_base_or_bn(checkpoint, failure):
    model = launcher().global_model(build(checkpoint, False)).train()
    batch = inputs()
    if failure == 'query_channels': batch['query'] = batch['query'][:, :3]
    elif failure == 'query_rank': batch['query'] = batch['query'][0]
    elif failure == 'query_dtype': batch['query'] = batch['query'].double()
    elif failure == 'query_nan': batch['query'][0, 0, 0, 0] = float('nan')
    elif failure == 'query_empty': batch['query'] = batch['query'][:0]
    elif failure == 'query_grid': batch['query'] = batch['query'][:, :, :63]
    elif failure == 'query_zone_low': batch['query'][0, 3, 0, 0] = -.1
    elif failure == 'query_zone_high': batch['query'][0, 3, 0, 0] = 1.1
    elif failure == 'query_device': batch['query'] = batch['query'].to('meta')
    elif failure == 'overlap_shape': batch['overlap'] = batch['overlap'][:, :, :32]
    elif failure == 'overlap_dtype': batch['overlap'] = batch['overlap'].double()
    elif failure == 'overlap_nonbinary': batch['overlap'][0, 0, 0, 0] = .5
    elif failure == 'overlap_nan': batch['overlap'][0, 0, 0, 0] = float('nan')
    elif failure == 'overlap_device': batch['overlap'] = batch['overlap'].to('meta')
    elif failure == 'availability_shape': batch['context_available'] = batch['context_available'][:, None]
    elif failure == 'availability_dtype': batch['context_available'] = batch['context_available'].float()
    elif failure == 'availability_device': batch['context_available'] = batch['context_available'].to('meta')
    before = buffers(model)
    handle = model.base.register_forward_pre_hook(lambda module, arguments: pytest.fail('Invalid inputs reached base'))
    try:
        with pytest.raises(ValueError):
            model(**batch)
    finally:
        handle.remove()
    assert_tensors_equal(buffers(model), before)


def test_eval_companion_invariance_and_strict_state_replay(checkpoint):
    module = launcher()
    model = module.global_model(build(checkpoint, False)).eval()
    batch = inputs(batch=8)
    before = buffers(model)
    with torch.no_grad():
        solo = model(**{name: value[:1] for name, value in batch.items()})['attack_logit']
        for count in (1, 2, 4, 8):
            for position in sorted({0, count // 2, count - 1}):
                indices = list(range(1, count))
                indices.insert(position, 0)
                actual = model(**{name: value[indices] for name, value in batch.items()})['attack_logit']
                torch.testing.assert_close(actual[position:position + 1], solo, atol=1e-5, rtol=1e-4)
        expected = model(**batch)
        restored = module.global_model(build(checkpoint, False)).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        assert_tensors_equal(restored(**batch), expected)
    assert_tensors_equal(buffers(model), before)


def test_context_and_overlap_do_not_change_global_outputs_or_gradients(checkpoint):
    module = launcher()
    first = module.global_model(build(checkpoint, False)).train()
    second = copy.deepcopy(first)
    first_batch, second_batch = inputs(), inputs()
    first_batch['query'].requires_grad_()
    second_batch['query'].requires_grad_()
    first_batch['reference'] = object()
    second_batch['reference'] = torch.full((1,), float('nan'))
    second_batch['context_available'].zero_()
    second_batch['overlap'].zero_()
    first_output, second_output = first(**first_batch), second(**second_batch)
    assert_tensors_equal(first_output, second_output)
    first_output['attack_logit'].sum().backward()
    second_output['attack_logit'].sum().backward()
    torch.testing.assert_close(first_batch['query'].grad, second_batch['query'].grad, atol=0, rtol=0)
    assert_tensors_equal({name: parameter.grad for name, parameter in first.base.named_parameters()},
                         {name: parameter.grad for name, parameter in second.base.named_parameters()})
    assert_tensors_equal(buffers(first), buffers(second))


def binding(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def global_request(tmp_path):
    helper_path = ROOT / 'scripts/train_direct_reference_on.py'
    spec = importlib.util.spec_from_file_location('on_helper_for_global_tests', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    external = {}
    for name in ('control_request', 'registration', 'phase_authority'):
        path = tmp_path / (name + '.json')
        path.write_text('{}')
        external[name] = binding(path)
    document = dict(schema_version='direct_reference_global_request_v1', arm='global_only', epochs=100,
        fold=0, seed=42, batch_size=8, source_sha256=helper.SCIENTIFIC_SHA256,
        helper_sha256=binding(helper_path)['sha256'], launcher_sha256=binding(SCRIPT)['sha256'] if SCRIPT.is_file() else 'd' * 64,
        original_run_dir=str(tmp_path / 'original'), config_sha256='b' * 64,
        diagnostic_request_sha256='c' * 64, smoke_gate=None, replay_result=None, **external)
    return helper, tmp_path / 'authority.json', document


def write_request(path, document):
    path.write_text(json.dumps(document, allow_nan=False))
    return binding(path)


def test_request_accepts_exact_schema_and_nullable_replay_bindings(global_request):
    module = launcher()
    helper, path, document = global_request
    actual = module.request_document(helper, write_request(path, document))
    assert actual == document
    for field in ('smoke_gate', 'replay_result'):
        companion = path.with_name(field + '.json')
        companion.write_text('{}')
        document[field] = binding(companion)
        assert module.request_document(helper, write_request(path, document)) == document


@pytest.mark.parametrize('field,value', [
    ('schema_version', 'direct_reference_on_request_v1'), ('arm', 'off'), ('epochs', 1),
    ('epochs', 100.0), ('fold', False), ('seed', 41), ('batch_size', 4), ('batch_size', 8.0),
    ('source_sha256', '0' * 64), ('helper_sha256', '0' * 64), ('launcher_sha256', '0' * 64),
    ('config_sha256', 'invalid'), ('diagnostic_request_sha256', 'z' * 64),
    ('original_run_dir', 'relative/run'), ('smoke_gate', True), ('replay_result', {}),
    ('resume', 'best.pt')])
def test_request_rejects_wrong_values_and_extra_fields(global_request, field, value):
    module = launcher()
    helper, path, document = global_request
    document[field] = value
    with pytest.raises(ValueError):
        module.request_document(helper, write_request(path, document))


@pytest.mark.parametrize('field', ['schema_version', 'arm', 'epochs', 'fold', 'seed', 'batch_size',
    'source_sha256', 'helper_sha256', 'launcher_sha256', 'original_run_dir', 'config_sha256',
    'diagnostic_request_sha256', 'control_request', 'registration', 'phase_authority', 'smoke_gate', 'replay_result'])
def test_request_requires_every_field(global_request, field):
    module = launcher()
    helper, path, document = global_request
    del document[field]
    with pytest.raises(ValueError):
        module.request_document(helper, write_request(path, document))


def test_request_authenticates_bytes_and_rejects_duplicate_keys(global_request):
    module = launcher()
    helper, path, document = global_request
    authority = write_request(path, document)
    authority['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        module.request_document(helper, authority)
    content = json.dumps(document)
    path.write_text(content[:-1] + ', "arm": "global_only"}')
    with pytest.raises(ValueError):
        module.request_document(helper, binding(path))


@pytest.mark.parametrize('field', ['control_request', 'registration', 'phase_authority', 'smoke_gate', 'replay_result'])
@pytest.mark.parametrize('failure', ['missing_hash', 'relative_path', 'bad_hash', 'extra_field'])
def test_request_requires_strict_external_bindings(global_request, field, failure):
    module = launcher()
    helper, path, document = global_request
    external = dict(document['registration'])
    if failure == 'missing_hash': del external['sha256']
    elif failure == 'relative_path': external['path'] = 'relative.json'
    elif failure == 'bad_hash': external['sha256'] = 'not-a-sha'
    elif failure == 'extra_field': external['trusted'] = True
    document[field] = external
    with pytest.raises(ValueError):
        module.request_document(helper, write_request(path, document))


@pytest.mark.parametrize('arguments', [[], ['--epochs', '1'], ['--arm', 'off'], ['--resume', 'best.pt'], ['--phase', 'invalid']])
def test_cli_rejects_missing_authority_and_unapproved_overrides(arguments):
    required = ['--authority', '/external/authority.json', '--authority-sha256', 'a' * 64,
                '--source-dir', '/source', '--on-helper', '/external/on-helper.py',
                '--output-root', '/isolated/output', '--phase', 'smoke'] if arguments else []
    with pytest.raises(SystemExit):
        launcher().main(required + arguments)


class SyntheticGlobalDataset(Dataset):
    def __init__(self, rows, augmented):
        self.rows = rows.reset_index(drop=True)
        self.augmented = augmented
        self.geometry_rng = random.Random(42)
        self.photometry_rng = random.Random(43)
        self.batch = inputs(batch=len(rows), height=32, width=32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        if self.augmented:
            self.geometry_rng.random()
            self.photometry_rng.random()
        return dict(query=self.batch['query'][index], overlap=self.batch['overlap'][index],
                    context_available=torch.tensor(False), label=int(row.label),
                    meta={name: str(row[name]) for name in ('image_id', 'design_id', 'attack_family')})


@pytest.fixture
def orchestration(checkpoint, global_request, monkeypatch):
    from refer_or_clear import direct_reference_control as off
    from refer_or_clear import direct_reference_pilot as core
    from refer_or_clear import direct_reference_train as service
    from refer_or_clear import pilots
    from refer_or_clear.data import direct_reference as bridge
    from refer_or_clear.train import selection_key

    module = launcher()
    helper, authority_path, authority = global_request
    authority_binding = write_request(authority_path, authority)
    rows = {role: pd.DataFrame([dict(image_id=f'{role}{index:04d}', role=role, design_id=f'design{index % 2}',
                                   attack_family='text_copy' if index % 2 else '', label=index % 2)
                               for index in range(34)]) for role in ('train', 'val')}
    context = dict(device=torch.device('cpu'), run_dir=Path(authority['original_run_dir']))
    initial = build(checkpoint, False)
    initialization = dict(state_sha256=pilots.tensor_sha256(initial.state_dict()),
                          provenance=initial.initialization_provenance,
                          initialization_sha256=pilots.json_sha256(initial.initialization_provenance))
    initial_binding = write_request(authority_path.with_name('original-off-initialization.json'),
                                    {'initialization': initialization})
    receipt = {'synthetic': True, 'training_gates': {'off': initial_binding}}
    del initial
    events, models, datasets, commands = [], [], [], []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def fresh(context_arg, enabled, receipt):
        assert context_arg is context and enabled is False and receipt['synthetic'] is True
        model = build(checkpoint, False)
        models.append(model)
        events.append('fresh-model')
        return model

    def dataset(context_arg, prepared, selected, enabled, augmented=False):
        assert context_arg is context and prepared == {'synthetic': True} and enabled is False
        assert selected.role.nunique() == 1 and selected.role.iloc[0] in ('train', 'val')
        value = SyntheticGlobalDataset(selected, augmented)
        datasets.append(value)
        events.append(('dataset', selected.role.iloc[0], len(selected), augmented))
        return value

    @contextmanager
    def pixel_audit(context_arg, selected):
        assert context_arg is context and set(selected.role) <= {'train', 'val'}
        events.append(('audit', selected.role.iloc[0], len(selected)))
        yield []

    service_boundary = SimpleNamespace(_fresh_model=fresh, _dataset=dataset, pixel_audit=pixel_audit,
                                       bind=service.bind, write_receipt=service.write_receipt,
                                       _receipt_initialization=service._receipt_initialization)
    def authorize_control(context_arg, binding_arg):
        assert context_arg is context and binding_arg == authority['control_request']
        helper.document(binding_arg)
        events.append('reauthorize')
        return context, {}, rows, {}
    off_boundary = SimpleNamespace(_csv=off._csv, _number=off._number, authorize_control=authorize_control)
    bundle = SimpleNamespace(torch=torch, DataLoader=DataLoader, core=core, pilots=pilots,
                             service=service_boundary, off=off_boundary, bridge=bridge, selection_key=selection_key)
    ready = dict(context=context, receipt=receipt, prepared={'synthetic': True},
                 rows=rows, authority=authority, binding=authority_binding)

    def child(command, *, check):
        assert check is True
        commands.append(command)
        arguments = dict(zip(command[2::2], command[3::2]))
        child_binding = dict(path=arguments['--authority'], sha256=arguments['--authority-sha256'])
        child_ready = dict(ready, authority=module.request_document(helper, child_binding), binding=child_binding)
        parent_pid = os.getpid()
        with monkeypatch.context() as child_process:
            child_process.setattr(module.os, 'getpid', lambda: parent_pid + 1)
            module.replay(helper, bundle, child_ready, Path(arguments['--output-root']))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, 'run', child)
    try:
        yield SimpleNamespace(module=module, helper=helper, bundle=bundle, ready=ready, models=models,
                              datasets=datasets, events=events, commands=commands, root=authority_path.parent)
    finally:
        torch.set_num_threads(previous_threads)


def run_synthetic_smoke(fixture):
    return fixture.module.train(fixture.helper, fixture.bundle, fixture.ready,
                                fixture.root / 'smoke-output', 'smoke', fixture.root / 'source')


def test_smoke_uses_real_core_and_replays_saved_checkpoint(orchestration):
    fixture = orchestration
    result = run_synthetic_smoke(fixture)
    assert result['passed'] is True and result['fresh_process_replay'] is True
    assert result['epochs_completed'] == 1 and result['optimizer_steps'] == 4 and result['selected_epoch'] == 1
    assert result['summary']['n_images'] == 32 and result['summary']['buffers_unchanged'] is True
    assert result['summary']['n_context_unavailable'] == 32
    assert result['invariance']['batch_sizes'] == [1, 2, 4, 8]
    assert result['adapter_unchanged'] is True and result['no_test_open'] is True
    assert len(fixture.models) == 2 and fixture.models[0] is not fixture.models[1]
    assert fixture.commands[0][0] == sys.executable
    assert Path(fixture.commands[0][1]) == SCRIPT
    assert fixture.commands[0][fixture.commands[0].index('--phase') + 1] == 'replay'
    child = fixture.helper.document(binding(fixture.root / 'smoke-output/replay-request.json'))
    assert {key: value for key, value in child.items() if key != 'replay_result'} == {
        key: value for key, value in fixture.ready['authority'].items() if key != 'replay_result'}
    assert child['replay_result'] == result['phase_result']
    saved = torch.load(result['checkpoint']['path'], map_location='cpu', weights_only=True)
    assert saved['phase'] == 'smoke' and saved['epoch'] == 1
    assert saved['authority'] == fixture.ready['binding']
    assert len(saved['optimizer']['param_groups']) == 2
    assert len(saved['optimizer']['state']) == len(list(fixture.models[0].base.parameters()))
    assert saved['scheduler']['last_epoch'] == 4
    assert_tensors_equal(saved['state_dict'], fixture.models[1].state_dict())
    replay = fixture.helper.document(result['fresh_replay'])
    assert replay['comparison']['passed'] is True and replay['comparison']['max_abs_difference'] == 0.
    assert replay['summary'] == result['summary']
    assert fixture.events.count('reauthorize') == 1
    gated = dict(fixture.ready, authority=dict(fixture.ready['authority'],
                 smoke_gate=binding(fixture.root / 'smoke-output/result.json')))
    before_validation = list(fixture.events)
    fixture.module.validate_smoke(fixture.helper, fixture.bundle, gated)
    assert fixture.events == before_validation


def test_checkpoint_saves_loadable_rng_and_augmented_dataset_state(orchestration):
    fixture = orchestration
    result = run_synthetic_smoke(fixture)
    saved = torch.load(result['checkpoint']['path'], map_location='cpu', weights_only=True)
    dataset = next(value for value in fixture.datasets if value.augmented)
    for field, actual in (('geometry_rng', dataset.geometry_rng), ('photometry_rng', dataset.photometry_rng)):
        restored = random.Random()
        restored.setstate(saved[field])
        expected = random.Random()
        expected.setstate(actual.getstate())
        assert [restored.random() for _ in range(3)] == [expected.random() for _ in range(3)]
    generator = torch.Generator()
    generator.set_state(saved['sampler_rng'])
    expected = torch.Generator().manual_seed(1043)
    torch.randperm(32, generator=expected)
    torch.testing.assert_close(generator.get_state(), expected.get_state(), atol=0, rtol=0)
    generator.set_state(saved['torch_rng'])
    assert saved['torch_rng'].dtype == torch.uint8
    assert isinstance(saved['cuda_rng'], list)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(saved['torch_rng'])
        torch.testing.assert_close(torch.rand(5), torch.rand(5, generator=generator), atol=0, rtol=0)
    optimizer = fixture.bundle.pilots.make_pilot_optimizer(fixture.models[1], {'track': 'D_running_pilot'}, 'train')
    scheduler = fixture.bundle.pilots.pilot_scheduler(optimizer, 'D_running_pilot', 4)
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    assert optimizer.state_dict()['param_groups'] == saved['optimizer']['param_groups']
    assert scheduler.state_dict() == saved['scheduler']
    for parameter_id, state in optimizer.state_dict()['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, saved['optimizer']['state'][parameter_id][name], atol=0, rtol=0)


@pytest.mark.parametrize('failure', ['architecture', 'phase', 'authority', 'epoch', 'selection_key', 'state_keys', 'initial_hash'])
def test_replay_rejects_invalid_checkpoint_or_initial_identity(orchestration, failure):
    fixture = orchestration
    result = run_synthetic_smoke(fixture)
    phase = fixture.helper.document(result['phase_result'])
    saved = torch.load(phase['checkpoint']['path'], map_location='cpu', weights_only=True)
    if failure == 'architecture': saved['architecture_id'] = 'wrong'
    elif failure == 'phase': saved['phase'] = 'train'
    elif failure == 'authority': saved['authority'] = dict(saved['authority'], sha256='0' * 64)
    elif failure == 'epoch': saved['epoch'] = 2
    elif failure == 'selection_key': saved['selection_key'] = [0.]
    elif failure == 'state_keys': saved['state_dict'].pop(next(iter(saved['state_dict'])))
    elif failure == 'initial_hash': phase['fresh_initial_state_sha256'] = '0' * 64
    torch.save(saved, phase['checkpoint']['path'])
    phase['checkpoint'] = binding(Path(phase['checkpoint']['path']))
    phase_binding = write_request(fixture.root / 'altered-phase.json', phase)
    child_authority = dict(fixture.ready['authority'], replay_result=phase_binding)
    child_binding = write_request(fixture.root / 'altered-replay-request.json', child_authority)
    ready = dict(fixture.ready, authority=child_authority, binding=child_binding)
    fixture.events.clear()
    with pytest.raises((ValueError, RuntimeError)):
        fixture.module.replay(fixture.helper, fixture.bundle, ready, fixture.root / 'rejected-replay')
    assert not any(isinstance(event, tuple) and event[0] == 'dataset' for event in fixture.events)
    assert not (fixture.root / 'rejected-replay').exists()


@pytest.mark.parametrize('failure', ['update_count', 'training_population', 'validation_population', 'evaluation_buffers'])
def test_training_rejects_invalid_core_receipts_before_child_replay(orchestration, failure):
    fixture = orchestration
    original_core = fixture.bundle.core

    def train(*arguments):
        stats = original_core.train_direct_reference_epoch(*arguments)
        if failure == 'update_count': stats['optimizer_steps'] = 3
        elif failure == 'training_population': stats['n_images'] = 31
        return stats

    def evaluate(*arguments):
        summary, predictions = original_core.evaluate_direct_reference(*arguments)
        if failure == 'validation_population': summary['n_images'] = 31
        elif failure == 'evaluation_buffers': summary['buffers_unchanged'] = False
        return summary, predictions

    fixture.bundle.core = SimpleNamespace(train_direct_reference_epoch=train, evaluate_direct_reference=evaluate,
                                          _batch_inputs=original_core._batch_inputs, _forward=original_core._forward)
    with pytest.raises(ValueError):
        run_synthetic_smoke(fixture)
    assert fixture.commands == []
    assert not (fixture.root / 'smoke-output/phase-result.json').exists()


def test_replay_requires_authority_equality_except_result_binding(orchestration):
    fixture = orchestration
    result = run_synthetic_smoke(fixture)
    authority = dict(fixture.ready['authority'], replay_result=result['phase_result'],
                     smoke_gate=fixture.ready['authority']['registration'])
    authority_binding = write_request(fixture.root / 'changed-gate-replay.json', authority)
    ready = dict(fixture.ready, authority=authority, binding=authority_binding)
    fixture.events.clear()
    with pytest.raises(ValueError):
        fixture.module.replay(fixture.helper, fixture.bundle, ready, fixture.root / 'changed-gate-output')
    assert fixture.events == []


@pytest.mark.parametrize('failure', ['missing', 'wrong_result'])
def test_full_training_requires_completed_smoke_before_model_or_pixels(orchestration, failure):
    fixture = orchestration
    gate = None if failure == 'missing' else write_request(fixture.root / 'wrong-smoke.json', {'passed': True})
    ready = dict(fixture.ready, authority=dict(fixture.ready['authority'], smoke_gate=gate))
    with pytest.raises(ValueError):
        fixture.module.train(fixture.helper, fixture.bundle, ready,
                             fixture.root / 'full-training', 'train', fixture.root / 'source')
    assert fixture.events == [] and fixture.models == [] and fixture.datasets == []
    assert fixture.commands == []
    assert not (fixture.root / 'full-training').exists()


@pytest.mark.parametrize('failure', [None, 'loss_outside', 'loss_nan', 'auc', 'count', 'missing_field'])
def test_compare_summary_only_tolerates_finite_image_loss_roundoff(failure):
    from refer_or_clear import pilots
    module = launcher()
    bundle = SimpleNamespace(pilots=pilots)
    expected = dict(image_loss=.5, accuracy=.75, auc=.875, n_images=32, buffers_unchanged=True)
    actual = dict(expected, image_loss=.5 + (pilots.ATOL + pilots.RTOL * .5) / 2)
    if failure == 'loss_outside': actual['image_loss'] = .5 + 2 * (pilots.ATOL + pilots.RTOL * .5)
    elif failure == 'loss_nan': actual['image_loss'] = float('nan')
    elif failure == 'auc': actual['auc'] += 1e-8
    elif failure == 'count': actual['n_images'] = 31
    elif failure == 'missing_field': del actual['buffers_unchanged']
    if failure is None:
        module.compare_summary(bundle, expected, actual)
    else:
        with pytest.raises(ValueError):
            module.compare_summary(bundle, expected, actual)


def test_smoke_rejects_hash_consistent_nested_phase_and_outer_drift(orchestration):
    fixture = orchestration
    result = run_synthetic_smoke(fixture)
    original_phase = fixture.helper.document(result['phase_result'])
    original_replay = fixture.helper.document(result['fresh_replay'])
    for failure in ('schema', 'phase', 'epochs', 'updates', 'initialization', 'outer_summary', 'outer_checkpoint'):
        phase, replay, outer = copy.deepcopy(original_phase), copy.deepcopy(original_replay), copy.deepcopy(result)
        if failure == 'schema': phase['schema_version'] = 'not-global'
        elif failure == 'phase': phase['phase'] = 'train'
        elif failure == 'epochs': phase['epochs_completed'] = 2
        elif failure == 'updates': phase['optimizer_steps'] = 3
        elif failure == 'initialization': phase['fresh_initial_state_sha256'] = '0' * 64
        elif failure == 'outer_summary': outer['summary']['accuracy'] += .01
        elif failure == 'outer_checkpoint': outer['checkpoint'] = binding(fixture.root / 'smoke-output/last.pt')
        phase_binding = write_request(fixture.root / f'nested-{failure}-phase.json', phase)
        replay['phase_result'] = phase_binding
        outer['phase_result'] = phase_binding
        outer['fresh_replay'] = write_request(fixture.root / f'nested-{failure}-replay.json', replay)
        gate = write_request(fixture.root / f'nested-{failure}-outer.json', outer)
        ready = dict(fixture.ready, authority=dict(fixture.ready['authority'], smoke_gate=gate))
        fixture.events.clear()
        with pytest.raises(ValueError):
            fixture.module.validate_smoke(fixture.helper, fixture.bundle, ready)
        assert fixture.events == []


def test_replay_rejects_history_selection_and_prediction_drift_before_weights(orchestration):
    fixture = orchestration
    result = run_synthetic_smoke(fixture)
    original = fixture.helper.document(result['phase_result'])
    history = fixture.bundle.off._csv(fixture.bundle.bridge._bytes(original['history']))
    predictions = fixture.bundle.off._csv(fixture.bundle.bridge._bytes(original['predictions']))
    original_bytes = fixture.bundle.bridge._bytes

    def authenticated_bytes(authority):
        assert authority != original['checkpoint'], 'Invalid phase reached checkpoint bytes'
        return original_bytes(authority)

    fixture.bundle.bridge = SimpleNamespace(_bytes=authenticated_bytes)
    for failure in ('history_hash', 'history_short', 'epoch_accounting', 'selected_epoch', 'selection_key',
                    'initialization', 'design_id', 'label', 'prediction_order', 'score', 'correction'):
        phase = copy.deepcopy(original)
        entries, stored_predictions = copy.deepcopy(history), copy.deepcopy(predictions)
        if failure == 'history_hash': phase['history']['sha256'] = '0' * 64
        elif failure == 'history_short':
            path = fixture.root / 'empty-history.csv'
            path.write_text(','.join(history[0]) + '\n')
            phase['history'] = binding(path)
        elif failure == 'epoch_accounting':
            entries[0]['optimizer_steps'] = '3'
            path = fixture.root / 'bad-history.csv'
            fixture.bundle.pilots._write_csv(path, entries)
            phase['history'] = binding(path)
        elif failure == 'selected_epoch': phase['selected_epoch'] = 2
        elif failure == 'selection_key': phase['selection_key'] = [0., 0.]
        elif failure == 'initialization': phase['fresh_initial_state_sha256'] = '0' * 64
        else:
            if failure == 'design_id': stored_predictions[0]['design_id'] = 'foreign-design'
            elif failure == 'label': stored_predictions[0]['label'] = '1'
            elif failure == 'prediction_order': stored_predictions.reverse()
            elif failure == 'score': stored_predictions[0]['bonafide_score'] = '1.0'
            elif failure == 'correction': stored_predictions[0]['correction_logit'] = '0.25'
            path = fixture.root / f'bad-{failure}-predictions.csv'
            fixture.bundle.pilots._write_csv(path, stored_predictions)
            phase['predictions'] = binding(path)
        phase_binding = write_request(fixture.root / f'bad-{failure}-phase.json', phase)
        authority = dict(fixture.ready['authority'], replay_result=phase_binding)
        ready = dict(fixture.ready, authority=authority,
                     binding=write_request(fixture.root / f'bad-{failure}-request.json', authority))
        fixture.events.clear()
        with pytest.raises(ValueError):
            fixture.module.replay(fixture.helper, fixture.bundle, ready, fixture.root / f'bad-{failure}-output')
        assert fixture.events == []


def test_full_history_ties_select_earliest_epoch_before_weights(orchestration):
    fixture = orchestration
    result = run_synthetic_smoke(fixture)
    phase = fixture.helper.document(result['phase_result'])
    entry = fixture.bundle.off._csv(fixture.bundle.bridge._bytes(phase['history']))[0]
    entries = [dict(entry, epoch=epoch, optimizer_steps=222, n_images=1776) for epoch in range(1, 101)]
    path = fixture.root / 'tied-full-history.csv'
    fixture.bundle.pilots._write_csv(path, entries)
    authority = dict(fixture.ready['authority'], smoke_gate=binding(fixture.root / 'smoke-output/result.json'))
    authority_binding = write_request(fixture.root / 'full-training-request.json', authority)
    phase.update(phase='train', epochs_completed=100, optimizer_steps=22200,
                 selected_epoch=2, history=binding(path), authority=authority_binding)
    fixture.events.clear()
    with pytest.raises(ValueError, match='earliest selected history'):
        fixture.module.phase_contract(fixture.helper, fixture.bundle, fixture.ready, phase)
    assert fixture.events == []
