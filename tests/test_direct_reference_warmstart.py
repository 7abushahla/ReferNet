import copy
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from test_direct_reference_control import authority, service, setup, write, bridge_binding
from refer_or_clear import direct_reference_control as off
from refer_or_clear import direct_reference_pilot as core
from refer_or_clear import pilots


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/train_direct_reference_warmstart.py'


@pytest.fixture
def runner():
    assert SCRIPT.is_file(), 'external exploratory warmstart runner missing'
    spec = importlib.util.spec_from_file_location('warmstart_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ready(runner, authority, service):
    helper = runner.load_helper(SCRIPT.with_name('train_direct_reference_on.py'))
    helper.SCIENTIFIC_SHA256 = authority['context']['identities']['source_sha256']
    bundle = SimpleNamespace(service=service, off=off, core=core, bridge=off.bridge,
                             pilots=pilots, torch=torch, DataLoader=DataLoader)
    context = authority['context']
    context['run_dir'] = authority['tmp'] / 'original-run'
    context['run_dir'].mkdir()
    prepared = bundle.service.prepare_training(context, authority['gate_binding'])
    model = bundle.service._fresh_model(context, False, authority['gate'])
    config = dict(context['config'], arm='off', reference_enabled=False)
    identities = bundle.service._checkpoint_identities(context, config, prepared['preprocess'], prepared['table_binding'],
                                                      pilots.json_sha256(model.initialization_provenance))
    checkpoint_path = authority['tmp'] / 'snapshot.pt'
    core.save_direct_reference_checkpoint(checkpoint_path, model, config, identities, 2, (.9, .95), 'train')
    history = [dict(epoch=index, n_images=1776, optimizer_steps=222, val_accuracy=.8 if index == 1 else .9,
                    val_auc=.9 if index == 1 else .95) for index in range(1, 5)]
    pilots._write_csv(authority['tmp'] / 'history.csv', history)
    request = dict(schema_version='direct_reference_warmstart_request_v1', exploratory_only=True,
                   arm='on', epochs=10, fold=0, seed=42, batch_size=8,
                   source_sha256=context['identities']['source_sha256'], config_sha256=authority['control']['config_sha256'],
                   diagnostic_request_sha256=context['request']['sha256'], helper_sha256=runner.HELPER_SHA,
                   original_run_dir=str(context['run_dir']),
                   launcher_sha256=helper.digest(SCRIPT), control_request=authority['control_binding'],
                   checkpoint=bridge_binding(checkpoint_path), history=bridge_binding(authority['tmp'] / 'history.csv'),
                   registration=prepared['table_binding'], phase_authority=prepared['preprocess']['phase_authority'])
    request_binding = write(authority['tmp'] / 'warmstart.json', request)
    authority['events'].clear()
    return dict(authority, helper=helper, bundle=bundle, request=request, request_binding=request_binding,
                output=authority['tmp'] / 'exploration', prepared=prepared)


def test_captured_partial_history_earliest_best_is_accepted(runner, ready):
    result = runner.authorize(ready['helper'], ready['bundle'], ready['context'], ready['request_binding'], ready['output'])
    assert result['checkpoint']['epoch'] == 2 and result['snapshot_epochs'] == 4
    assert result['checkpoint']['selection_key'] == [.9, .95]
    assert 'registration' not in ready['events']


@pytest.mark.parametrize('field,value', [('epochs', 100), ('epochs', 10.0), ('fold', True), ('seed', 41),
                                       ('exploratory_only', False), ('arm', 'off'), ('batch_size', 16),
                                       ('helper_sha256', '0' * 64), ('launcher_sha256', '0' * 64),
                                       ('config_sha256', '0' * 64), ('source_sha256', '0' * 64),
                                       ('diagnostic_request_sha256', '0' * 64), ('resume', 'old.pt')])
def test_authority_is_fixed_and_fail_closed(runner, ready, field, value):
    request = dict(ready['request'], **{field: value})
    binding = write(Path(ready['request_binding']['path']), request)
    with pytest.raises(ValueError):
        runner.authorize(ready['helper'], ready['bundle'], ready['context'], binding, ready['output'])
    assert not ready['events']


@pytest.mark.parametrize('failure', ['hash', 'new-best', 'earlier-tie', 'nan', 'gap', 'steps', 'checkpoint-identity', 'checkpoint-arm', 'phase-path', 'overlap'])
def test_snapshot_and_original_authorities_cannot_be_reinterpreted(runner, ready, failure):
    request = copy.deepcopy(ready['request'])
    if failure == 'hash': request['checkpoint']['sha256'] = '0' * 64
    if failure in ('new-best', 'earlier-tie', 'nan', 'gap', 'steps'):
        history = off._csv(ready['bundle'].bridge._bytes(request['history']))
        if failure == 'new-best': history[-1]['val_accuracy'] = '1.0'
        if failure == 'earlier-tie': history[0].update(val_accuracy='.9', val_auc='.95')
        if failure == 'nan': history[0]['val_auc'] = 'nan'
        if failure == 'gap': history[1]['epoch'] = '3'
        if failure == 'steps': history[0]['optimizer_steps'] = '221'
        pilots._write_csv(Path(request['history']['path']), history)
        request['history'] = bridge_binding(Path(request['history']['path']))
    if failure.startswith('checkpoint-'):
        checkpoint = torch.load(request['checkpoint']['path'], weights_only=True)
        if failure == 'checkpoint-identity': checkpoint['identities']['source_sha256'] = '0' * 64
        else: checkpoint['arm'] = 'on'
        torch.save(checkpoint, request['checkpoint']['path'])
        request['checkpoint'] = bridge_binding(Path(request['checkpoint']['path']))
    if failure == 'phase-path': request['phase_authority'] = write(ready['tmp'] / 'other-phase.json', {})
    output = ready['context']['run_dir'] / 'nested-output' if failure == 'overlap' else ready['output']
    binding = write(Path(ready['request_binding']['path']), request)
    with pytest.raises(ValueError):
        runner.authorize(ready['helper'], ready['bundle'], ready['context'], binding, output)


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)
        self.classifier = nn.Linear(4, 1)
    def features(self, query):
        return self.bn(self.encoder(query.mean((2, 3))))


class TinyDirect(nn.Module):
    def __init__(self, base, *, reference_enabled, adapter_seed=42):
        super().__init__()
        self.base, self.reference_enabled = base, reference_enabled
        self.adapter = nn.Linear(4, 1)
    def _validate_inputs(self, query, overlap, reference, context_available):
        available = context_available & overlap.flatten(1).any(1)
        return overlap.bool(), available
    def forward(self, query, *, overlap, reference, context_available):
        _, available = self._validate_inputs(query, overlap, reference, context_available)
        features = self.base.features(query)
        query_logit = self.base.classifier(features).squeeze(1)
        correction = torch.zeros_like(query_logit)
        if available.any():
            value = features[available]
            if self.reference_enabled:
                reference_query = torch.cat((reference[available], query[available, 3:4]), 1)
                value = value + self.base.features(reference_query)
            correction = correction.index_copy(0, available.nonzero(as_tuple=True)[0], self.adapter(value).squeeze(1))
        return dict(attack_logit=query_logit + correction, query_logit=query_logit, correction_logit=correction)


def batch(available=True):
    return dict(query=torch.randn(8, 4, 32, 32), reference=torch.randn(8, 3, 32, 32),
                overlap=torch.full((8, 1, 32, 32), float(available)), context_available=torch.full((8,), available),
                label=torch.arange(8) % 2)


def test_frozen_base_bn_and_adapter_only_updates_and_reference_gradients(runner):
    torch.manual_seed(42)
    original = TinyDirect(TinyBase(), reference_enabled=False)
    model = runner.frozen_model(original, original.state_dict())
    before = pilots.tensor_sha256(model.base.state_dict())
    adapter_before = pilots.tensor_sha256(model.adapter.state_dict())
    optimizer = runner.adapter_optimizer(model, torch)
    scheduler = pilots.pilot_scheduler(optimizer, 'D_running_pilot', 2220)
    sample = batch()
    sample['reference'].requires_grad_()
    stats = runner.train_epoch(model, [sample], optimizer, scheduler, torch.device('cpu'), SimpleNamespace(core=core, torch=torch))
    assert stats['optimizer_steps'] == 1 and scheduler.last_epoch == 1
    assert pilots.tensor_sha256(model.base.state_dict()) == before
    assert pilots.tensor_sha256(model.adapter.state_dict()) != adapter_before
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.base.parameters())
    assert not model.base.training and model.adapter.training and not model.base.bn.training
    assert sample['reference'].grad.abs().sum() > 0
    assert {id(parameter) for group in optimizer.param_groups for parameter in group['params']} == {id(parameter) for parameter in model.adapter.parameters()}
    assert optimizer.defaults['betas'] == (.9, .999) and optimizer.defaults['weight_decay'] == .05 and optimizer.defaults['eps'] == 1e-8


def test_all_unavailable_counts_loss_without_update_or_scheduler(runner):
    original = TinyDirect(TinyBase(), reference_enabled=False)
    model = runner.frozen_model(original, original.state_dict())
    optimizer = runner.adapter_optimizer(model, torch)
    scheduler = pilots.pilot_scheduler(optimizer, 'D_running_pilot', 2220)
    before = pilots.tensor_sha256(model.state_dict())
    stats = runner.train_epoch(model, [batch(False)], optimizer, scheduler, torch.device('cpu'), SimpleNamespace(core=core, torch=torch))
    assert stats['n_images'] == 8 and stats['skipped_batches'] == 1 and stats['optimizer_steps'] == 0
    assert scheduler.last_epoch == 0 and not optimizer.state
    assert pilots.tensor_sha256(model.state_dict()) == before
    assert stats['loss'] >= 0


def test_strict_full_off_state_is_required(runner):
    original = TinyDirect(TinyBase(), reference_enabled=False)
    state = dict(original.state_dict())
    state.pop('adapter.weight')
    with pytest.raises(RuntimeError): runner.frozen_model(original, state)


def test_exploratory_checkpoint_contains_complete_continuation_state(runner, tmp_path):
    original = TinyDirect(TinyBase(), reference_enabled=False)
    model = runner.frozen_model(original, original.state_dict())
    optimizer = runner.adapter_optimizer(model, torch)
    scheduler = pilots.pilot_scheduler(optimizer, 'D_running_pilot', 2220)
    dataset = SimpleNamespace(geometry_rng=random.Random(42), photometry_rng=random.Random(43))
    generators = dict(sampler=torch.Generator().manual_seed(1043), val=torch.Generator().manual_seed(3043))
    runner.save_state(tmp_path / 'best.pt', model, optimizer, scheduler, 0, (.9, .95), dataset, generators,
                      dict(source_sha256='a' * 64), torch)
    saved = torch.load(tmp_path / 'best.pt', weights_only=True)
    assert saved['schema_version'] == 'direct_reference_warmstart_checkpoint_v1'
    assert saved['epoch'] == 0 and saved['exploratory_only'] is True
    assert set(saved['rng']) == {'python', 'numpy', 'torch', 'cuda', 'dataset', 'generators'}
    assert set(saved) >= {'state_dict', 'optimizer', 'scheduler', 'rng', 'bindings', 'selection_key'}
    assert saved['rng']['dataset']['geometry_rng'] == dataset.geometry_rng.getstate()


def test_full_ten_epoch_exploration_includes_cold_on_zero_and_no_context_policy(runner, ready, monkeypatch):
    authorized = runner.authorize(ready['helper'], ready['bundle'], ready['context'], ready['request_binding'], ready['output'])
    baseline = TinyDirect(TinyBase(), reference_enabled=False)
    authorized['baseline'] = baseline
    authorized['checkpoint'] = dict(authorized['checkpoint'], state_dict=copy.deepcopy(baseline.state_dict()))
    active = {'role': None}
    counts = {'train': 0, 'val': 0}
    class Dataset:
        def __init__(self, rows, augmented):
            self.rows, self.augmented = rows.to_dict('records'), augmented
            self.geometry_rng, self.photometry_rng = random.Random(42), random.Random(43)
        def __len__(self): return len(self.rows)
        def __getitem__(self, index):
            row = self.rows[index]
            assert active['role'] == row['role'] and row['role'] in ('train', 'val')
            counts[row['role']] += 1
            if self.augmented:
                self.geometry_rng.random()
                self.photometry_rng.random()
            return dict(query=torch.full((4, 32, 32), 1. if row['label'] else -1.),
                        reference=torch.zeros(3, 32, 32), overlap=torch.zeros(1, 32, 32),
                        context_available=torch.tensor(False), label=row['label'],
                        meta={name: row[name] for name in ('image_id', 'design_id', 'attack_family')})
    def dataset(context, prepared, rows, enabled, augmented=False):
        assert augmented == rows.role.eq('train').all()
        return Dataset(rows, augmented)
    @contextmanager
    def audit(context, rows):
        role = rows.role.iloc[0]
        assert rows.role.eq(role).all() and role in ('train', 'val')
        active['role'] = role
        yield []
        active['role'] = None
    monkeypatch.setattr(ready['bundle'].service, '_dataset', dataset)
    monkeypatch.setattr(ready['bundle'].service, 'pixel_audit', audit)
    monkeypatch.setattr(core, 'save_direct_reference_checkpoint', lambda *args: pytest.fail('formal checkpoint schema used'))
    with audit(ready['context'], ready['rows']['val']):
        baseline_summary, _ = core.evaluate_direct_reference(baseline, DataLoader(Dataset(ready['rows']['val'], False), batch_size=1), torch.device('cpu'))
    authorized['checkpoint']['selection_key'] = [baseline_summary['accuracy'], baseline_summary['auc']]
    counts['val'] = 0
    result = runner.explore(ready['helper'], ready['bundle'], authorized, ready['output'], ready['request_binding'])
    assert result['epochs_completed'] == 10 and result['selected_epoch'] == 0
    assert result['optimizer_steps'] == 0 and result['skipped_batches'] == 2220
    assert counts['train'] == 17760 and counts['val'] == 222 * 15
    assert result['base_unchanged'] and result['stored_bn_unchanged'] and result['no_test_open']
    assert result['independently_trained_matched_comparison'] is False
    assert (ready['output'] / 'off_baseline.json').exists() and (ready['output'] / 'epoch0.json').exists()
    last = torch.load(ready['output'] / 'last.pt', weights_only=True)
    best = torch.load(ready['output'] / 'best.pt', weights_only=True)
    assert last['epoch'] == 10 and best['epoch'] == 0
    assert last['scheduler']['last_epoch'] == 0
    assert 'train' in last['rng']['generators'] and 'sampler' in last['rng']['generators']
