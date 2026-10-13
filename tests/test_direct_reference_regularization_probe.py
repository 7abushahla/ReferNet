import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn
import yaml

import test_protocol_a_reference as reference_tests
from test_protocol_a_reference import api, inputs, contract_api, inventory

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def probe():
    path = ROOT / 'scripts/probe_direct_reference_regularization.py'
    assert path.is_file(), 'regularization probe is not implemented'
    specification = importlib.util.spec_from_file_location('regularization_probe', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class SyntheticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Parameter(torch.zeros(1))
        self.head = nn.Parameter(torch.zeros(1))
        self.reference_enabled = False
        self.architecture_id = 'original'
        self.initialization_provenance = dict(base=dict(checkpoint_sha256='b' * 64))

    def parameter_groups(self):
        return [self.encoder], [self.head]


@pytest.fixture
def setup(probe, tmp_path, monkeypatch):
    events, models = [], []
    rows = {role: pd.DataFrame([dict(image_id=f'{role}-{index:04}', role=role, design_id='sidtd/example',
        identity_id=f'{role}-{index:04}', component_id=f'{role}-{index:04}', label=index % 2,
        attack_family='text_copy' if index % 2 else '') for index in range(count)])
        for role, count in (('train', 1778), ('val', 222))}
    metadata = dict(protocol='A', fold=0, records_sha256='c' * 64, split_sha256='d' * 64,
                    enrollment_manifest_sha256='e' * 64)
    context = SimpleNamespace(identities=json.dumps(metadata), raw_root=tmp_path / 'raw',
        product_root=str(tmp_path / 'products'), manifest=dict(output_root=str(tmp_path / 'banks')),
        contract=SimpleNamespace(fold=0))
    def source_rows(role):
        assert role in ('train', 'val'), 'TEST rows accessed'
        events.append(('rows', role))
        return rows[role].copy()
    context.rows = source_rows
    base = yaml.safe_load((ROOT / 'configs/direct_reference_pilot.yaml').read_text())
    base_request = dict(schema_version='protocol_a_request_v1', protocol='A', fold=0, source_sha256='a' * 64,
        weights=dict(path=str(tmp_path / 'unopened.pt'), sha256='b' * 64), output_root=str(tmp_path / 'baseline'))
    base_binding = probe.driver.write(tmp_path / 'base.json', base_request)
    authority = dict(schema_version='direct_reference_regularization_probe_request_v1',
        base_request=base_binding, source_sha256='a' * 64, probability=0.0, auxiliary_query_weight=0., max_epochs=100,
        output_root=str(tmp_path / 'probe'))
    def load(binding, **kwargs):
        assert not kwargs and binding == base_binding
        events.append('authenticated-base')
        return base_request, context, base, tmp_path / 'baseline'
    monkeypatch.setattr(probe.driver, 'load_request', load)
    monkeypatch.setattr(probe.driver.service, 'effective_config', lambda base, context, enabled: copy.deepcopy(base))
    def fresh(request, enabled):
        assert request == base_request and enabled is False
        model = SyntheticModel()
        models.append(model)
        events.append('fresh-pretrained')
        return model
    monkeypatch.setattr(probe.driver, 'fresh_model', fresh)
    def regularize(model, probability):
        model.global_dropout = nn.Dropout(probability)
        model.local_dropout = nn.Dropout(probability)
        model.architecture_id = 'anchor_direct_reference_ia18_head_dropout_v1'
        model.initialization_provenance.update(architecture_id=model.architecture_id,
            regularization=dict(policy='head_readout_dropout_v1', p=probability))
        return model
    monkeypatch.setattr(probe, 'regularize_model', regularize)
    def build(context, *, roles, output_path):
        assert roles == ['train', 'val']
        events.append('registration-build')
        contents = dict(entries={'shared': {'seed': 42}}, roles=roles)
        probe.driver.write(output_path, contents)
        return SimpleNamespace(serialized=json.dumps(contents))
    def registration(context, binding, *, roles):
        assert roles == ['train', 'val']
        events.append('registration-load')
        return SimpleNamespace(serialized=json.dumps(probe.driver._json(binding)))
    monkeypatch.setattr(probe.driver.bridge, 'build_protocol_a_registration', build)
    monkeypatch.setattr(probe.driver.bridge, 'load_protocol_a_registration', registration)
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, context, table, *, role, reference_enabled, augment, cache_bytes):
            assert role in ('train', 'val') and reference_enabled is False
            assert augment['enabled'] is (role == 'train') and augment['seed'] == 42
            events.append(('dataset', role, cache_bytes))
            self.rows = context.rows(role).to_dict('records')
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            return self.rows[index]
    monkeypatch.setattr(probe.driver.bridge, 'ProtocolAReferenceDataset', Dataset)
    def schedule(optimizer, track, total):
        assert total == 22200 and track == 'D_running_pilot'
        events.append(('schedule', total))
        return SimpleNamespace(last_epoch=0)
    monkeypatch.setattr(probe.driver.core, 'pilot_scheduler', schedule)
    def train(model, loader, optimizer, scheduler, device, auxiliary_query_weight):
        model.train()
        assert device == 'cuda' and loader.batch_size == 8 and loader.num_workers == 0 and loader.drop_last
        assert not optimizer.state
        count = len(loader) * 8
        scheduler.last_epoch += len(loader)
        with torch.no_grad():
            model.encoder.add_(1.)
        events.append(('updates', len(loader)))
        return dict(loss=.2, image_loss=.2, aux_query_loss=0., n_images=count,
                    optimizer_steps=len(loader), bce_map=None, dice_map=None)
    monkeypatch.setattr(probe, 'train_epoch', train)
    def evaluate(model, loader, device):
        assert loader.batch_size == 1 and loader.num_workers == 0 and not loader.drop_last
        model.eval()
        data = loader.dataset
        selected = [data.dataset.rows[index] for index in data.indices] if isinstance(data, torch.utils.data.Subset) else data.rows
        predictions = [dict(image_id=row['image_id'], label=row['label'], design_id=row['design_id'],
            attack_family=row['attack_family'], attack_logit=2. if row['label'] else -2.,
            query_logit=2. if row['label'] else -2., correction_logit=0.,
            bonafide_score=.2 if row['label'] else .8, attack_probability=.8 if row['label'] else .2,
            context_available=True, referral_required=False) for row in selected]
        summary = probe.driver.core.summarize_predictions(predictions)
        epoch = int(model.encoder.item())
        summary.update(accuracy=.8 if epoch == 1 else .9, auc=.85 if epoch < 3 else .95,
                       image_loss=.2, buffers_unchanged=True)
        return summary, predictions
    monkeypatch.setattr(probe.driver.core, 'evaluate_direct_reference', evaluate)
    monkeypatch.setattr(probe.driver, 'invariant', lambda model, loader: dict(passed=True,
        n_images=4, max_abs_difference=0., buffers_unchanged=True))
    parent_pid = os.getpid()
    def child(binding, phase, gate_sha256=None):
        with monkeypatch.context() as child_patch:
            child_patch.setattr(probe.os, 'getpid', lambda: parent_pid + 1)
            return probe.replay(binding, phase=phase, gate_sha256=gate_sha256)
    monkeypatch.setattr(probe, '_child', child)
    def request():
        path = tmp_path / 'request.json'
        path.write_text(json.dumps(authority))
        return probe.driver.bind(path)
    return SimpleNamespace(authority=authority, request=request, events=events, models=models,
                           rows=rows, context=context, base_request=base_request, root=tmp_path)


@pytest.mark.parametrize('probability,auxiliary', [(0.0, 0.), (.1, 0.), (0., .5)])
def test_smoke_source_roles_schedule_cache_replay(probe, setup, probability, auxiliary):
    if probability or auxiliary:
        common = probe.driver.write(setup.root / 'common.json', dict(entries={'shared': {'seed': 42}}, roles=['train', 'val']))
        setup.authority['registration'] = common
    setup.authority['probability'] = probability
    setup.authority['auxiliary_query_weight'] = auxiliary
    gate = probe.smoke(setup.request())
    report = probe.driver._json(gate)
    assert report['probability'] == probability and report['no_test_open'] is True
    assert report['auxiliary_query_weight'] == auxiliary
    assert report['registration_entries_sha256'] == probe.driver.json_sha256({'shared': {'seed': 42}})
    assert setup.events.count('registration-build') == (0 if probability or auxiliary else 1)
    assert ('updates', 4) in setup.events and ('schedule', 22200) in setup.events
    assert ('dataset', 'train', 3 * 1024**3) in setup.events
    assert ('dataset', 'val', 1024**3) in setup.events
    assert len(setup.models) == 2
    assert not setup.models[-1].global_dropout.training


def test_fresh_full_train_earliest_selection_and_same_table(probe, setup):
    request = setup.request()
    gate = probe.smoke(request)
    result = probe.train(request, gate['sha256'])
    assert result['epochs_completed'] == 100 and result['selected_epoch'] == 3
    assert result['optimizer_steps'] == 22200 and result['diagnostic_only'] is True
    assert setup.events.count('registration-build') == 1
    assert ('updates', 222) in setup.events
    assert len(setup.models) == 4
    assert result['fresh_initial_state_sha256'] == probe.driver._json(gate)['fresh_initial_state_sha256']
    assert result['replay']['passed'] is True
    assert result['schema_version'] == 'direct_reference_regularization_probe_result_v1'
    checkpoint = torch.load(Path(result['checkpoint']['path']), map_location='cpu', weights_only=True)
    assert checkpoint['schema_version'] == 'direct_reference_regularization_probe_checkpoint_v1'
    assert checkpoint['epoch'] == 3 and checkpoint['phase'] == 'train'
    assert len(probe.driver.predictions(result['predictions'])) == 222


@pytest.mark.parametrize('key,value', [('probability', True), ('probability', .2), ('probability', .3),
                                     ('auxiliary_query_weight', True), ('auxiliary_query_weight', .1), ('max_epochs', 10),
                                     ('source_sha256', 'f' * 64), ('unexpected', 'resume')])
def test_invalid_authority_before_model_or_registration(probe, setup, key, value):
    setup.authority[key] = value
    with pytest.raises(ValueError):
        probe.smoke(setup.request())
    assert not setup.models and 'registration-build' not in setup.events


def test_nonzero_probability_requires_matched_registration(probe, setup):
    setup.authority['probability'] = .1
    with pytest.raises(ValueError, match='registration'):
        probe.smoke(setup.request())
    assert not setup.models


@pytest.mark.parametrize('failure', ['fold', 'count', 'role', 'order', 'output'])
def test_a0_membership_and_output_guards(probe, setup, failure):
    if failure == 'fold':
        setup.base_request['fold'] = 1
    elif failure == 'count':
        setup.rows['train'].drop(index=0, inplace=True)
    elif failure == 'role':
        setup.rows['val'].loc[0, 'role'] = 'test'
    elif failure == 'order':
        setup.rows['train'] = setup.rows['train'].iloc[::-1]
    else:
        setup.authority['output_root'] = str(setup.root / 'baseline' / 'new')
    with pytest.raises(ValueError):
        probe.smoke(setup.request())
    assert not setup.models and 'registration-build' not in setup.events


def test_gate_mismatch_cannot_start_full_training(probe, setup):
    request = setup.request()
    gate = probe.smoke(request)
    before = len(setup.models)
    with pytest.raises(ValueError, match='SHA'):
        probe.train(request, '0' * 64)
    assert len(setup.models) == before
    with pytest.raises(ValueError):
        probe.train(request, None)


def test_existing_smoke_output_is_not_rebuilt(probe, setup):
    request = setup.request()
    probe.smoke(request)
    with pytest.raises((ValueError, FileExistsError)):
        probe.smoke(request)
    assert setup.events.count('registration-build') == 1


def test_replay_requires_matching_phase_and_checkpoint_bytes(probe, setup):
    request = setup.request()
    gate = probe.smoke(request)
    with pytest.raises((ValueError, FileNotFoundError)):
        probe.replay(request, phase='train', gate_sha256=gate['sha256'])
    parent = probe.driver._json(probe.driver.bind(setup.root / 'probe' / 'smoke' / 'result.json'))
    with Path(parent['checkpoint']['path']).open('ab') as stream:
        stream.write(b'mutated')
    with pytest.raises(ValueError):
        probe.replay(request, phase='smoke')


def test_no_evaluate_or_test_cli_entry(probe):
    with pytest.raises(SystemExit):
        probe.main(['evaluate', '--request', '/unused.json', '--request-sha256', 'a' * 64])


def test_auxiliary_gradient_protects_query_and_eval_loss_stays_fused(probe):
    class Classifier(nn.Module):
        reference_enabled = False
        def __init__(self):
            super().__init__()
            self.query = nn.Parameter(torch.zeros(1))
            self.local = nn.Parameter(torch.zeros(1))
        def forward(self, query, **kwargs):
            global_logit = self.query.expand(len(query))
            local = self.local.expand(len(query))
            return dict(attack_logit=global_logit + local, query_logit=global_logit, correction_logit=local)
    batch = dict(query=torch.zeros(1, 4, 32, 32), overlap=torch.ones(1, 1, 32, 32),
        context_available=torch.ones(1, dtype=torch.bool), label=torch.ones(1),
        meta=dict(image_id=['train-synthetic'], design_id=['design'], attack_family=['text_copy']))
    gradients = []
    for weight in (0., .5):
        model = Classifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        before = torch.get_rng_state().clone()
        stats = probe.train_epoch(model, [batch], optimizer, None, 'cpu', weight)
        assert torch.equal(before, torch.get_rng_state())
        gradients.append((model.query.grad.item(), model.local.grad.item()))
        assert stats['aux_query_loss'] == pytest.approx(weight * .69314718)
        assert stats['image_loss'] == pytest.approx(.69314718)
        assert stats['loss'] == pytest.approx(stats['image_loss'] + stats['aux_query_loss'])
        summary, predictions = probe.driver.core.evaluate_direct_reference(model, [batch], 'cpu')
        assert summary['image_loss'] == pytest.approx(.69314718)
    assert gradients[0] == (-.5, -.5)
    assert gradients[1] == (-.75, -.5)


def test_auxiliary_cannot_be_combined_with_dropout(probe, setup):
    setup.authority.update(probability=.1, auxiliary_query_weight=.5)
    with pytest.raises(ValueError, match='combination'):
        probe.smoke(setup.request())


def test_three_probes_share_one_control_table_and_initial_tensors(probe, setup):
    registration = None
    reports = []
    for index, (probability, auxiliary) in enumerate(((0., 0.), (.1, 0.), (0., .5))):
        setup.authority.update(probability=probability, auxiliary_query_weight=auxiliary,
                               output_root=str(setup.root / f'probe-{index}'))
        if registration is not None:
            setup.authority['registration'] = registration
        request = probe.driver.write(setup.root / f'authority-{index}.json', copy.deepcopy(setup.authority))
        gate = probe.driver._json(probe.smoke(request))
        registration = gate['registration']
        reports.append(gate)
    assert setup.events.count('registration-build') == 1
    assert len({report['registration']['sha256'] for report in reports}) == 1
    assert len({report['registration_entries_sha256'] for report in reports}) == 1
    assert len({report['fresh_initial_state_sha256'] for report in reports}) == 1


@pytest.mark.parametrize('budget', [1024**3, 3 * 1024**3, 4 * 1024**3])
def test_existing_cached_uncached_off_parity_at_probe_budgets(api, inputs, monkeypatch, budget):
    original = reference_tests.dataset
    def dataset(*args, **kwargs):
        if kwargs.get('cache_bytes', 0):
            kwargs['cache_bytes'] = budget
        return original(*args, **kwargs)
    monkeypatch.setattr(reference_tests, 'dataset', dataset)
    reference_tests.test_cached_uncached_augmented_tensors_private_rng_and_mutation(api, inputs, False)
