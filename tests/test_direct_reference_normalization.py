import importlib.util
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace
import yaml


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def diagnostic():
    path = ROOT / 'scripts/diagnose_direct_reference_normalization.py'
    assert path.is_file(), 'normalization diagnostic is not implemented'
    specification = importlib.util.spec_from_file_location('normalization_diagnostic', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class Samples(Dataset):
    def __init__(self, count=128, role='train'):
        self.count = count
        self.role = role
        self.geometry_rng = random.Random(23)
        self.photometry_rng = random.Random(24)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        self.geometry_rng.random()
        self.photometry_rng.random()
        label = index % 2
        query = torch.zeros(4, 32, 32)
        query[0].fill_(5 + 2 * label)
        query[3].fill_(1)
        return dict(query=query, overlap=torch.ones(1, 32, 32),
            context_available=torch.tensor(index % 17 != 0), label=label,
            meta=dict(image_id=f'image-{index:03}', role=self.role, design_id='sidtd/toy',
                attack_family='text_copy' if index % 4 == 1 else 'text_inpaint' if label else ''))


class Classifier(nn.Module):
    reference_enabled = False

    def __init__(self, failure=None):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout(.9)
        self.failure = failure
        self.calls = []

    def forward(self, query, *, overlap, reference, context_available):
        assert not self.training and not self.dropout.training
        self.calls.append(self.bn.training)
        random.random()
        np.random.random()
        torch.rand(1)
        features = self.bn(query[:, :1]).mean((1, 2, 3))
        if self.failure == 'exception' and self.bn.training:
            raise RuntimeError('synthetic forward failure')
        if self.failure == 'nan' and self.bn.training:
            features = features * float('nan')
        correction = torch.zeros_like(features)
        return dict(attack_logit=features, query_logit=features, correction_logit=correction)


def loader(samples=None, batch_size=8):
    return DataLoader(samples or Samples(), batch_size=batch_size, shuffle=False,
        generator=torch.Generator().manual_seed(73), num_workers=0)


def snapshot(model, batches):
    return dict(buffers={name: value.clone() for name, value in model.named_buffers()},
        parameters={name: value.clone() for name, value in model.named_parameters()},
        flags=[module.training for module in model.modules()],
        python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        generator=batches.generator.get_state(), geometry=batches.dataset.geometry_rng.getstate(),
        photometry=batches.dataset.photometry_rng.getstate())


def assert_restored(before, model, batches):
    for name, value in model.named_buffers():
        assert torch.equal(value, before['buffers'][name])
    for name, value in model.named_parameters():
        assert torch.equal(value, before['parameters'][name])
    assert before['flags'] == [module.training for module in model.modules()]
    assert random.getstate() == before['python']
    current = np.random.get_state()
    assert current[0] == before['numpy'][0] and np.array_equal(current[1], before['numpy'][1])
    assert current[2:] == before['numpy'][2:]
    assert torch.equal(torch.get_rng_state(), before['torch'])
    assert torch.equal(batches.generator.get_state(), before['generator'])
    assert batches.dataset.geometry_rng.getstate() == before['geometry']
    assert batches.dataset.photometry_rng.getstate() == before['photometry']


def test_paired_bn_metrics_components_families_and_exact_restoration(diagnostic):
    model = Classifier()
    model.bn.eval()
    batches = loader()
    before = snapshot(model, batches)
    report = diagnostic.diagnose_normalization(model, batches, 'cpu')
    assert report['state_restored'] is True
    assert report['n_images'] == 128 and len(report['image_ids']) == 128
    stored, batch = report['conditions']['stored_bn'], report['conditions']['batch_bn']
    assert stored['components']['sum']['accuracy'] == .5
    assert batch['components']['sum']['accuracy'] == 1.
    assert batch['components']['sum']['image_loss'] < stored['components']['sum']['image_loss']
    assert batch['components']['correction']['accuracy'] == .5
    assert set(batch['components']['sum']['families']) == {'genuine', 'text_copy', 'text_inpaint'}
    assert stored['buffers_changed_temporarily'] is False
    assert batch['buffers_changed_temporarily'] is True
    assert model.calls == [False] * 16 + [True] * 16
    assert_restored(before, model, batches)


@pytest.mark.parametrize('failure', ['exception', 'nan'])
def test_failure_restores_buffers_flags_and_all_rng(diagnostic, failure):
    model = Classifier(failure)
    batches = loader()
    before = snapshot(model, batches)
    with pytest.raises((RuntimeError, ValueError)):
        diagnostic.diagnose_normalization(model, batches, 'cpu')
    assert_restored(before, model, batches)


@pytest.mark.parametrize('count,role,batch_size', [(120, 'train', 8), (128, 'val', 8), (128, 'train', 4)])
def test_fixed_train_population_guard_before_forward(diagnostic, count, role, batch_size):
    model = Classifier()
    batches = loader(Samples(count, role), batch_size)
    before = snapshot(model, batches)
    with pytest.raises(ValueError):
        diagnostic.diagnose_normalization(model, batches, 'cpu')
    assert not model.calls
    assert_restored(before, model, batches)


def test_balanced_selection_is_lexical_and_outcome_independent(diagnostic):
    rows = pd.DataFrame([dict(image_id=f'{label}-{index:03}', label=label, role='train')
        for label in (0, 1) for index in range(80)]).iloc[::-1].reset_index(drop=True)
    selected = rows.iloc[diagnostic.select_train_indices(rows)]
    assert selected.label.tolist() == [0, 1] * 64
    assert selected.image_id.tolist()[::2] == [f'0-{index:03}' for index in range(64)]
    assert selected.image_id.tolist()[1::2] == [f'1-{index:03}' for index in range(64)]
    rows.loc[0, 'role'] = 'test'
    with pytest.raises(ValueError):
        diagnostic.select_train_indices(rows)


def test_cli_gpu_guard_precedes_inputs(diagnostic, monkeypatch):
    monkeypatch.setattr(diagnostic.driver, 'require_gpu_allocation',
        lambda: (_ for _ in ()).throw(ValueError('allocated GPU required')))
    monkeypatch.setattr(diagnostic.driver, 'load_request', lambda *args: pytest.fail('input before GPU guard'))
    assert diagnostic.main(['--request', '/absent/request.json', '--request-sha256', 'a' * 64,
        '--train-result', '/absent/train.json', '--train-result-sha256', 'b' * 64,
        '--registration', '/absent/registration.json', '--registration-sha256', 'c' * 64,
        '--output', '/absent/output.json']) == 1


@pytest.fixture
def saved(diagnostic, tmp_path):
    config = yaml.safe_load((ROOT / 'configs/direct_reference_pilot.yaml').read_text())
    provenance = dict(architecture_id=diagnostic.driver.core.ARCHITECTURE_ID, reference_enabled=False,
        base=dict(requested_initialization='levir', checkpoint_sha256=config['weights_sha256']))
    identities = {name: 'a' * 64 for name in diagnostic.driver.core.IDENTITY_FIELDS}
    identities.update(config_sha256=diagnostic.driver.json_sha256(config), weights_sha256=config['weights_sha256'],
        initialization_sha256=diagnostic.driver.json_sha256(provenance))
    checkpoint = dict(schema_version='direct_reference_pilot_v1', architecture_id=config['architecture_id'],
        arm='off', config=config, config_sha256=identities['config_sha256'], identities=identities,
        epoch=66, phase='train', selected=True, state_dict={'weight': torch.ones(1)},
        initialization_provenance=provenance)
    path = tmp_path / 'best.pt'
    torch.save(checkpoint, path)
    train = dict(passed=True, phase='train', arm='off', diagnostic_only=False, epochs_completed=100,
        selected_epoch=66, identities=identities, checkpoint=diagnostic.driver.bind(path))
    binding = diagnostic.driver.write(tmp_path / 'train.json', train)
    context = SimpleNamespace(contract=SimpleNamespace(fold=0), identities=json.dumps(dict(protocol='A', fold=0,
        records_sha256='a' * 64, split_sha256='a' * 64, enrollment_manifest_sha256='a' * 64)))
    return SimpleNamespace(train=train, binding=binding, checkpoint=checkpoint, context=context)


def test_legacy_reader_keeps_saved_config_and_identities(diagnostic, saved):
    train, checkpoint = diagnostic.load_selected_checkpoint(saved.binding, saved.context)
    assert train == saved.train
    assert checkpoint['config'] == saved.checkpoint['config']
    assert checkpoint['identities'] == saved.train['identities']
    assert checkpoint['epoch'] == 66


@pytest.mark.parametrize('failure', ['hash', 'epoch', 'source', 'arm', 'incomplete'])
def test_legacy_receipt_and_checkpoint_binding_fail_closed(diagnostic, saved, failure):
    train = saved.train
    if failure == 'hash':
        train['checkpoint']['sha256'] = '0' * 64
    elif failure == 'epoch':
        train['selected_epoch'] = 65
    elif failure == 'source':
        train['identities']['source_sha256'] = 'b' * 64
    elif failure == 'arm':
        train['arm'] = 'on'
    else:
        train['epochs_completed'] = 99
    binding = diagnostic.driver.write(Path(saved.binding['path']).with_name('changed.json'), train)
    with pytest.raises(ValueError):
        diagnostic.load_selected_checkpoint(binding, saved.context)


def test_checkpoint_mutation_during_reader_rejected(diagnostic, saved, monkeypatch):
    original = diagnostic.driver.core.read_direct_reference_checkpoint
    def mutate(path, config, identities):
        checkpoint = original(path, config, identities)
        Path(path).write_bytes(b'changed checkpoint')
        return checkpoint
    monkeypatch.setattr(diagnostic.driver.core, 'read_direct_reference_checkpoint', mutate)
    with pytest.raises(ValueError, match='SHA256'):
        diagnostic.load_selected_checkpoint(saved.binding, saved.context)


def test_cli_source_train_only_with_original_checkpoint(diagnostic, saved, monkeypatch, tmp_path_factory):
    inputs = Path(saved.binding['path']).parent
    output = tmp_path_factory.mktemp('normalization-output') / 'result.json'
    model = Classifier()
    model.architecture_id = diagnostic.driver.core.ARCHITECTURE_ID
    saved.checkpoint['state_dict'] = model.state_dict()
    path = Path(saved.train['checkpoint']['path'])
    torch.save(saved.checkpoint, path)
    saved.train['checkpoint'] = diagnostic.driver.bind(path)
    binding = diagnostic.driver.write(inputs / 'cli-train.json', saved.train)
    context = saved.context
    context.raw_root = inputs / 'raw'
    context.product_root = str(inputs / 'products')
    context.manifest = dict(output_root=str(inputs / 'banks'))
    rows = pd.DataFrame([dict(image_id=f'image-{index:03}', role='train', label=index % 2)
                         for index in range(128)])
    def source_rows(role):
        assert role == 'train', 'VAL/TEST accessed'
        return rows.copy()
    context.rows = source_rows
    config = saved.checkpoint['config']
    request = dict(source_sha256='b' * 64)
    monkeypatch.setattr(diagnostic.driver, 'require_gpu_allocation', lambda: None)
    monkeypatch.setattr(diagnostic.driver, 'load_request', lambda value: (request, context, config, inputs / 'baseline'))
    monkeypatch.setattr(diagnostic.driver, 'fresh_model', lambda request, enabled: model)
    table = diagnostic.driver.write(inputs / 'registration.json', {})
    def load(context, binding, *, roles):
        assert binding == table and roles == ['train', 'val']
        return {}
    monkeypatch.setattr(diagnostic.driver.bridge, 'load_protocol_a_registration', load)
    def dataset(context, table, *, role, reference_enabled, augment, cache_bytes):
        assert role == 'train' and not reference_enabled and augment['enabled'] is False
        return Samples()
    monkeypatch.setattr(diagnostic.driver.bridge, 'ProtocolAReferenceDataset', dataset)
    original = diagnostic.diagnose_normalization
    def evaluate(model, loader, device):
        assert device == 'cuda'
        return original(model, loader, 'cpu')
    monkeypatch.setattr(diagnostic, 'diagnose_normalization', evaluate)
    assert diagnostic.main(['--request', str(inputs / 'request.json'), '--request-sha256', 'b' * 64,
        '--train-result', binding['path'], '--train-result-sha256', binding['sha256'],
        '--registration', table['path'], '--registration-sha256', table['sha256'], '--output', str(output)]) == 0
    report = json.loads(output.read_text())
    assert report['role'] == 'train' and report['n_images'] == 128 and report['checkpoint_epoch'] == 66
    assert report['checkpoint'] == saved.train['checkpoint']
    assert report['trained_config'] == saved.checkpoint['config']
    assert report['trained_identities']['source_sha256'] == 'a' * 64
    assert report['diagnostic_source_sha256'] == 'b' * 64
