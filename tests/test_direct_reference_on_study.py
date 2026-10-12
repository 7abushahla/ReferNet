import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_direct_reference_on import (
    artifacts, authority, bridge_binding, completed, on_products, ready, revise_scores, service, setup, write,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/train_direct_reference_on_study.py'
POLICY = 'report_observed_no_performance_gate'


@pytest.fixture
def study_module():
    assert SCRIPT.is_file(), 'separate ON study runner missing'
    specification = importlib.util.spec_from_file_location('_on_study_test', SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def study_ready(study_module, ready, monkeypatch):
    monkeypatch.setattr(study_module, 'SCIENTIFIC_SHA256', ready['context']['identities']['source_sha256'])
    document = dict(ready['on'], schema_version='direct_reference_on_study_request_v1', anchor_policy=POLICY,
                    launcher_sha256=bridge_binding(SCRIPT)['sha256'])
    binding = write(ready['tmp'] / 'on-study-request.json', document)
    return dict(ready, study_module=study_module, study=document, study_binding=binding)


def test_completed_anchor_is_reported_without_changing_shared_registration(study_ready):
    fixture = study_ready
    result = fixture['study_module'].authorize(fixture['bundle'], fixture['context'], fixture['study_binding'])
    assert result['anchor_baseline'] == dict(policy=POLICY, accuracy=1., auc=1.,
        original_targets=dict(accuracy=.97, auc=.99), original_targets_met=True)
    assert result['prepared']['table_binding'] == fixture['result']['registration']
    assert result['prepared']['preprocess']['phase_authority'] == fixture['study']['phase_authority']
    assert fixture['events'] == []


@pytest.mark.parametrize('metric', ['accuracy', 'auc'])
def test_below_target_anchor_is_authorized_and_reported_without_weakening_old_on(study_ready, metric):
    fixture = study_ready
    document = dict(fixture['study'], off_completion=revise_scores(fixture, metric))
    binding = write(Path(fixture['study_binding']['path']), document)
    completion = json.loads(Path(document['off_completion']['path']).read_text())
    summary = json.loads(Path(completion['fresh_replay']['path']).read_text())['summary']
    assert summary[metric] < {'accuracy': .97, 'auc': .99}[metric]
    result = fixture['study_module'].authorize(fixture['bundle'], fixture['context'], binding)
    assert result['anchor_baseline'] == dict(policy=POLICY, accuracy=summary['accuracy'], auc=summary['auc'],
        original_targets=dict(accuracy=.97, auc=.99), original_targets_met=False)
    old_binding = write(Path(fixture['on_binding']['path']), dict(fixture['on'], off_completion=document['off_completion']))
    with pytest.raises(ValueError, match='competence'):
        fixture['module'].authorize(fixture['bundle'], fixture['context'], old_binding)
    assert fixture['events'] == []


@pytest.mark.parametrize('field,value', [
    ('schema_version', 'direct_reference_on_request_v1'), ('schema_version', 'other'),
    ('anchor_policy', 'require_original_targets'), ('anchor_policy', True), ('anchor_policy', None),
    ('arm', 'off'), ('epochs', 99), ('epochs', 100.0), ('fold', False), ('seed', 41),
    ('source_sha256', '0' * 64), ('config_sha256', '0' * 64),
    ('diagnostic_request_sha256', '0' * 64), ('launcher_sha256', '0' * 64), ('resume', 'selected.pt'),
])
def test_study_requires_separate_fixed_authority_and_reporting_policy(study_ready, field, value):
    fixture = study_ready
    binding = write(Path(fixture['study_binding']['path']), dict(fixture['study'], **{field: value}))
    with pytest.raises(ValueError):
        fixture['study_module'].authorize(fixture['bundle'], fixture['context'], binding)
    assert fixture['events'] == []


def test_missing_anchor_policy_does_not_implicitly_revoke_performance_gate(study_ready):
    fixture = study_ready
    document = dict(fixture['study'])
    document.pop('anchor_policy')
    binding = write(Path(fixture['study_binding']['path']), document)
    with pytest.raises(ValueError):
        fixture['study_module'].authorize(fixture['bundle'], fixture['context'], binding)
    assert fixture['events'] == []


@pytest.mark.parametrize('failure', [
    'authority-hash', 'off-hash', 'off-incomplete', 'flag-only', 'replay-hash', 'replay-failed',
    'replay-ids', 'replay-identity', 'replay-summary', 'replay-pid', 'table-hash', 'phase-hash',
    'phase-path', 'existing-on',
])
def test_reporting_policy_preserves_hash_completion_replay_and_shared_product_guards(study_ready, failure):
    fixture = study_ready
    document = copy.deepcopy(fixture['study'])
    if failure == 'off-hash':
        document['off_completion']['sha256'] = '0' * 64
    elif failure in ('off-incomplete', 'flag-only'):
        completion = dict(fixture['completion'], epochs_completed=99) if failure == 'off-incomplete' else dict(passed=True)
        document['off_completion'] = write(Path(document['off_completion']['path']), completion)
    elif failure.startswith('replay-'):
        completion = copy.deepcopy(fixture['completion'])
        if failure == 'replay-hash':
            completion['fresh_replay']['sha256'] = '0' * 64
        else:
            report = copy.deepcopy(fixture['fresh'])
            if failure == 'replay-failed': report['passed'] = False
            if failure == 'replay-ids': report['image_ids'].reverse()
            if failure == 'replay-identity': report['identities'] = {}
            if failure == 'replay-summary': report['summary']['accuracy'] = .1
            if failure == 'replay-pid': report['pid'] = False
            completion['fresh_replay'] = write(fixture['directory'] / 'fresh-replay.json', report)
        document['off_completion'] = write(Path(document['off_completion']['path']), completion)
    elif failure == 'table-hash':
        document['registration']['sha256'] = '0' * 64
    elif failure == 'phase-hash':
        document['phase_authority']['sha256'] = '0' * 64
    elif failure == 'phase-path':
        original = json.loads(Path(document['phase_authority']['path']).read_text())
        document['phase_authority'] = write(fixture['tmp'] / 'other-phase.json', original)
    elif failure == 'existing-on':
        (fixture['context']['run_dir'] / 'training/on').mkdir()
    binding = write(Path(fixture['study_binding']['path']), document)
    if failure == 'authority-hash': binding['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        fixture['study_module'].authorize(fixture['bundle'], fixture['context'], binding)
    assert fixture['events'] == []


@pytest.mark.parametrize('metric', ['accuracy', 'auc'])
def test_below_target_anchor_still_requires_valid_fresh_replay(study_ready, metric):
    fixture = study_ready
    document = dict(fixture['study'], off_completion=revise_scores(fixture, metric))
    completion = json.loads(Path(document['off_completion']['path']).read_text())
    report = json.loads(Path(completion['fresh_replay']['path']).read_text())
    report['passed'] = False
    completion['fresh_replay'] = write(Path(completion['fresh_replay']['path']), report)
    document['off_completion'] = write(Path(document['off_completion']['path']), completion)
    binding = write(Path(fixture['study_binding']['path']), document)
    with pytest.raises(ValueError):
        fixture['study_module'].authorize(fixture['bundle'], fixture['context'], binding)
    assert fixture['events'] == []


@pytest.mark.parametrize('metric', ['accuracy', 'auc'])
def test_study_completion_records_observed_anchor_and_separate_replay_schema(study_ready, monkeypatch, metric):
    fixture = study_ready
    module = fixture['study_module']
    document = dict(fixture['study'], off_completion=revise_scores(fixture, metric))
    binding = write(Path(fixture['study_binding']['path']), document)
    expected_baseline = module.authorize(fixture['bundle'], fixture['context'], binding)['anchor_baseline']
    assert expected_baseline['original_targets_met'] is False
    training_root = fixture['context']['run_dir'] / 'training'
    before = {path: path.read_bytes() for path in training_root.rglob('*') if path.is_file()}
    legacy_bytes = Path(fixture['module'].__file__).read_bytes()
    invoked = []

    def training(context, enabled, gate, *, prepared):
        invoked.append((enabled, gate, prepared['table_binding']))
        assert context is fixture['context']
        assert enabled is True
        assert prepared['table_binding'] == document['registration']
        return on_products(fixture, prepared, monkeypatch)

    monkeypatch.setattr(fixture['bundle'].service, 'run_training', training)

    def child(command, **kwargs):
        assert kwargs == dict(check=True)
        assert command[0] == sys.executable and Path(command[1]) == SCRIPT
        assert command[command.index('--phase') + 1] == 'replay'
        assert command[command.index('--on-request') + 1] == binding['path']
        assert command[command.index('--on-request-sha256') + 1] == binding['sha256']
        result_binding = bridge_binding(training_root / 'on/train.json')
        assert command[command.index('--train-result') + 1] == result_binding['path']
        assert command[command.index('--train-result-sha256') + 1] == result_binding['sha256']
        report = module.replay(fixture['bundle'], fixture['context'], binding, result_binding)
        assert report['schema_version'] == 'direct_reference_on_study_replay_v1'
        report['pid'] = os.getpid() + 1
        write(training_root / 'on/fresh-replay.json', report)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, 'run', child)
    args = SimpleNamespace(source_dir=Path('/restored/code'), request=Path('/restored/request.json'),
                           request_sha256='e' * 64, on_request=Path(binding['path']),
                           on_request_sha256=binding['sha256'], run_dir=fixture['context']['run_dir'], device='cuda')
    result = module.train(fixture['bundle'], fixture['context'], binding, args)
    assert result['schema_version'] == 'direct_reference_on_study_result_v1'
    assert result['passed'] is True and result['epochs_completed'] == 100
    assert result['anchor_baseline'] == expected_baseline
    assert json.loads((training_root / 'on/on.json').read_text()) == result
    assert invoked == [(True, fixture['gate_binding'], document['registration'])]
    assert all(path.read_bytes() == content for path, content in before.items())
    assert Path(fixture['module'].__file__).read_bytes() == legacy_bytes
    completion = json.loads(Path(document['off_completion']['path']).read_text())
    off_report = module.document(completion['fresh_replay'])
    assert off_report['schema_version'] == 'direct_reference_control_replay_v1'
    report = module.document(result['fresh_replay'])
    assert report['schema_version'] == 'direct_reference_on_study_replay_v1'
    assert report['n_images'] == 222 and report['buffers_unchanged'] is True
    assert report['max_abs_difference'] == 0.
    assert report['image_ids'] == fixture['rows']['val']['image_id'].tolist()
    assert report['on_request'] == binding
    assert report['launcher_sha256'] == bridge_binding(SCRIPT)['sha256']
    assert report['identities']['registration_sha256'] == fixture['result']['identities']['registration_sha256']
    assert report['identities']['preprocess_sha256'] == fixture['result']['identities']['preprocess_sha256']


@pytest.mark.parametrize('schema', ['direct_reference_on_replay_v1', 'direct_reference_control_replay_v1'])
def test_study_on_replay_does_not_accept_legacy_or_off_schema(study_ready, schema):
    fixture = study_ready
    module = fixture['study_module']
    authorized = module.authorize(fixture['bundle'], fixture['context'], fixture['study_binding'])
    chosen = module.selected(fixture['bundle'], fixture['context'], authorized, fixture['result_binding'], 'off')
    report = dict(fixture['fresh'], schema_version=schema, arm='on', on_request=fixture['study_binding'],
                  launcher_sha256=bridge_binding(SCRIPT)['sha256'])
    with pytest.raises(ValueError):
        module.check_replay(fixture['bundle'], report, chosen, fixture['result_binding'], fixture['study_binding'], 'on')
    assert fixture['events'] == []
