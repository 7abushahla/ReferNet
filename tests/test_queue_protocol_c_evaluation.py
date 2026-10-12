import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from test_queue_protocol_a_evaluation import ready as a_ready, put, bind


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/queue_protocol_c_evaluation.py'


def launcher():
    assert SCRIPT.is_file(), 'Protocol C metadata controller is missing'
    spec = importlib.util.spec_from_file_location('protocol_c_queue_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_shot_and_watch_interfaces_exist():
    module = launcher()
    assert callable(module.once) and callable(module.watch) and callable(module.main)


@pytest.fixture
def ready(a_ready):
    module = launcher()
    root = a_ready['tmp']
    frozen = a_ready['source']
    for index in range(33):
        (frozen / f'src/refer_or_clear/original_{index:02d}.py').write_text('VALUE = 1\n')
    entries = module.common.source_entries(frozen)
    original = {name: a_ready['template'][name] for name in ('records', 'split', 'enrollment', 'weights', 'config')}
    target_split = dict(path=str(root / 'never-open-c-split.csv'), sha256='f' * 64)
    identities = dict(records_sha256=original['records']['sha256'], source_split_sha256=original['split']['sha256'],
                      target_split_sha256=target_split['sha256'])
    target = put(root / 'c-target-enrollment.json', dict(schema_version='protocol_c_target_enrollment_v1', protocol='C', **identities))
    products = dict(median=dict(path=str(root / 'never-open-median.png'), sha256='a' * 64))
    enrollment = put(root / 'c-support.json', dict(schema_version='protocol_c_enrollment_v1', protocol='C', **identities,
                     target_enrollment_authority=target, output_root=str(root / 'original-c-bank'),
                     banks={'fantasyid/headline': {'enrollment': {'products': products}}}))
    relocation = put(root / 'c-product-root.json', dict(schema_version='protocol_c_product_root_v1', protocol='C',
                     enrollment_manifest_sha256=enrollment['sha256'], original_root=str(root / 'original-c-bank'),
                     relocated_root=str(root / 'relocated-c-bank')))
    template = dict(schema_version='protocol_c_current_score_v1', protocol='C', seed=42, eval_batch_size=1,
                    training_source_sha256=module.common.TRAINING_SOURCE_SHA256,
                    evaluation_source_sha256=module.common.object_sha256(entries),
                    critical_source_hashes={entry['path']: entry['sha256'] for entry in entries},
                    source_selection=original, target_split=target_split, target_enrollment=target,
                    enrollment=enrollment, root=relocation, raw_root=str(root / 'never-open-c-raw'),
                    output_root=str(root / 'c-scoring-output'))
    authority = dict(a_ready['authority'], schema_version='protocol_c_evaluation_queue_v1',
                     score_template=put(root / 'reviewed-c-template.json', template),
                     request_path=str(root / 'requests/c-score.json'), ledger_path=str(root / 'ledger/c-submissions.jsonl'))
    a_ready.update(module=module, template=template, authority=authority,
                   binding=put(root / 'c-queue-authority.json', authority))
    return a_ready


def update_template(ready):
    authority = ready['authority']
    authority['score_template'] = put(Path(authority['score_template']['path']), ready['template'])
    ready['binding'] = put(Path(ready['binding']['path']), authority)


def test_exact_c_request_nested_original_selection_and_six_wrapper_arguments(ready):
    module = ready['module']
    result = module.once(ready['binding'])
    request = module.common.document(result['request'])
    assert request['schema_version'] == 'protocol_c_current_score_v1' and request['protocol'] == 'C'
    assert 'arms' not in request and 'fold' not in request
    source = dict(request['source_selection'])
    arms = source.pop('arms')
    assert source == ready['template']['source_selection']
    assert set(source) == {'records', 'split', 'enrollment', 'weights', 'config'}
    assert not {'schema_version', 'protocol', 'fold', 'root', 'raw_root'} & set(source)
    assert source['split'] != request['target_split'] and source['enrollment'] != request['enrollment']
    for arm in ('off', 'on'):
        assert arms[arm] == dict(train_result=bind(ready['paths'][arm]['train']), completion=bind(ready['paths'][arm]['completion']),
                                fresh_replay=bind(ready['paths'][arm]['replay']))
    authority = ready['authority']
    expected = ['sbatch', '--parsable', authority['wrapper']['path'], authority['evaluation_source_dir'],
                authority['request_path'], result['request']['sha256'], authority['inputs_dir'],
                authority['inputs_manifest']['path'], authority['inputs_manifest']['sha256']]
    assert ready['calls'][-1][0] == result['command'] == expected
    assert module.once(ready['binding']) == result and len(ready['calls']) == 2
    records = [json.loads(line) for line in Path(authority['ledger_path']).read_text().splitlines()]
    assert [row['status'] for row in records] == ['intent', 'submitted']
    assert all(row['schema_version'] == 'protocol_c_evaluation_queue_ledger_v1' for row in records)


def test_controller_does_not_require_a_test_results_or_performance_gate(ready, monkeypatch):
    module = ready['module']
    for arm in ('off', 'on'):
        path = ready['paths'][arm]['replay']
        report = json.loads(path.read_text())
        report['summary'].update(accuracy=.4, auc=.3)
        replay_binding = put(path, report)
        completion_path = ready['paths'][arm]['completion']
        completed = json.loads(completion_path.read_text())
        completed['fresh_replay'] = replay_binding
        put(completion_path, completed)
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.suffix not in {'.pt', '.pth', '.csv', '.png', '.jpg'}
        assert not path.name.startswith('test_') and path.name != 'evaluation.json'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)
    assert module.once(ready['binding'])['status'] == 'submitted'


@pytest.mark.parametrize('change', ['a_template', 'fake_a_authority', 'target_split_as_source', 'arms', 'fold', 'core_count', 'core_hash'])
def test_c_template_and_original_source_contract_fail_closed(ready, change):
    template = ready['template']
    if change == 'a_template':
        template['schema_version'] = 'protocol_a_current_score_v1'
    elif change == 'fake_a_authority':
        template['source_selection']['schema_version'] = 'protocol_a_evaluation_v1'
    elif change == 'target_split_as_source':
        template['source_selection']['split'] = template['target_split']
    elif change == 'arms':
        template['source_selection']['arms'] = {}
    elif change == 'fold':
        template['fold'] = 0
    elif change == 'core_count':
        template['critical_source_hashes'].pop(next(iter(template['critical_source_hashes'])))
    else:
        template['critical_source_hashes'][next(iter(template['critical_source_hashes']))] = '0' * 64
    update_template(ready)
    with pytest.raises(ValueError):
        ready['module'].once(ready['binding'])
    assert not any(call[0][0] == 'sbatch' for call in ready['calls'])


@pytest.mark.parametrize('artifact', ['train', 'completion', 'replay'])
def test_parent_completion_missing_fails_before_submission(ready, artifact):
    ready['paths']['on'][artifact].unlink()
    with pytest.raises(FileNotFoundError):
        ready['module'].once(ready['binding'])
    assert len(ready['calls']) == 1 and ready['calls'][0][0][0] == 'sacct'


@pytest.mark.parametrize('name', ['target_enrollment', 'enrollment', 'root'])
def test_support_authorities_required_before_c_submission(ready, name):
    Path(ready['template'][name]['path']).unlink()
    with pytest.raises(FileNotFoundError):
        ready['module'].once(ready['binding'])
    assert ready['calls'] == []


def test_unbuilt_c_enrollment_rejected(ready):
    binding = ready['template']['enrollment']
    value = json.loads(Path(binding['path']).read_text())
    value['banks']['fantasyid/headline']['enrollment'].pop('products')
    ready['template']['enrollment'] = put(Path(binding['path']), value)
    update_template(ready)
    with pytest.raises(ValueError, match='completed C support products'):
        ready['module'].once(ready['binding'])
    assert ready['calls'] == []


@pytest.mark.parametrize('state,exit_code', [('FAILED', '1:0'), ('CANCELLED', '0:15'), ('COMPLETED', '1:0')])
def test_parent_failure_no_c_scoring_submission(ready, monkeypatch, state, exit_code):
    monkeypatch.setattr(ready['module'].common.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(stdout=f'38152|{state}|{exit_code}\n'))
    with pytest.raises(ValueError, match='did not complete successfully'):
        ready['module'].once(ready['binding'])
    assert not Path(ready['authority']['request_path']).exists()


def test_watch_waits_then_submits_c_only_once(ready, monkeypatch):
    module = ready['module']
    original = module.common.subprocess.run
    states = iter(['RUNNING', 'COMPLETED'])

    def run(command, **kwargs):
        if command[0] == 'sacct':
            ready['calls'].append((command, kwargs))
            return SimpleNamespace(stdout=f'38152|{next(states)}|0:0\n')
        return original(command, **kwargs)

    monkeypatch.setattr(module.common.subprocess, 'run', run)
    seconds, delays = [0], []

    def sleep(delay):
        seconds[0] += delay
        delays.append(delay)

    result = module.watch(ready['binding'], sleeper=sleep, clock=lambda: seconds[0])
    assert result['status'] == 'submitted' and delays == [300]
    assert [call[0][0] for call in ready['calls']] == ['sacct', 'sacct', 'sbatch']
    assert module.watch(ready['binding'], sleeper=sleep, clock=lambda: seconds[0]) == result
    assert len(ready['calls']) == 3 and delays == [300]


def test_watch_timeout_explicit_and_non_submitting(ready, monkeypatch):
    module = ready['module']
    monkeypatch.setattr(module.common.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout='38152|RUNNING|0:0\n'))
    seconds, delays = [0], []

    def sleep(delay):
        seconds[0] += delay
        delays.append(delay)

    with pytest.raises(module.WatchTimeout) as outcome:
        module.watch(ready['binding'], timeout_seconds=650, sleeper=sleep, clock=lambda: seconds[0])
    assert delays == [300, 300, 50]
    assert outcome.value.report['protocol'] == 'C' and outcome.value.report['status'] == 'timeout'
    assert outcome.value.report['submission_attempted'] is False
    assert not Path(ready['authority']['request_path']).exists()


@pytest.mark.parametrize('failure', ['nonzero', 'ambiguous', 'transport'])
def test_uncertain_c_submission_cannot_be_retried(ready, monkeypatch, failure):
    module = ready['module']
    original = module.common.subprocess.run

    def run(command, **kwargs):
        if command[0] == 'sacct':
            return original(command, **kwargs)
        ready['calls'].append((command, kwargs))
        if failure == 'nonzero':
            raise subprocess.CalledProcessError(1, command)
        if failure == 'transport':
            raise OSError('uncertain submit')
        return SimpleNamespace(stdout='unknown job response')

    monkeypatch.setattr(module.common.subprocess, 'run', run)
    with pytest.raises((ValueError, OSError, subprocess.CalledProcessError)):
        module.once(ready['binding'])
    with pytest.raises(ValueError, match='no resubmission'):
        module.watch(ready['binding'], sleeper=lambda *args: pytest.fail('retried failed submission'), clock=lambda: 0)
    assert len(ready['calls']) == 2


def test_existing_intent_blocks_c_submission(ready):
    put(Path(ready['authority']['ledger_path']), dict(status='intent'))
    with pytest.raises(ValueError, match='no resubmission'):
        ready['module'].once(ready['binding'])
    assert ready['calls'] == []


def test_cli_timeout_exits_two_with_c_status(ready, monkeypatch, capsys):
    module = ready['module']
    monkeypatch.setattr(module.common.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout='38152|PENDING|0:0\n'))
    seconds = [0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: seconds[0])
    monkeypatch.setattr(module.time, 'sleep', lambda delay: seconds.__setitem__(0, seconds[0] + delay))
    with pytest.raises(SystemExit) as outcome:
        module.main(['--authority', ready['binding']['path'], '--authority-sha256', ready['binding']['sha256'],
                     '--watch', '--timeout-seconds', '300'])
    assert outcome.value.code == 2
    assert json.loads(capsys.readouterr().err.splitlines()[-1])['protocol'] == 'C'
