import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/queue_protocol_a_evaluation.py'


def launcher():
    assert SCRIPT.is_file(), 'Protocol A metadata controller is missing'
    spec = importlib.util.spec_from_file_location('protocol_a_queue_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_one_shot_interface_exists():
    module = launcher()
    assert callable(module.once)
    assert callable(module.main)


def bind(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return bind(path)


@pytest.fixture
def ready(tmp_path, monkeypatch):
    module = launcher()
    source = tmp_path / 'frozen-evaluation'
    scientific_file = source / 'src/refer_or_clear/data/direct_reference.py'
    scientific_file.parent.mkdir(parents=True)
    scientific_file.write_text('VALUE = 1\n')
    inputs = tmp_path / 'durable-inputs'
    inputs.mkdir()
    manifest_path = tmp_path / 'inputs.sha256'
    names = ('config.yaml', 'records.csv', 'split.csv', 'enrollment.json', 'product-root.json', 'weights.pth', 'request.json')
    manifest_path.write_text(''.join(f'{"c" * 64}  {name}\n' for name in names))
    wrapper = tmp_path / 'reviewed-gpu-wrapper.sh'
    wrapper.write_text('#!/bin/bash\nexit 0\n')
    run = tmp_path / 'original-run'
    identifiers = dict(records_sha256='a' * 64, split_sha256='b' * 64, enrollment_sha256='c' * 64,
                       weights_sha256='d' * 64, registration_sha256='e' * 64, phase_authority_sha256='f' * 64,
                       preprocess_sha256='1' * 64, source_sha256=module.TRAINING_SOURCE_SHA256)
    paths = {}
    for arm in ('off', 'on'):
        directory = run / 'training' / arm
        ids = dict(identifiers, config_sha256=('2' if arm == 'off' else '3') * 64,
                   initialization_sha256=('4' if arm == 'off' else '5') * 64)
        checkpoint = dict(path=str(directory / 'best.pt'), sha256='6' * 64)
        registration = dict(path=str(run / 'training/shared/registration.json'), sha256=identifiers['registration_sha256'])
        original_authority = dict(path=str(tmp_path / f'{arm}-authority.json'), sha256='7' * 64)
        train = dict(passed=True, phase='train', diagnostic_only=False, epochs_completed=100, optimizer_steps=22200,
                     role='val', arm=arm, scheduler=dict(total_updates=22200, warmup_updates=1000),
                     selected_epoch=66, fresh_initial_state_sha256='8' * 64, identities=ids, checkpoint=checkpoint,
                     registration=registration, predictions=dict(path=str(directory / 'val_predictions.csv'), sha256='9' * 64),
                     replay=dict(passed=True, fresh_process=False, phase='train', n_images=222))
        train_binding = put(directory / 'train.json', train)
        key = 'control_request' if arm == 'off' else 'on_request'
        replay = dict(schema_version='direct_reference_control_replay_v1' if arm == 'off' else 'direct_reference_on_study_replay_v1',
                      passed=True, fresh_process=True, phase='train', arm=arm, n_images=222, buffers_unchanged=True,
                      train_result=train_binding, checkpoint=checkpoint, identities=ids, pid=1234,
                      history=dict(path=str(directory / 'log.csv'), sha256='a' * 64),
                      image_ids=[f'val-{index:03d}' for index in range(222)], summary=dict(n_images=222, buffers_unchanged=True),
                      replay=dict(passed=True, n_images=222), launcher_sha256='b' * 64, **{key: original_authority})
        replay_binding = put(directory / 'fresh-replay.json', replay)
        completion = dict(schema_version='direct_reference_control_result_v1' if arm == 'off' else 'direct_reference_on_study_result_v1',
                          passed=True, arm=arm, epochs_completed=100, train_result=train_binding, checkpoint=checkpoint,
                          identities=ids, fresh_replay=replay_binding, launcher_sha256='b' * 64,
                          scientific_source_sha256=module.TRAINING_SOURCE_SHA256, **{key: original_authority})
        if arm == 'off':
            completion['phase'] = 'train'
        paths[arm] = dict(train=directory / 'train.json', replay=directory / 'fresh-replay.json',
                          completion=directory / ('control.json' if arm == 'off' else 'on.json'))
        put(paths[arm]['completion'], completion)
    template = dict(schema_version='protocol_a_current_score_v1', protocol='A', fold=0, seed=42, eval_batch_size=1,
                    training_source_sha256=module.TRAINING_SOURCE_SHA256,
                    evaluation_source_sha256=module.object_sha256(module.source_entries(source)),
                    critical_source_hashes={'src/refer_or_clear/data/direct_reference.py': bind(scientific_file)['sha256']},
                    raw_root=str(tmp_path / 'never-open-raw'), output_root=str(tmp_path / 'new-scoring-output'))
    for field, filename, identity in (('records', 'records.csv', 'records_sha256'), ('split', 'split.csv', 'split_sha256'),
                                    ('enrollment', 'enrollment.json', 'enrollment_sha256'), ('weights', 'weights.pth', 'weights_sha256')):
        template[field] = dict(path=str(inputs / filename), sha256=identifiers[identity])
    for field in ('root', 'config'):
        template[field] = dict(path=str(inputs / f'{field}.json'), sha256='c' * 64)
    template_binding = put(tmp_path / 'reviewed-template.json', template)
    authority = dict(schema_version='protocol_a_evaluation_queue_v1', parent_job_id=38152,
                     evaluation_source_dir=str(source), score_template=template_binding, wrapper=bind(wrapper),
                     inputs_dir=str(inputs), inputs_manifest=bind(manifest_path), run_dir=str(run),
                     request_path=str(tmp_path / 'requests/score.json'), ledger_path=str(tmp_path / 'ledger/submissions.jsonl'),
                     submit_cwd=str(tmp_path))
    authority_binding = put(tmp_path / 'queue-authority.json', authority)
    calls = []

    def command(arguments, **kwargs):
        calls.append((arguments, kwargs))
        if arguments[0] == 'sacct':
            assert arguments == ['sacct', '-n', '-P', '-X', '-j', '38152', '--format=JobIDRaw,State%40,ExitCode']
            return SimpleNamespace(stdout='38152|COMPLETED|0:0\n')
        assert arguments[0] == 'sbatch'
        return SimpleNamespace(stdout='38200;cluster\n')

    monkeypatch.setattr(module.subprocess, 'run', command)
    return dict(module=module, authority=authority, binding=authority_binding, template=template,
                paths=paths, calls=calls, source=source, scientific_file=scientific_file, tmp=tmp_path)


def rewrite_authority(ready):
    ready['binding'] = put(Path(ready['binding']['path']), ready['authority'])


def change_parent(ready, monkeypatch, state, exit_code='0:0'):
    def command(arguments, **kwargs):
        ready['calls'].append((arguments, kwargs))
        assert arguments[0] == 'sacct', 'sbatch called before successful parent'
        return SimpleNamespace(stdout=f'38152|{state}|{exit_code}\n')
    monkeypatch.setattr(ready['module'].subprocess, 'run', command)


def test_exact_six_argument_wrapper_submission_and_one_way_ledger(ready):
    module, authority = ready['module'], ready['authority']
    result = module.once(ready['binding'])
    assert result['status'] == 'submitted' and result['job_id'] == 38200
    expected = ['sbatch', '--parsable', authority['wrapper']['path'], authority['evaluation_source_dir'],
                authority['request_path'], result['request']['sha256'], authority['inputs_dir'],
                authority['inputs_manifest']['path'], authority['inputs_manifest']['sha256']]
    assert ready['calls'][1][0] == expected and result['command'] == expected
    assert ready['calls'][1][1] == dict(cwd=authority['submit_cwd'], check=True, capture_output=True, text=True)
    request = module.document(result['request'])
    assert {key: value for key, value in request.items() if key != 'arms'} == ready['template']
    for arm in ('off', 'on'):
        assert request['arms'][arm] == dict(train_result=bind(ready['paths'][arm]['train']),
                                          completion=bind(ready['paths'][arm]['completion']),
                                          fresh_replay=bind(ready['paths'][arm]['replay']))
    records = [json.loads(line) for line in Path(authority['ledger_path']).read_text().splitlines()]
    assert [record['status'] for record in records] == ['intent', 'submitted']
    assert records[1]['request'] == result['request']
    assert module.once(ready['binding']) == result
    assert len(ready['calls']) == 2


@pytest.mark.parametrize('state', ['PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED'])
def test_active_parent_never_reads_receipts_or_submits(ready, monkeypatch, state):
    change_parent(ready, monkeypatch, state)
    for paths in ready['paths'].values():
        for path in paths.values():
            path.unlink()
    result = ready['module'].once(ready['binding'])
    assert result == dict(status='waiting', parent_job_id=38152, parent_state=state)
    assert len(ready['calls']) == 1
    assert not Path(ready['authority']['request_path']).exists()
    assert not Path(ready['authority']['ledger_path']).exists()


@pytest.mark.parametrize('state,exit_code', [('FAILED', '1:0'), ('TIMEOUT', '0:0'), ('CANCELLED by 123', '0:15'),
                                          ('OUT_OF_MEMORY', '0:9'), ('COMPLETED', '1:0'), ('COMPLETED+', '0:0')])
def test_failure_states_fail_closed(ready, monkeypatch, state, exit_code):
    change_parent(ready, monkeypatch, state, exit_code)
    with pytest.raises(ValueError, match='did not complete successfully'):
        ready['module'].once(ready['binding'])
    assert len(ready['calls']) == 1 and not Path(ready['authority']['request_path']).exists()


@pytest.mark.parametrize('artifact', ['train', 'completion', 'replay'])
def test_completed_slurm_without_outcome_json_never_submits(ready, artifact):
    ready['paths']['on'][artifact].unlink()
    with pytest.raises(FileNotFoundError):
        ready['module'].once(ready['binding'])
    assert len(ready['calls']) == 1
    assert not Path(ready['authority']['ledger_path']).exists()


@pytest.mark.parametrize('artifact,field,value', [('train', 'epochs_completed', 99), ('train', 'optimizer_steps', 22199),
                                               ('train', 'epochs_completed', True), ('completion', 'passed', False),
                                               ('replay', 'fresh_process', False), ('replay', 'n_images', 221)])
def test_incomplete_or_wrong_receipts_never_submit(ready, artifact, field, value):
    path = ready['paths']['on'][artifact]
    value_document = json.loads(path.read_text())
    value_document[field] = value
    put(path, value_document)
    with pytest.raises(ValueError):
        ready['module'].once(ready['binding'])
    assert len(ready['calls']) == 1


def test_no_checkpoint_weights_csv_query_or_model_access(ready, monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.suffix not in {'.pt', '.pth', '.csv', '.png', '.jpg'}, f'Forbidden input opened: {path}'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)
    assert ready['module'].once(ready['binding'])['status'] == 'submitted'


@pytest.mark.parametrize('mutation', ['source', 'wrapper', 'manifest', 'critical', 'template_arms'])
def test_reviewed_source_template_wrapper_and_manifest_fail_closed(ready, mutation):
    if mutation == 'source':
        ready['scientific_file'].write_text('VALUE = 2\n')
    elif mutation == 'wrapper':
        Path(ready['authority']['wrapper']['path']).write_text('unreviewed wrapper')
    elif mutation == 'manifest':
        Path(ready['authority']['inputs_manifest']['path']).write_text('unreviewed inventory')
    else:
        template = ready['template']
        if mutation == 'critical':
            template['critical_source_hashes']['src/refer_or_clear/data/direct_reference.py'] = '0' * 64
        else:
            template['arms'] = {}
        ready['authority']['score_template'] = put(Path(ready['authority']['score_template']['path']), template)
        rewrite_authority(ready)
    with pytest.raises(ValueError):
        ready['module'].once(ready['binding'])
    assert ready['calls'] == []


@pytest.mark.parametrize('failure', ['nonzero', 'ambiguous', 'transport'])
def test_uncertain_sbatch_is_reserved_and_never_retried(ready, monkeypatch, failure):
    module = ready['module']
    original = module.subprocess.run

    def command(arguments, **kwargs):
        if arguments[0] == 'sacct':
            return original(arguments, **kwargs)
        ready['calls'].append((arguments, kwargs))
        if failure == 'nonzero':
            raise subprocess.CalledProcessError(1, arguments)
        if failure == 'transport':
            raise OSError('connection lost after possible submission')
        return SimpleNamespace(stdout='Submitted batch job 38200\n')

    monkeypatch.setattr(module.subprocess, 'run', command)
    with pytest.raises((ValueError, OSError, subprocess.CalledProcessError)):
        module.once(ready['binding'])
    records = [json.loads(line) for line in Path(ready['authority']['ledger_path']).read_text().splitlines()]
    assert [record['status'] for record in records] == ['intent', 'failed_or_unknown']
    with pytest.raises(ValueError, match='no resubmission'):
        module.once(ready['binding'])
    assert len(ready['calls']) == 2


def test_existing_intent_after_process_interruption_blocks_submission(ready):
    put(Path(ready['authority']['ledger_path']), dict(status='intent'))
    with pytest.raises(ValueError, match='no resubmission'):
        ready['module'].once(ready['binding'])
    assert ready['calls'] == []


def test_exclusive_existing_request_never_overwritten_or_submitted(ready):
    path = Path(ready['authority']['request_path'])
    put(path, dict(existing='do not replace'))
    original = path.read_bytes()
    with pytest.raises(ValueError, match='exclusive scoring request'):
        ready['module'].once(ready['binding'])
    assert path.read_bytes() == original and len(ready['calls']) == 1


def test_slurm_steps_do_not_override_parent_state(ready, monkeypatch):
    def command(arguments, **kwargs):
        assert arguments[0] == 'sacct'
        return SimpleNamespace(stdout='38152.batch|COMPLETED|0:0\n38152|RUNNING|0:0\n38152.extern|COMPLETED|0:0\n')
    monkeypatch.setattr(ready['module'].subprocess, 'run', command)
    assert ready['module'].once(ready['binding'])['status'] == 'waiting'


@pytest.mark.parametrize('stdout', ['', '38152.batch|COMPLETED|0:0\n', '38152|COMPLETED|0:0\n38152|COMPLETED|0:0\n'])
def test_missing_or_ambiguous_parent_accounting_rejected(ready, monkeypatch, stdout):
    monkeypatch.setattr(ready['module'].subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=stdout))
    with pytest.raises(ValueError, match='unambiguous parent'):
        ready['module'].once(ready['binding'])


def test_cli_requires_explicit_once_and_reports_submission(ready, capsys):
    module = ready['module']
    arguments = ['--authority', ready['binding']['path'], '--authority-sha256', ready['binding']['sha256']]
    with pytest.raises(SystemExit):
        module.main(arguments)
    module.main(arguments + ['--once'])
    assert json.loads(capsys.readouterr().out)['job_id'] == 38200


def test_watch_waits_metadata_only_then_submits_exactly_once(ready, monkeypatch):
    module = ready['module']
    original = module.subprocess.run
    states = iter(['PENDING', 'RUNNING', 'COMPLETED'])

    def command(arguments, **kwargs):
        if arguments[0] == 'sacct':
            ready['calls'].append((arguments, kwargs))
            return SimpleNamespace(stdout=f'38152|{next(states)}|0:0\n')
        return original(arguments, **kwargs)

    monkeypatch.setattr(module.subprocess, 'run', command)
    seconds, slept = [0], []

    def sleeper(delay):
        slept.append(delay)
        seconds[0] += delay

    result = module.watch(ready['binding'], sleeper=sleeper, clock=lambda: seconds[0])
    assert result['status'] == 'submitted' and result['job_id'] == 38200
    assert slept == [300, 300]
    assert [call[0][0] for call in ready['calls']] == ['sacct', 'sacct', 'sacct', 'sbatch']
    assert len(ready['calls'][-1][0][3:]) == 6
    assert module.watch(ready['binding'], sleeper=sleeper, clock=lambda: seconds[0]) == result
    assert len(ready['calls']) == 4 and slept == [300, 300]


def test_watch_timeout_is_bounded_non_success_and_no_submission(ready, monkeypatch, capsys):
    module = ready['module']
    change_parent(ready, monkeypatch, 'RUNNING')
    seconds, delays = [0], []

    def sleeper(delay):
        delays.append(delay)
        seconds[0] += delay

    with pytest.raises(module.WatchTimeout) as outcome:
        module.watch(ready['binding'], poll_seconds=300, timeout_seconds=650, sleeper=sleeper, clock=lambda: seconds[0])
    assert delays == [300, 300, 50]
    assert outcome.value.report == dict(status='timeout', parent_job_id=38152, polls=3,
                                       timeout_seconds=650, submission_attempted=False)
    assert len(ready['calls']) == 3
    assert not Path(ready['authority']['request_path']).exists()
    assert not Path(ready['authority']['ledger_path']).exists()
    assert len(capsys.readouterr().err.splitlines()) == 3


def test_watch_parent_failure_exits_without_sleep(ready, monkeypatch):
    change_parent(ready, monkeypatch, 'FAILED', '1:0')
    with pytest.raises(ValueError, match='did not complete successfully'):
        ready['module'].watch(ready['binding'], sleeper=lambda delay: pytest.fail('sleep after parent failure'), clock=lambda: 0)
    assert len(ready['calls']) == 1


def test_watch_existing_incomplete_ledger_never_retries(ready):
    put(Path(ready['authority']['ledger_path']), dict(status='intent'))
    with pytest.raises(ValueError, match='no resubmission'):
        ready['module'].watch(ready['binding'], sleeper=lambda delay: pytest.fail('sleep after ambiguous intent'), clock=lambda: 0)
    assert ready['calls'] == []


@pytest.mark.parametrize('poll,deadline', [(0, 72000), (True, 72000), (300, 0), (300, 72001), (300, True)])
def test_watch_deadline_and_poll_bounds(ready, poll, deadline):
    with pytest.raises(ValueError):
        ready['module'].watch(ready['binding'], poll_seconds=poll, timeout_seconds=deadline)
    assert ready['calls'] == []


def test_watch_cli_timeout_prints_status_and_exits_nonzero(ready, monkeypatch, capsys):
    module = ready['module']
    change_parent(ready, monkeypatch, 'PENDING')
    seconds = [0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: seconds[0])
    monkeypatch.setattr(module.time, 'sleep', lambda delay: seconds.__setitem__(0, seconds[0] + delay))
    with pytest.raises(SystemExit) as outcome:
        module.main(['--authority', ready['binding']['path'], '--authority-sha256', ready['binding']['sha256'],
                     '--watch', '--poll-seconds', '300', '--timeout-seconds', '300'])
    assert outcome.value.code == 2
    report = json.loads(capsys.readouterr().err.splitlines()[-1])
    assert report['status'] == 'timeout' and report['submission_attempted'] is False


def test_ledger_command_tampering_never_submits_again(ready):
    module = ready['module']
    module.once(ready['binding'])
    path = Path(ready['authority']['ledger_path'])
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]['command'][-1] = 'unreviewed-inputs-inventory'
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))
    with pytest.raises(ValueError, match='reviewed submission mismatch'):
        module.once(ready['binding'])
    assert len(ready['calls']) == 2
