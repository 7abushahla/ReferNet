import importlib.util
import getpass
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def launcher():
    path = ROOT / 'scripts/queue_protocol_replications.py'
    assert path.is_file(), 'replication metadata controller is missing'
    specification = importlib.util.spec_from_file_location('replication_queue', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_metadata_controller_interface():
    module = launcher()
    assert callable(module.once) and callable(module.watch) and callable(module.main)


def bind(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return bind(path)


@pytest.fixture
def ready(tmp_path, monkeypatch):
    module = launcher()
    source = tmp_path / 'source'
    for protocol in ('a', 'b'):
        path = source / 'scripts' / f'protocol_{protocol}_study.py'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('SYNTHETIC = True\n')
        wrapper = source / 'scripts/cluster' / f'protocol_{protocol}_study.sh'
        wrapper.parent.mkdir(exist_ok=True)
        wrapper.write_text('#!/bin/bash\nexit 0\n')
    source_sha = module.common.object_sha256(module.common.source_entries(source))
    user = getpass.getuser()
    submitted, states, commands = {}, {101: 'COMPLETED'}, []
    population = []

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == 'squeue':
            rows = population + [(job, state) for job, state in submitted.items()]
            return SimpleNamespace(stdout=''.join(f'{job}|{user}|{state}\n' for job, state in rows))
        if command[0] == 'sacct':
            job = int(command[command.index('-j') + 1])
            state = states.get(job, 'COMPLETED')
            if state == 'ACCOUNTING_PENDING':
                return SimpleNamespace(stdout='')
            return SimpleNamespace(stdout=f'{job}|{user}|{state}|{"0:0" if state == "COMPLETED" else "1:0"}|\n')
        assert command[0] == 'sbatch' and kwargs['cwd'] == str(tmp_path)
        job = 500 + len(submitted)
        submitted[job] = 'PENDING'
        return SimpleNamespace(stdout=f'{job};synthetic\n')

    monkeypatch.setattr(module.subprocess, 'run', run)

    def make_unit(protocol='A', index=1, review=True, smoke_job=101):
        folder = tmp_path / f'{protocol}-{index}'
        folder.mkdir()
        inputs = {}
        for name, suffix in (('records', '.csv'), ('split', '.csv'), ('enrollment', '.json'), ('root', '.json'), ('config', '.yaml')):
            path = folder / (name + suffix)
            path.write_text('{}')
            inputs[name] = bind(path)
        output = folder / 'run'
        request = dict(schema_version=f'protocol_{protocol.lower()}_request_v1', protocol=protocol,
            **{'fold' if protocol == 'A' else 'group': index}, source_sha256=source_sha,
            **inputs, weights=dict(path=str(folder / 'never-open-weights.pth'),
                sha256='da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618'),
            raw_root=str(folder / 'never-open-pixels'), output_root=str(output), cache_bytes=0)
        request_binding = put(folder / 'request.json', request)
        unit = dict(protocol=protocol, index=index, source_dir=str(source), source_sha256=source_sha,
            request=request_binding, wrapper=bind(source / 'scripts/cluster' / f'protocol_{protocol.lower()}_study.sh'),
            review=None, jobs=dict(smoke=smoke_job, off=None, on=None, evaluate=None))
        if smoke_job is None:
            return unit
        registration = put(output / 'registration.json', dict(synthetic=True))
        reports, smokes = {}, {}
        for arm in ('off', 'on'):
            directory = output / 'gate' / arm
            predictions = directory / 'val_predictions.csv'
            predictions.parent.mkdir(parents=True)
            predictions.write_text('image_id,bonafide_score\nsynthetic,.1\n')
            config = dict(protocol=protocol, arm=arm, reference_enabled=arm == 'on', epochs=100,
                batch_size=8, eval_batch_size=1, seed=42, **{'fold' if protocol == 'A' else 'group': index},
                precision=dict(dtype='float32', amp=False, cudnn_tf32=False, matmul_tf32=False))
            smoke = dict(passed=True, no_test_open=True, pid=10, request=request_binding, config=config,
                identities=dict(source_sha256=source_sha), stats=dict(n_images=32, optimizer_steps=4),
                summary=dict(buffers_unchanged=True, image_loss=.1),
                batch_invariance=dict(passed=True, buffers_unchanged=True, n_images=4),
                predictions=bind(predictions), checkpoint=dict(path=str(directory / 'selected.pt'), sha256='a' * 64),
                fresh_initial_state_sha256='b' * 64, initialization=dict(reference_enabled=arm == 'on', tensors='c' * 64))
            smoke_binding = put(directory / 'smoke.json', smoke)
            fresh = dict(passed=True, fresh_process=True, no_test_open=True, pid=20, parent_result=smoke_binding,
                request=request_binding, checkpoint=smoke['checkpoint'], identities=smoke['identities'], n_images=32,
                summary=smoke['summary'], comparison=dict(passed=True))
            reports[arm] = dict(smoke=smoke_binding, replay=put(directory / 'fresh-replay.json', fresh))
            smokes[arm] = smoke
        gate = dict(schema_version=f'protocol_{protocol.lower()}_training_gate_v1', protocol=protocol,
            **{'fold' if protocol == 'A' else 'group': index}, source_sha256=source_sha, request=request_binding,
            records_sha256=inputs['records']['sha256'], split_sha256=inputs['split']['sha256'],
            enrollment_sha256=inputs['enrollment']['sha256'], weights_sha256=request['weights']['sha256'],
            registration=registration, registration_sha256=registration['sha256'], arm_reports=reports,
            passed=True, no_test_open=True, both_arm_smoke=True, replay=True, batch_invariance=True,
            recipe_sha256=module.common.object_sha256(smokes['off']['config']),
            accounting=dict(n_train=35, n_val=33, batches_per_epoch=4, images_per_epoch=32, total_optimizer_steps=400))
        gate_binding = put(output / 'gate/gate.json', gate)
        if review:
            note = folder / 'review.md'
            note.write_text('Synthetic independent review, not a scientific outcome.\n')
            unit['review'] = put(folder / 'review.json', dict(schema_version='protocol_replication_review_v1',
                passed=True, independent_review=True, reviewer='synthetic-reviewer', unit=module.identity(unit),
                gate=gate_binding, note=bind(note)))
        return unit

    unit = make_unit()
    authority = dict(schema_version='protocol_replication_queue_v1', user=user,
        priority_policy=module.POLICY, ledger_path=str(tmp_path / 'central.jsonl'), submit_cwd=str(tmp_path), units=[unit])

    def binding():
        return put(tmp_path / 'queue.json', authority)

    return dict(module=module, authority=authority, binding=binding, make_unit=make_unit,
                population=population, states=states, submitted=submitted, commands=commands, root=tmp_path)


def test_three_jobs_use_reviewed_gate_afterok_and_fixed_nice(ready):
    result = ready['module'].once(ready['binding']())
    assert result['status'] == 'submitted' and result['job_ids'] == [500, 501, 502]
    calls = [command for command in ready['commands'] if command[0] == 'sbatch']
    assert len(calls) == 3
    assert calls[0][2] == calls[1][2] == '--nice=100'
    assert calls[2][2:4] == ['--nice=0', '--dependency=afterok:500:501']
    assert calls[0][-3:-1] == ['train', 'off'] and calls[1][-3:-1] == ['train', 'on']
    assert calls[2][-3:-1] == ['evaluate', 'off']
    assert calls[0][-1] == calls[1][-1] == calls[2][-1]
    records = ready['module'].read_ledger(Path(ready['authority']['ledger_path']))[0]
    assert [record['status'] for record in records] == ['collected', 'intent', 'submitted', 'submitted', 'submitted', 'complete']
    assert len(records[1]['plans']) == 3 and records[1]['admission_budget'] == 3


def test_repeat_never_resubmits_and_observes_all_live_jobs(ready):
    bound = ready['binding']()
    ready['module'].once(bound)
    result = ready['module'].once(bound)
    assert result['status'] == 'waiting' and result['units'] == {'A:1': 'train_pending'}
    assert len(ready['submitted']) == 3 and result['live_user_jobs'] == 3


@pytest.mark.parametrize('count,expected', [(25, 'submitted'), (26, 'waiting'), (28, 'waiting'), (30, 'waiting')])
def test_all_user_capacity_includes_thesis_and_watcher_reservation(ready, count, expected):
    ready['population'].extend((9000 + index, 'RUNNING') for index in range(count))
    result = ready['module'].once(ready['binding']())
    assert result['status'] == expected
    assert len(ready['submitted']) == (3 if expected == 'submitted' else 0)
    assert not any(command[0] in ('scancel', 'scontrol', 'ssh') for command in ready['commands'])


def test_over_thirty_halts_without_touching_existing_jobs(ready):
    ready['population'].extend((9000 + index, 'CONFIGURING') for index in range(31))
    with pytest.raises(ValueError, match='ceiling'):
        ready['module'].once(ready['binding']())
    assert not ready['submitted']


def test_actual_gate_without_independent_review_is_collected_not_admitted(ready):
    ready['authority']['units'][0]['review'] = None
    result = ready['module'].once(ready['binding']())
    assert result['units'] == {'A:1': 'independent_review_pending'} and not ready['submitted']


def test_b1_is_eligible_b0_is_excluded_and_a_has_priority(ready):
    ready['authority']['units'].append(ready['make_unit']('B', 1, smoke_job=102))
    ready['states'][102] = 'COMPLETED'
    assert ready['module'].once(ready['binding']())['unit'] == 'A:1'
    assert ready['module'].once(ready['binding']())['unit'] == 'B:1'
    assert [command[2] for command in ready['commands'] if command[0] == 'sbatch'][3:] == ['--nice=1000', '--nice=1000', '--nice=0']
    ready['authority']['units'][1]['index'] = 0
    with pytest.raises(ValueError):
        ready['module'].once(ready['binding']())


def test_a_smoke_precedes_all_full_training_and_uses_six_arguments(ready):
    ready['authority']['units'].append(ready['make_unit']('A', 2, review=False, smoke_job=None))
    result = ready['module'].once(ready['binding']())
    assert result['phase'] == 'smoke' and result['unit'] == 'A:2'
    call = next(command for command in ready['commands'] if command[0] == 'sbatch')
    assert call[2] == '--nice=0' and len(call[4:]) == 6 and call[-3:] == ['smoke', 'off', '-']


@pytest.mark.parametrize('failure', ['source', 'request', 'note', 'review', 'gate', 'smoke', 'fresh', 'predictions'])
def test_bound_metadata_corruption_never_submits(ready, failure):
    module = ready['module']
    unit = ready['authority']['units'][0]
    request = module.document(unit['request'])
    review = module.document(unit['review'])
    gate = module.document(review['gate'])
    if failure == 'source':
        path = Path(unit['source_dir']) / 'scripts/protocol_a_study.py'
    elif failure == 'request':
        path = Path(unit['request']['path'])
    elif failure in ('note', 'gate'):
        path = Path(review[failure]['path'])
    elif failure == 'review':
        path = Path(unit['review']['path'])
    elif failure == 'predictions':
        smoke = module.document(gate['arm_reports']['on']['smoke'])
        path = Path(smoke['predictions']['path'])
    else:
        path = Path(gate['arm_reports']['on']['smoke' if failure == 'smoke' else 'replay']['path'])
    path.write_text('corrupted synthetic metadata')
    with pytest.raises(ValueError):
        module.once(ready['binding']())
    assert not ready['submitted']


@pytest.mark.parametrize('call_number', [1, 2, 3])
def test_partial_or_ambiguous_submit_reserves_without_retry(ready, monkeypatch, call_number):
    original = ready['module'].subprocess.run
    attempts = []

    def broken(command, **kwargs):
        if command[0] == 'sbatch':
            attempts.append(command)
            if len(attempts) == call_number:
                return SimpleNamespace(stdout='ambiguous response')
        return original(command, **kwargs)

    monkeypatch.setattr(ready['module'].subprocess, 'run', broken)
    bound = ready['binding']()
    with pytest.raises(ValueError):
        ready['module'].once(bound)
    with pytest.raises(ValueError, match='manual reconciliation'):
        ready['module'].once(bound)
    assert len(attempts) == call_number


def test_failed_parent_blocks_no_resubmit(ready):
    ready['states'][101] = 'FAILED'
    with pytest.raises(ValueError, match='parent failed'):
        ready['module'].once(ready['binding']())
    assert not ready['submitted']


def test_missing_accounting_waits_and_reserves_capacity(ready):
    ready['states'][101] = 'ACCOUNTING_PENDING'
    result = ready['module'].once(ready['binding']())
    assert result['units'] == {'A:1': 'smoke_pending'} and result['counted_live_and_unseen'] == 1
    assert not ready['submitted']


def test_no_weights_checkpoint_pixels_or_scientific_imports(ready, monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.suffix not in ('.pt', '.pth', '.png', '.jpg')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)
    assert ready['module'].once(ready['binding']())['status'] == 'submitted'
    source = (ROOT / 'scripts/queue_protocol_replications.py').read_text()
    assert 'import torch' not in source and 'import refer_or_clear' not in source


def test_watch_uses_mocked_clock_and_nonzero_timeout(ready, monkeypatch, capsys):
    module = ready['module']
    now, sleeps = [0], []
    monkeypatch.setattr(module, 'once', lambda binding: dict(status='waiting', units={'A:1': 'review_pending'}))

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    with pytest.raises(TimeoutError):
        module.watch(ready['binding'](), poll_seconds=300, timeout_seconds=601, sleeper=sleep, clock=lambda: now[0])
    assert sleeps == [300, 300, 1] and not ready['submitted']
    monkeypatch.setattr(module, 'watch', lambda *args, **kwargs: (_ for unused in ()).throw(TimeoutError('deadline')))
    bound = ready['binding']()
    assert module.main(['--authority', bound['path'], '--authority-sha256', bound['sha256'], '--watch']) == 2
    assert '"status": "timeout"' in capsys.readouterr().err
