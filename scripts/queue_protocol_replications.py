"""Metadata-only fixed A/B replication admissions with one central ledger.

No scientific modules are imported. The reviewed allocated wrappers remain
responsible for scientific receipt validation and all checkpoint/pixel access.
Unknown or partially submitted intents require manual reconciliation, never
automatic retry. Existing jobs are observed, never canceled or reprioritized.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_protocol_a_evaluation as common

POLICY = 'aws_capacity_a_first_v1'
JOB_NAMES = {'smoke', 'off', 'on', 'evaluate'}


def metadata(binding, suffixes=('.json', '.csv', '.md', '.sh', '.yaml', '.yml')):
    path = common.binding_path(binding)
    common.require(path.suffix in suffixes, 'metadata-only artifact required')
    payload = path.read_bytes()
    common.require(common.digest(payload) == binding['sha256'], 'metadata artifact SHA256 mismatch')
    return payload


def document(binding):
    return common.parse(metadata(binding, ('.json',)))


def capture(path):
    path = common.absolute(str(path))
    common.require(path.suffix == '.json', 'JSON receipt required')
    payload = path.read_bytes()
    return common.parse(payload), dict(path=str(path), sha256=common.digest(payload))


def unit_key(unit):
    return f"{unit['protocol']}:{unit['index']}"


def identity(unit):
    return {name: unit[name] for name in ('protocol', 'index', 'source_dir', 'source_sha256', 'request', 'wrapper')}


def validate_authority(binding):
    authority = document(binding)
    common.require(set(authority) == {'schema_version', 'user', 'priority_policy', 'ledger_path', 'submit_cwd', 'units'}
                   and authority['schema_version'] == 'protocol_replication_queue_v1'
                   and authority['priority_policy'] == POLICY and authority['user'] == getpass.getuser(),
                   'fixed replication policy and current Slurm user required')
    ledger = common.absolute(authority['ledger_path'])
    common.absolute(str(ledger) + '.lock')
    common.require(common.absolute(authority['submit_cwd']).is_dir(), 'existing submission directory required')
    units = authority['units']
    common.require(isinstance(units, list) and units and len(units) <= 20, 'bounded explicit A/B units required')
    keys, roots, jobs = set(), [], set()
    for unit in units:
        common.require(isinstance(unit, dict) and set(unit) == {'protocol', 'index', 'source_dir', 'source_sha256',
                       'request', 'wrapper', 'review', 'jobs'} and unit['protocol'] in ('A', 'B')
                       and type(unit['index']) is int and 1 <= unit['index'] <= 9, 'explicit A1..9/B1..9 replication unit required; B0 excluded')
        common.require(unit_key(unit) not in keys, 'duplicate replication unit')
        keys.add(unit_key(unit))
        source = common.absolute(unit['source_dir'])
        common.require(common.checksum(unit['source_sha256'])
                       and common.object_sha256(common.source_entries(source)) == unit['source_sha256'],
                       'frozen scientific source inventory mismatch')
        runner = source / 'scripts' / f"protocol_{unit['protocol'].lower()}_study.py"
        wrapper = source / 'scripts' / 'cluster' / f"protocol_{unit['protocol'].lower()}_study.sh"
        common.require(runner.is_file() and wrapper.is_file(), 'fixed study runner/wrapper source required')
        common.require(metadata(unit['wrapper'], ('.sh',)) == wrapper.read_bytes(), 'wrapper differs from frozen study source')
        request = document(unit['request'])
        index_name = 'fold' if unit['protocol'] == 'A' else 'group'
        common.fixed_fields(request, dict(schema_version=f"protocol_{unit['protocol'].lower()}_request_v1",
                            protocol=unit['protocol'], source_sha256=unit['source_sha256'],
                            **{index_name: unit['index']}), 'source/request/unit binding mismatch')
        root = common.absolute(request['output_root'])
        protected = [source, common.binding_path(unit['request']), common.binding_path(unit['wrapper']),
                     common.absolute(binding['path']), common.absolute(request['raw_root'])]
        for name in ('records', 'split', 'enrollment', 'root', 'metadata_relocation', 'target_enrollment', 'config'):
            if request.get(name) is not None:
                metadata(request[name])
                protected.append(common.binding_path(request[name]))
        protected.append(common.binding_path(request['weights']))
        common.require(request['weights']['sha256'] ==
                       'da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618',
                       'official initialization receipt required')
        if unit['review'] is not None:
            review = document(unit['review'])
            common.require(set(review) == {'schema_version', 'passed', 'independent_review', 'reviewer',
                           'unit', 'gate', 'note'} and review['schema_version'] == 'protocol_replication_review_v1'
                           and review['passed'] is True and review['independent_review'] is True
                           and isinstance(review['reviewer'], str) and review['reviewer'].strip()
                           and review['unit'] == identity(unit), 'externally bound independent unit review required')
            metadata(review['note'], ('.md',))
            common.require(common.binding_path(review['gate']) == root / 'gate/gate.json', 'canonical reviewed gate required')
            protected.extend([common.binding_path(unit['review']), common.binding_path(review['note'])])
        common.require(isinstance(unit['jobs'], dict) and set(unit['jobs']) == JOB_NAMES, 'explicit existing job slots required')
        common.require((unit['jobs']['off'] is None) == (unit['jobs']['on'] is None), 'existing partial train pair requires manual reconciliation')
        for job in unit['jobs'].values():
            common.require(job is None or type(job) is int and job > 0 and job not in jobs, 'unique positive existing job IDs required')
            if job is not None:
                jobs.add(job)
        for path in protected:
            common.require(not root.is_relative_to(path) and not path.is_relative_to(root)
                           and not ledger.is_relative_to(path) and not path.is_relative_to(ledger),
                           'queue/run output overlaps protected authority')
        common.require(not ledger.is_relative_to(root) and not root.is_relative_to(ledger)
                       and all(not root.is_relative_to(other) and not other.is_relative_to(root) for other in roots),
                       'central ledger or unit outputs overlap')
        roots.append(root)
    return authority


def user_jobs(user):
    result = subprocess.run(['squeue', '-h', '-r', '--user', user, '--format=%i|%u|%T'],
                            check=True, capture_output=True, text=True)
    jobs = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split('|')
        common.require(len(fields) == 3 and re.fullmatch(r'[1-9][0-9]*(?:_[0-9]+)?', fields[0])
                       and fields[1] == user and fields[2] in common.ACTIVE_STATES
                       and fields[0] not in jobs, 'unambiguous all-user live Slurm metadata required')
        jobs[fields[0]] = fields[2]
    common.require(len(jobs) <= 30, 'all-user live job ceiling exceeded; no new submissions')
    return jobs


def job_state(job, user, visible):
    if str(job) in visible:
        return visible[str(job)]
    result = subprocess.run(['sacct', '-n', '-P', '-X', '-j', str(job), '--format=JobIDRaw,User,State%40,ExitCode'],
                            check=True, capture_output=True, text=True)
    rows = [line.strip().split('|') for line in result.stdout.splitlines() if line.strip()]
    rows = [row for row in rows if row[0] == str(job)]
    if not rows:
        return 'ACCOUNTING_PENDING'
    common.require(len(rows) == 1 and len(rows[0]) >= 4 and rows[0][1] == user,
                   'unambiguous owned parent accounting required')
    state, exit_code = rows[0][2].strip(), rows[0][3].strip()
    common.require(state in common.ACTIVE_STATES or state == 'COMPLETED' and exit_code == '0:0',
                   'owned parent failed; manual review required')
    return state


def read_ledger(path):
    records = [common.parse(line) for line in path.read_text().splitlines()] if path.exists() else []
    actions, collections, previous = {}, {}, None
    job_ids = set()
    for record in records:
        common.require(record.get('schema_version') == 'protocol_replication_ledger_v1'
                       and record.get('previous_sha256') == previous, 'central ledger chain corruption')
        previous = common.object_sha256(record)
        status, action = record.get('status'), record.get('action')
        if status == 'collected':
            common.require(action not in collections, 'duplicate collected artifact set')
            collections[action] = record
            continue
        if status == 'intent':
            common.require(action not in actions, 'duplicate submission intent')
            actions[action] = dict(intent=record, submissions=[], complete=False)
        else:
            common.require(action in actions, 'orphan central ledger event')
            current = actions[action]
            if status == 'submitted':
                position = len(current['submissions'])
                common.require(not current['complete'] and position < len(current['intent']['plans'])
                               and record.get('plan') == current['intent']['plans'][position]
                               and type(record.get('job_id')) is int and record['job_id'] > 0
                               and record['job_id'] not in job_ids, 'ambiguous ledger submission')
                current['submissions'].append(record)
                job_ids.add(record['job_id'])
            elif status == 'complete':
                common.require(not current['complete'] and len(current['submissions']) == len(current['intent']['plans']),
                               'incomplete submission set')
                current['complete'] = True
            else:
                raise ValueError('failed or ambiguous central intent; manual reconciliation required')
    common.require(all(value['complete'] for value in actions.values()),
                   'unfinished central intent; no automatic resubmission')
    return records, actions, collections


def append(path, records, value):
    record = dict(value, schema_version='protocol_replication_ledger_v1', utc=datetime.now(timezone.utc).isoformat(),
                  previous_sha256=common.object_sha256(records[-1]) if records else None)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a') as stream:
        common.append(stream, record)
    records.append(record)


def bound_metadata(binding, expected_path):
    common.require(common.binding_path(binding) == expected_path, 'noncanonical nested receipt path')
    return document(binding) if expected_path.suffix == '.json' else metadata(binding)


def gate_artifacts(unit, request, gate_binding):
    root = common.absolute(request['output_root'])
    gate = bound_metadata(gate_binding, root / 'gate/gate.json')
    index_name = 'fold' if unit['protocol'] == 'A' else 'group'
    common.fixed_fields(gate, dict(schema_version=f"protocol_{unit['protocol'].lower()}_training_gate_v1",
                        protocol=unit['protocol'], source_sha256=unit['source_sha256'], request=unit['request'],
                        **{index_name: unit['index']}), 'actual smoke gate identity mismatch')
    common.require(all(gate.get(name) is True for name in ('passed', 'no_test_open', 'both_arm_smoke', 'replay', 'batch_invariance')),
                   'actual successful nested both-arm gate required')
    for name, target in (('records', 'records_sha256'), ('split', 'split_sha256'),
                         ('enrollment', 'enrollment_sha256'), ('weights', 'weights_sha256')):
        common.require(gate.get(target) == request[name]['sha256'], 'gate input receipt mismatch')
    registration = gate['registration']
    bound_metadata(registration, root / 'registration.json')
    common.require(gate.get('registration_sha256') == registration['sha256'], 'gate registration mismatch')
    common.require(set(gate['arm_reports']) == {'off', 'on'}, 'both nested smoke arms required')
    artifacts, smokes = {'gate': gate_binding, 'registration': registration}, {}
    for arm in ('off', 'on'):
        reports = gate['arm_reports'][arm]
        directory = root / 'gate' / arm
        smoke = bound_metadata(reports['smoke'], directory / 'smoke.json')
        replay = bound_metadata(reports['replay'], directory / 'fresh-replay.json')
        common.require(smoke.get('passed') is True and smoke.get('no_test_open') is True
                       and smoke.get('request') == unit['request'] and type(smoke.get('pid')) is int and smoke['pid'] > 0,
                       'actual smoke parent evidence required')
        common.fixed_fields(smoke['stats'], dict(n_images=32, optimizer_steps=4), 'bounded four-update smoke required')
        config = smoke['config']
        common.fixed_fields(config, dict(protocol=unit['protocol'], arm=arm, reference_enabled=arm == 'on',
                            epochs=100, batch_size=8, eval_batch_size=1, seed=42,
                            **{index_name: unit['index']}), 'smoke fixed recipe/unit mismatch')
        common.fixed_fields(config['precision'], dict(dtype='float32', amp=False, cudnn_tf32=False, matmul_tf32=False),
                            'unchanged FP32 smoke required')
        common.require(smoke['summary'].get('buffers_unchanged') is True
                       and smoke['batch_invariance'].get('passed') is True
                       and smoke['batch_invariance'].get('buffers_unchanged') is True
                       and smoke['batch_invariance'].get('n_images') == 4, 'stored-buffer invariance evidence required')
        bound_metadata(smoke['predictions'], directory / 'val_predictions.csv')
        common.require(common.binding_path(smoke['checkpoint']) == directory / 'selected.pt', 'selected smoke checkpoint receipt required')
        common.require(replay.get('passed') is True and replay.get('fresh_process') is True
                       and replay.get('no_test_open') is True and type(replay.get('pid')) is int
                       and replay['pid'] > 0 and replay['pid'] != smoke['pid']
                       and replay.get('parent_result') == reports['smoke'] and replay.get('request') == unit['request']
                       and replay.get('checkpoint') == smoke['checkpoint'] and replay.get('identities') == smoke['identities']
                       and replay.get('n_images') == 32 and replay.get('summary') == smoke['summary']
                       and replay.get('comparison', {}).get('passed') is True, 'fresh smoke replay chain required')
        common.require(common.checksum(smoke.get('fresh_initial_state_sha256')), 'fresh initialization fingerprint required')
        artifacts.update({f'{arm}_smoke': reports['smoke'], f'{arm}_replay': reports['replay'], f'{arm}_predictions': smoke['predictions']})
        smokes[arm] = smoke
    common.require(smokes['off']['fresh_initial_state_sha256'] == smokes['on']['fresh_initial_state_sha256']
                   and {key: value for key, value in smokes['off']['initialization'].items() if key != 'reference_enabled'} ==
                       {key: value for key, value in smokes['on']['initialization'].items() if key != 'reference_enabled'}
                   and gate['recipe_sha256'] == common.object_sha256(smokes['off']['config']), 'matched smoke initialization/recipe required')
    accounting = gate['accounting']
    common.require(all(type(accounting.get(name)) is int and accounting[name] > 0 for name in ('n_train', 'n_val', 'batches_per_epoch', 'images_per_epoch', 'total_optimizer_steps'))
                   and accounting['n_train'] // 8 == accounting['batches_per_epoch']
                   and accounting['images_per_epoch'] == accounting['batches_per_epoch'] * 8
                   and accounting['total_optimizer_steps'] == accounting['batches_per_epoch'] * 100,
                   'fold/group-derived full training accounting required')
    return gate, artifacts


def completed_artifacts(unit, request, gate_binding, gate):
    root = common.absolute(request['output_root'])
    artifacts, parents = {}, {}
    index_name = 'fold' if unit['protocol'] == 'A' else 'group'
    for arm in ('off', 'on'):
        directory = root / 'training' / arm
        parent, parent_binding = capture(directory / 'train.json')
        common.fixed_fields(parent, dict(schema_version=f"protocol_{unit['protocol'].lower()}_train_v1", protocol=unit['protocol'],
                            phase='train', arm=arm, role='val', passed=True, diagnostic_only=False, no_test_open=True,
                            epochs_completed=100, request=unit['request'], diagnostic_gate=gate_binding,
                            optimizer_steps=gate['accounting']['total_optimizer_steps'], n_val=gate['accounting']['n_val'],
                            **{index_name: unit['index']}), 'completed full train receipt required')
        common.require(parent['identities'].get('source_sha256') == unit['source_sha256']
                       and parent['identities'].get('phase_authority_sha256') == gate_binding['sha256'], 'training source/gate mismatch')
        common.require(common.binding_path(parent['checkpoint']) == directory / 'best.pt', 'selected checkpoint receipt required')
        for name, filename in (('history', 'log.csv'), ('summary', 'val_summary.json'), ('predictions', 'val_predictions.csv'),
                               ('in_process_replay', 'replay.json')):
            bound_metadata(parent[name], directory / filename)
            artifacts[f'{arm}_{name}'] = parent[name]
        fresh, fresh_binding = capture(directory / 'fresh-replay.json')
        common.require(type(parent.get('pid')) is int and parent['pid'] > 0
                       and type(fresh.get('pid')) is int and fresh['pid'] > 0 and fresh['pid'] != parent['pid']
                       and fresh.get('passed') is True and fresh.get('fresh_process') is True and fresh.get('no_test_open') is True
                       and fresh.get('parent_result') == parent_binding and fresh.get('request') == unit['request']
                       and fresh.get('checkpoint') == parent['checkpoint'] and fresh.get('identities') == parent['identities']
                       and fresh.get('n_images') == gate['accounting']['n_val']
                       and fresh.get('comparison', {}).get('passed') is True
                       and fresh.get('summary', {}).get('buffers_unchanged') is True, 'full fresh replay required before evaluation')
        artifacts.update({f'{arm}_train': parent_binding, f'{arm}_fresh_replay': fresh_binding})
        parents[arm] = parent
    return artifacts, parents


def command(unit, phase, arm, gate_sha256):
    nice = 0 if phase in ('smoke', 'evaluate') else 100 if unit['protocol'] == 'A' else 1000
    return ['sbatch', '--parsable', f'--nice={nice}', unit['wrapper']['path'], unit['source_dir'],
            unit['request']['path'], unit['request']['sha256'], phase, arm, gate_sha256]


def once(binding):
    authority = validate_authority(binding)
    ledger = common.absolute(authority['ledger_path'])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(ledger) + '.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        authority = validate_authority(binding)
        records, actions, collections = read_ledger(ledger)
        visible = user_jobs(authority['user'])
        jobs = {job for unit in authority['units'] for job in unit['jobs'].values() if job is not None}
        jobs.update(event['job_id'] for action in actions.values() for event in action['submissions'])
        states = {job: job_state(job, authority['user'], visible) for job in sorted(jobs)}
        live = len(visible) + sum(str(job) not in visible and state != 'COMPLETED' for job, state in states.items())
        candidates, statuses = [], {}
        for unit in authority['units']:
            key = unit_key(unit)
            request = document(unit['request'])
            root = common.absolute(request['output_root'])
            slots = dict(unit['jobs'])
            for phase in ('smoke', 'train', 'evaluate'):
                previous = actions.get(f'{key}:{phase}')
                if previous is not None:
                    common.require(previous['intent']['unit'] == identity(unit), 'replication unit changed after submission')
                    for event in previous['submissions']:
                        slot = event['plan']['slot']
                        common.require(slots[slot] in (None, event['job_id']), 'existing job disagrees with central ledger')
                        slots[slot] = event['job_id']

            def collect(phase, artifacts):
                action = f'{key}:{phase}'
                if action in collections:
                    common.require(collections[action]['unit'] == identity(unit) and collections[action]['artifacts'] == artifacts,
                                   'collected metadata changed; manual review required')
                else:
                    record = dict(status='collected', action=action, unit=identity(unit), artifacts=artifacts)
                    append(ledger, records, record)
                    collections[action] = records[-1]

            if slots['smoke'] is None:
                common.require(not (root / 'gate').exists(), 'unowned smoke outputs require manual adoption')
                statuses[key] = 'smoke_ready' if unit['protocol'] == 'A' else 'manual_b_smoke_required'
                if unit['protocol'] == 'A':
                    candidates.append((0, key, unit, 'smoke', '-', None))
                continue
            if states[slots['smoke']] != 'COMPLETED':
                statuses[key] = 'smoke_pending'
                continue
            unused_gate, gate_binding = capture(root / 'gate/gate.json')
            gate, artifacts = gate_artifacts(unit, request, gate_binding)
            collect('smoke', artifacts)
            if unit['review'] is None:
                statuses[key] = 'independent_review_pending'
                continue
            review = document(unit['review'])
            common.require(review['gate'] == gate_binding, 'independent review does not pin the actual gate')
            if slots['off'] is None:
                common.require(not (root / 'training').exists(), 'unowned train outputs require manual adoption')
                statuses[key] = 'train_ready'
                candidates.append((2 if unit['protocol'] == 'A' else 3, key, unit, 'train', gate_binding['sha256'], gate_binding))
                continue
            if any(states[slots[arm]] != 'COMPLETED' for arm in ('off', 'on')):
                statuses[key] = 'train_pending'
                continue
            artifacts, parents = completed_artifacts(unit, request, gate_binding, gate)
            collect('train', artifacts)
            if slots['evaluate'] is None:
                common.require(not (root / 'evaluation').exists(), 'unowned evaluation outputs require manual adoption')
                statuses[key] = 'evaluation_ready'
                candidates.append((1, key, unit, 'evaluate', gate_binding['sha256'], gate_binding))
                continue
            if states[slots['evaluate']] != 'COMPLETED':
                statuses[key] = 'evaluation_pending'
                continue
            result, result_binding = capture(root / 'evaluation/result.json')
            index_name = 'fold' if unit['protocol'] == 'A' else 'group'
            common.fixed_fields(result, dict(schema_version=f"protocol_{unit['protocol'].lower()}_evaluation_result_v1",
                                protocol=unit['protocol'], passed=True, same_query_population=True,
                                request=unit['request'], **{index_name: unit['index']}), 'completed evaluation result required')
            artifacts = dict(result=result_binding, authorization=result['authorization'])
            authorization = document(result['authorization'])
            common.require(authorization.get('checkpoints') == {arm: parents[arm]['checkpoint'] for arm in ('off', 'on')},
                           'evaluation selected parent checkpoint mismatch')
            common.require(set(result['summaries']) == {'off', 'on'}, 'both evaluated arms required')
            for arm in ('off', 'on'):
                summary = bound_metadata(result['summaries'][arm], root / 'evaluation' / arm / 'test_summary.json')
                common.require(summary.get('source_checkpoint') == parents[arm]['checkpoint'], 'test summary checkpoint mismatch')
                bound_metadata(summary['predictions'], root / 'evaluation' / arm / 'test_predictions.csv')
                artifacts[f'{arm}_summary'] = result['summaries'][arm]
                artifacts[f'{arm}_predictions'] = summary['predictions']
            collect('evaluate', artifacts)
            statuses[key] = 'complete'
        for unused_priority, key, unit, phase, gate_sha256, gate_binding in sorted(candidates, key=lambda item: (item[0], item[1])):
            needed = 1 if phase == 'smoke' else 3
            if live + needed > 28:
                continue
            validate_authority(binding)
            if gate_binding is not None:
                gate_artifacts(unit, document(unit['request']), gate_binding)
                common.require(document(unit['review'])['gate'] == gate_binding, 'review changed before admission')
            current = user_jobs(authority['user'])
            current_live = len(current) + sum(str(job) not in current and state != 'COMPLETED' for job, state in states.items())
            common.require(current_live + needed <= 28, 'live all-user capacity changed before admission')
            arms = ('off', 'on') if phase == 'train' else ('off',)
            plans = [dict(arm=arm, slot=arm if phase == 'train' else phase,
                          command=command(unit, phase, arm, gate_sha256), dependency=None) for arm in arms]
            if phase == 'train':
                plans.append(dict(arm='off', slot='evaluate', command=command(unit, 'evaluate', 'off', gate_sha256),
                                  dependency='submitted_train_pair'))
            action = f'{key}:{phase}'
            append(ledger, records, dict(status='intent', action=action, unit=identity(unit), authority=binding,
                   review=unit['review'], gate=gate_binding, plans=plans, live_user_jobs=len(current), admission_budget=needed))
            try:
                submitted_jobs = []
                for plan in plans:
                    executed = list(plan['command'])
                    if plan['dependency'] is not None:
                        common.require(len(submitted_jobs) == 2, 'two recorded training parents required')
                        executed.insert(3, '--dependency=afterok:' + ':'.join(map(str, submitted_jobs)))
                    response = subprocess.run(executed, check=True, capture_output=True, text=True,
                                              cwd=authority['submit_cwd'])
                    common.require(re.fullmatch(r'[1-9][0-9]*(?:;[^\s;]+)?', response.stdout.strip()), 'ambiguous sbatch response')
                    job_id = int(response.stdout.strip().split(';')[0])
                    common.require(job_id not in jobs, 'duplicate returned job ID')
                    jobs.add(job_id)
                    submitted_jobs.append(job_id)
                    append(ledger, records, dict(status='submitted', action=action, plan=plan, executed_command=executed, job_id=job_id))
                append(ledger, records, dict(status='complete', action=action))
            except BaseException:
                append(ledger, records, dict(status='failed_or_unknown', action=action))
                raise
            return dict(status='submitted', unit=key, phase=phase, job_ids=[record['job_id'] for record in records
                        if record.get('action') == action and record.get('status') == 'submitted'], live_user_jobs=len(current))
        return dict(status='complete' if all(value == 'complete' for value in statuses.values()) else 'waiting',
                    units=statuses, live_user_jobs=len(visible), counted_live_and_unseen=live, ceiling=30, replication_ceiling=28)


def watch(binding, *, poll_seconds=300, timeout_seconds=86400, sleeper=time.sleep, clock=time.monotonic):
    common.require(type(poll_seconds) is int and poll_seconds > 0 and type(timeout_seconds) is int
                   and 0 < timeout_seconds <= 86400, 'positive metadata poll and maximum 24-hour deadline required')
    deadline = clock() + timeout_seconds
    while clock() < deadline:
        result = once(binding)
        print(json.dumps(result, sort_keys=True), file=sys.stderr, flush=True)
        if result['status'] == 'complete':
            return result
        if result['status'] != 'submitted':
            sleeper(min(poll_seconds, max(0, deadline - clock())))
    raise TimeoutError('replication metadata watch deadline reached; pending units were not authorized or resubmitted')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--authority-sha256', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--once', action='store_true')
    mode.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=300)
    parser.add_argument('--timeout-seconds', type=int, default=86400)
    args = parser.parse_args(argv)
    binding = dict(path=args.authority, sha256=args.authority_sha256)
    try:
        result = watch(binding, poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_seconds) if args.watch else once(binding)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(json.dumps(dict(status='timeout' if isinstance(error, TimeoutError) else 'blocked', error=str(error))), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
