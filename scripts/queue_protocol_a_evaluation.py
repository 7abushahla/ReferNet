"""One-shot metadata-only transition from completed ON38152 to Protocol A scoring.

The externally pinned protocol_a_evaluation_queue_v1 authority has exact fields
schema_version, parent_job_id (38152), evaluation_source_dir, score_template,
wrapper, inputs_dir, inputs_manifest, run_dir, request_path, ledger_path, and
submit_cwd. score_template is a
path/SHA256-bound protocol_a_current_score_v1 document WITHOUT arms; it contains
the frozen evaluation inventory and coordinator-attested original critical
hashes. wrapper is a path/SHA256 binding to the reviewed allocated-GPU launcher.
inputs_manifest binds the reviewed seven-file plain SHA256 inventory.
The wrapper receives SOURCE_DIR REQUEST_PATH REQUEST_SHA INPUTS_DIR
INPUTS_MANIFEST INPUTS_SHA, in order. The scoring request pins the source hash.
Input restoration is the wrapper's responsibility, never a login-node action.

--once checks sacct exactly once. Active parents return status=waiting. Failed
parents, missing completion metadata, and ambiguous submissions fail closed.
--watch polls every 300 seconds for at most 20 hours, stopping on submission or
any failure. Waiting and timeout status are printed as JSON metadata. Timeout
exits nonzero without reserving or retrying a submission.
An exclusive JSONL ledger records intent before sbatch and submission afterward.
Any existing incomplete/failed ledger blocks resubmission. Successful repeated
calls return the recorded job without submitting again. No checkpoint, weights,
CSV, support product, model, or query is opened by this controller.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from anchor_job_manifest import source_entries, object_sha256

TRAINING_SOURCE_SHA256 = 'a5ff28ae2a332b977551241f2d07641d51df78d54a4cc1b58169c7338ce11292'
ACTIVE_STATES = {'PENDING', 'RUNNING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED', 'REQUEUED', 'RESIZING'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute(value):
    require(isinstance(value, str), 'absolute path required')
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts
            and not any(parent.is_symlink() for parent in (path, *path.parents)), 'absolute non-symlink path required')
    return path


def checksum(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def binding_path(binding):
    require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'} and checksum(binding['sha256']),
            'path/SHA256 binding required')
    return absolute(binding['path'])


def pairs(entries):
    value = {}
    for key, item in entries:
        require(key not in value, 'duplicate JSON key')
        value[key] = item
    return value


def parse(payload):
    value = json.loads(payload, object_pairs_hook=pairs)
    json.dumps(value, allow_nan=False)
    require(isinstance(value, dict), 'JSON object required')
    return value


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def document(binding):
    payload = binding_path(binding).read_bytes()
    require(digest(payload) == binding['sha256'], 'authority SHA256 mismatch')
    return parse(payload)


def capture(path):
    payload = absolute(str(path)).read_bytes()
    return parse(payload), dict(path=str(path), sha256=digest(payload))


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def validate_authority(binding):
    authority = document(binding)
    fields = {'schema_version', 'parent_job_id', 'evaluation_source_dir', 'score_template', 'wrapper',
              'inputs_dir', 'inputs_manifest', 'run_dir', 'request_path', 'ledger_path', 'submit_cwd'}
    require(set(authority) == fields and authority['schema_version'] == 'protocol_a_evaluation_queue_v1'
            and type(authority['parent_job_id']) is int and authority['parent_job_id'] == 38152,
            'strict reviewed ON38152 queue authority required')
    template = document(authority['score_template'])
    fixed = dict(schema_version='protocol_a_current_score_v1', protocol='A', fold=0, seed=42,
                 eval_batch_size=1, training_source_sha256=TRAINING_SOURCE_SHA256)
    extra = {'evaluation_source_sha256', 'critical_source_hashes', 'records', 'split', 'enrollment',
             'root', 'raw_root', 'weights', 'config', 'output_root'}
    require(set(template) == set(fixed) | extra
            and all(type(template[key]) is type(value) and template[key] == value for key, value in fixed.items()),
            'reviewed scoring template without arms required')
    source = absolute(authority['evaluation_source_dir'])
    entries = source_entries(source)
    require(checksum(template['evaluation_source_sha256'])
            and object_sha256(entries) == template['evaluation_source_sha256'], 'frozen evaluation source mismatch')
    critical = template['critical_source_hashes']
    inventory = {entry['path']: entry['sha256'] for entry in entries}
    require(isinstance(critical, dict) and critical, 'original critical source hashes required')
    require(all(isinstance(path, str) and path.startswith('src/') and path.endswith('.py')
                and checksum(value) and inventory.get(path) == value for path, value in critical.items()),
            'critical original source mismatch')
    for name in ('records', 'split', 'enrollment', 'root', 'weights', 'config'):
        binding_path(template[name])
    wrapper = binding_path(authority['wrapper'])
    require(digest(wrapper.read_bytes()) == authority['wrapper']['sha256'], 'reviewed GPU wrapper SHA256 mismatch')
    inputs = absolute(authority['inputs_dir'])
    require(inputs.is_dir(), 'existing durable input directory required')
    manifest = binding_path(authority['inputs_manifest'])
    inventory_payload = manifest.read_bytes()
    require(digest(inventory_payload) == authority['inputs_manifest']['sha256'], 'durable inputs manifest SHA256 mismatch')
    inventory_rows = [line for line in inventory_payload.decode().splitlines() if line.strip()]
    require(len(inventory_rows) == 7, 'reviewed seven-file inputs manifest required')
    names = []
    for row in inventory_rows:
        match = re.fullmatch(r'([0-9a-f]{64}) [ *](.+)', row)
        require(match is not None, 'plain SHA256 inputs manifest required')
        name = match.group(2)
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts and '\\' not in name,
                'safe relative input inventory paths required')
        names.append(name)
    require(len(set(names)) == 7, 'unique input inventory paths required')
    run = absolute(authority['run_dir'])
    protected = [Path(__file__).resolve(), ROOT / 'scripts/anchor_job_manifest.py', source, run, wrapper,
                 absolute(binding['path']), binding_path(authority['score_template']),
                 absolute(template['raw_root']), inputs, manifest]
    protected += [binding_path(template[name]) for name in ('records', 'split', 'enrollment', 'root', 'weights', 'config')]
    outputs = [absolute(authority[name]) for name in ('request_path', 'ledger_path')] + [absolute(template['output_root'])]
    require(all(not output.is_relative_to(path) and not path.is_relative_to(output) for output in outputs for path in protected),
            'queue output overlaps immutable input')
    require(all(not first.is_relative_to(second) and not second.is_relative_to(first)
                for index, first in enumerate(outputs) for second in outputs[index + 1:]), 'queue output paths overlap')
    require(absolute(authority['submit_cwd']).is_dir(), 'existing submission directory required')
    return authority, template


def parent_state(job_id):
    response = subprocess.run(['sacct', '-n', '-P', '-X', '-j', str(job_id), '--format=JobIDRaw,State%40,ExitCode'],
                              check=True, capture_output=True, text=True)
    rows = [line.strip().split('|') for line in response.stdout.splitlines() if line.strip()]
    parents = [row for row in rows if row[0] == str(job_id)]
    require(len(parents) == 1 and len(parents[0]) >= 3, 'unambiguous parent Slurm accounting required')
    state, exit_code = parents[0][1].strip(), parents[0][2].strip()
    if state in ACTIVE_STATES:
        return state
    require(state == 'COMPLETED' and exit_code == '0:0', 'parent job did not complete successfully')
    return state


def fixed_fields(document_value, expected, message):
    require(all(type(document_value.get(key)) is type(value) and document_value[key] == value
                for key, value in expected.items()), message)


def completed_arms(authority, template):
    run = absolute(authority['run_dir'])
    arms, identities, initial_states, captured = {}, {}, {}, []
    for arm in ('off', 'on'):
        directory = run / 'training' / arm
        train, train_binding = capture(directory / 'train.json')
        completion, completion_binding = capture(directory / ('control.json' if arm == 'off' else 'on.json'))
        replay, replay_binding = capture(directory / 'fresh-replay.json')
        fixed_fields(train, dict(passed=True, phase='train', diagnostic_only=False, epochs_completed=100,
                                optimizer_steps=22200, role='val', arm=arm), 'full arm training metadata required')
        require(train.get('scheduler') == dict(total_updates=22200, warmup_updates=1000), 'full scheduler metadata required')
        checkpoint = train.get('checkpoint')
        require(binding_path(checkpoint) == directory / 'best.pt', 'canonical selected checkpoint binding required')
        require(binding_path(train.get('predictions')) == directory / 'val_predictions.csv', 'canonical validation CSV binding required')
        require(binding_path(train.get('registration')) == run / 'training/shared/registration.json', 'canonical training registration required')
        require(type(train.get('selected_epoch')) is int and 1 <= train['selected_epoch'] <= 100
                and checksum(train.get('fresh_initial_state_sha256')), 'selected epoch/initial state metadata required')
        identity = train.get('identities')
        require(isinstance(identity, dict) and all(checksum(value) for value in identity.values()), 'training identities required')
        expected_ids = dict(source_sha256=TRAINING_SOURCE_SHA256,
                            records_sha256=template['records']['sha256'], split_sha256=template['split']['sha256'],
                            enrollment_sha256=template['enrollment']['sha256'], weights_sha256=template['weights']['sha256'],
                            registration_sha256=train['registration']['sha256'])
        require(all(identity.get(key) == value for key, value in expected_ids.items()), 'original training identity mismatch')
        fixed_fields(completion, dict(schema_version='direct_reference_control_result_v1' if arm == 'off' else 'direct_reference_on_study_result_v1',
                                     passed=True, arm=arm, epochs_completed=100, train_result=train_binding,
                                     checkpoint=checkpoint, identities=identity, fresh_replay=replay_binding),
                     'complete root receipt binding required')
        require(arm != 'off' or completion.get('phase') == 'train', 'OFF completion phase mismatch')
        require(arm != 'on' or completion.get('scientific_source_sha256') == TRAINING_SOURCE_SHA256,
                'ON scientific source mismatch')
        fixed_fields(replay, dict(schema_version='direct_reference_control_replay_v1' if arm == 'off' else 'direct_reference_on_study_replay_v1',
                                 passed=True, fresh_process=True, phase='train', arm=arm, n_images=222, buffers_unchanged=True,
                                 train_result=train_binding, checkpoint=checkpoint, identities=identity), 'full fresh replay metadata required')
        require(type(replay.get('pid')) is int and replay['pid'] > 0, 'fresh replay process metadata required')
        require(binding_path(replay.get('history')) == directory / 'log.csv', 'canonical history binding required')
        image_ids = replay.get('image_ids')
        require(isinstance(image_ids, list) and len(image_ids) == 222
                and all(isinstance(value, str) and value for value in image_ids)
                and len(set(image_ids)) == 222, 'full validation membership metadata required')
        fixed_fields(replay.get('summary', {}), dict(n_images=222, buffers_unchanged=True), 'full replay summary required')
        fixed_fields(replay.get('replay', {}), dict(passed=True, n_images=222), 'successful replay comparison required')
        fixed_fields(train.get('replay', {}), dict(passed=True, fresh_process=False, phase='train', n_images=222),
                     'successful in-process replay required')
        authority_key = 'control_request' if arm == 'off' else 'on_request'
        binding_path(completion.get(authority_key))
        require(completion[authority_key] == replay.get(authority_key), 'completion/replay authority mismatch')
        if arm == 'on':
            require(checksum(completion.get('launcher_sha256'))
                    and completion['launcher_sha256'] == replay.get('launcher_sha256'), 'ON launcher metadata mismatch')
        arms[arm] = dict(train_result=train_binding, completion=completion_binding, fresh_replay=replay_binding)
        identities[arm], initial_states[arm] = identity, train['fresh_initial_state_sha256']
        captured.extend((train_binding, completion_binding, replay_binding))
    require(initial_states['off'] == initial_states['on'], 'matched initial state metadata required')
    for key in ('source_sha256', 'records_sha256', 'split_sha256', 'enrollment_sha256', 'weights_sha256',
                'registration_sha256', 'phase_authority_sha256', 'preprocess_sha256'):
        require(checksum(identities['off'].get(key)) and identities['off'].get(key) == identities['on'].get(key),
                'matched original training identities required')
    for binding in captured:
        document(binding)
    return arms


def append(stream, value):
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def submission_command(authority, request):
    return ['sbatch', '--parsable', authority['wrapper']['path'], authority['evaluation_source_dir'],
            request['path'], request['sha256'], authority['inputs_dir'],
            authority['inputs_manifest']['path'], authority['inputs_manifest']['sha256']]


def existing_submission(authority, binding, template):
    path = absolute(authority['ledger_path'])
    if not path.exists():
        return None
    records = [parse(line) for line in path.read_text().splitlines() if line.strip()]
    require(len(records) == 2 and records[0].get('status') == 'intent' and records[1].get('status') == 'submitted',
            'incomplete or failed submission ledger, manual review required; no resubmission')
    require(all(record.get('schema_version') == 'protocol_a_evaluation_queue_ledger_v1'
                and record.get('queue_authority') == binding for record in records), 'existing ledger authority mismatch')
    require(records[0].get('request') == records[1].get('request')
            and records[1]['request']['path'] == authority['request_path'], 'existing request ledger mismatch')
    request = document(records[1]['request'])
    require(set(request) == set(template) | {'arms'}
            and all(request.get(key) == value for key, value in template.items())
            and isinstance(request['arms'], dict) and set(request['arms']) == {'off', 'on'},
            'existing request differs from reviewed template')
    require(all(record.get('wrapper') == authority['wrapper'] and record.get('inputs_manifest') == authority['inputs_manifest']
                and record.get('parent_job_id') == authority['parent_job_id']
                and record.get('command') == submission_command(authority, records[1]['request']) for record in records),
            'existing ledger reviewed submission mismatch')
    require(type(records[1].get('job_id')) is int and records[1]['job_id'] > 0, 'existing submitted job identity required')
    return records[1]


def once(binding):
    authority, template = validate_authority(binding)
    previous = existing_submission(authority, binding, template)
    if previous is not None:
        return previous
    state = parent_state(authority['parent_job_id'])
    if state != 'COMPLETED':
        return dict(status='waiting', parent_job_id=authority['parent_job_id'], parent_state=state)
    arms = completed_arms(authority, template)
    validate_authority(binding)
    payload = encode(dict(template, arms=arms))
    request = dict(path=authority['request_path'], sha256=digest(payload))
    require(not absolute(template['output_root']).exists(), 'new scoring output required')
    require(not absolute(request['path']).exists(), 'new exclusive scoring request required')
    command = submission_command(authority, request)
    common = dict(schema_version='protocol_a_evaluation_queue_ledger_v1', queue_authority=binding,
                  parent_job_id=authority['parent_job_id'], request=request, wrapper=authority['wrapper'],
                  inputs_manifest=authority['inputs_manifest'], command=command)
    ledger = absolute(authority['ledger_path'])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open('x') as stream:
        append(stream, dict(common, status='intent', utc=datetime.now(timezone.utc).isoformat()))
        try:
            path = absolute(request['path'])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as request_stream:
                request_stream.write(payload)
                request_stream.flush()
                os.fsync(request_stream.fileno())
            response = subprocess.run(command, cwd=authority['submit_cwd'], check=True, capture_output=True, text=True)
            match = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?\s*', response.stdout)
            require(match is not None, 'ambiguous sbatch response, manual review required; no resubmission')
            result = dict(common, status='submitted', job_id=int(match.group(1)), utc=datetime.now(timezone.utc).isoformat())
            append(stream, result)
            return result
        except Exception as error:
            append(stream, dict(common, status='failed_or_unknown', error_type=type(error).__name__,
                                utc=datetime.now(timezone.utc).isoformat()))
            raise


class WatchTimeout(TimeoutError):
    def __init__(self, report):
        self.report = report
        super().__init__('Protocol A metadata watch deadline reached without submission')


def watch(binding, *, poll_seconds=300, timeout_seconds=72000, sleeper=None, clock=None):
    require(type(poll_seconds) is int and poll_seconds > 0, 'positive integer polling interval required')
    require(type(timeout_seconds) is int and 0 < timeout_seconds <= 72000, 'positive deadline of at most 20 hours required')
    sleeper = time.sleep if sleeper is None else sleeper
    clock = time.monotonic if clock is None else clock
    deadline = clock() + timeout_seconds
    polls = 0
    while True:
        if clock() >= deadline:
            raise WatchTimeout(dict(status='timeout', parent_job_id=38152, polls=polls,
                                    timeout_seconds=timeout_seconds, submission_attempted=False))
        result = once(binding)
        polls += 1
        if result.get('status') == 'submitted':
            return result
        require(result.get('status') == 'waiting', 'unexpected queue transition, manual review required')
        print(json.dumps(dict(result, polls=polls), sort_keys=True), file=sys.stderr, flush=True)
        remaining = deadline - clock()
        if remaining > 0:
            sleeper(min(poll_seconds, remaining))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--authority-sha256', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--once', action='store_true')
    mode.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=300)
    parser.add_argument('--timeout-seconds', type=int, default=72000)
    args = parser.parse_args(argv)
    binding = dict(path=args.authority, sha256=args.authority_sha256)
    try:
        result = watch(binding, poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_seconds) if args.watch else once(binding)
    except WatchTimeout as error:
        print(json.dumps(error.report, sort_keys=True), file=sys.stderr, flush=True)
        raise SystemExit(2) from error
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
