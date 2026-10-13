"""Passive fixed-host bootstrap. No network operation occurs on import.

SHA-bound reviewed_replication_resume_v1 manifest fields are archive (path/SHA),
tool_files (three relative script paths mapped to SHA256), metadata_files
([{binding: path/SHA, destination: absolute remote path}]), remote_root,
remote_authority, and local_ledger. The new remote_root is under
/shared/b00090279/refer-or-clear/results/monitoring. Metadata retains original
destinations under that monitoring directory or the project evidence directory.
The archive contains only queue_protocol_replications.py and its two stdlib
dependencies. Metadata may be JSON or Markdown, never scientific payloads.
Existing destinations must match their original SHA; nothing is overwritten.
An exclusive durable local intent precedes deployment. Any ambiguous failure
blocks automatic retries. A remote exclusive launch intent likewise guards the
single nohup queue invocation. A PID is a launch receipt, not a job outcome.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_protocol_a_evaluation as common

HOST = 'b00090279@hpc-gpu.aus.edu'
OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20']
PYTHON = '/shared/b00090279/conda/envs/refer-or-clear/bin/python'
PREFIX = Path('/shared/b00090279/refer-or-clear/results/monitoring')
METADATA_PREFIXES = (PREFIX, Path('/shared/b00090279/refer-or-clear/evidence'))
TOOLS = {'scripts/queue_protocol_replications.py',
         'scripts/queue_protocol_a_evaluation.py', 'scripts/anchor_job_manifest.py'}

INSTALLER = r'''
import hashlib, io, json, os, pathlib, subprocess, sys, tarfile
plan = json.loads(sys.argv[1])
root = pathlib.Path(plan['remote_root'])
staging = pathlib.Path(plan['staging'])
def safe(path):
    if not path.is_absolute() or '..' in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('unsafe destination')
def sha(data):
    return hashlib.sha256(data).hexdigest()
def exclusive(path, data, mode=0o444):
    safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.chmod(path, mode)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
safe(root); safe(staging)
payload = (staging / 'tools.tar').read_bytes()
if sha(payload) != plan['archive']['sha256']: raise ValueError('archive mismatch')
files = {}
with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
    for member in archive.getmembers():
        if not member.isfile() or member.name not in plan['tool_files'] or member.name in files:
            raise ValueError('unsafe tool member')
        data = archive.extractfile(member).read()
        if sha(data) != plan['tool_files'][member.name]: raise ValueError('tool mismatch')
        files[member.name] = data
if set(files) != set(plan['tool_files']): raise ValueError('missing tool')
destinations = {root / name: data for name, data in files.items()}
for index, entry in enumerate(plan['metadata_files']):
    data = (staging / ('input-' + str(index))).read_bytes()
    if sha(data) != entry['binding']['sha256']: raise ValueError('metadata mismatch')
    destination = pathlib.Path(entry['destination'])
    if destination in destinations: raise ValueError('destination collision')
    destinations[destination] = data
for path, data in destinations.items():
    safe(path)
    if path.exists() and (not path.is_file() or path.read_bytes() != data):
        raise ValueError('existing destination mismatch')
ledger = pathlib.Path(str(root) + '.launch.jsonl')
pid_path = pathlib.Path(str(root) + '.pid')
safe(ledger); safe(pid_path)
if ledger.exists() or pid_path.exists(): raise ValueError('remote launch already reserved')
for path, data in destinations.items():
    if not path.exists(): exclusive(path, data)
    else: os.chmod(path, 0o444)
directories = {root}
for path in destinations:
    directories.update(parent for parent in path.parents if parent == root or parent.is_relative_to(root))
for directory in directories: os.chmod(directory, 0o555)
exclusive(ledger, (json.dumps({'status':'intent', 'manifest_sha256':plan['manifest_sha256']})+'\n').encode(), 0o600)
with open(str(root) + '.log', 'xb') as log:
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    environment['PATH'] = '/opt/slurm/bin:' + environment.get('PATH', os.defpath)
    process = subprocess.Popen(['nohup', plan['python'], str(root / 'scripts/queue_protocol_replications.py'),
        '--authority', plan['remote_authority'], '--authority-sha256', plan['authority_sha256'],
        '--watch', '--poll-seconds', '300', '--timeout-seconds', '72000'],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=environment)
exclusive(pid_path, (str(process.pid)+'\n').encode())
result = {'status':'started', 'pid':process.pid, 'manifest_sha256':plan['manifest_sha256'],
          'remote_ledger':str(ledger), 'remote_pid':str(pid_path)}
with open(ledger, 'ab') as stream:
    stream.write((json.dumps(result)+'\n').encode()); stream.flush(); os.fsync(stream.fileno())
print(json.dumps(result))
'''


def validate(binding):
    plan = common.document(binding)
    common.require(set(plan) == {'schema_version', 'archive', 'tool_files', 'metadata_files',
                               'remote_root', 'remote_authority', 'local_ledger'}
                   and plan['schema_version'] == 'reviewed_replication_resume_v1', 'strict resume manifest required')
    root = common.absolute(plan['remote_root'])
    common.require(root.is_relative_to(PREFIX) and root != PREFIX, 'fixed monitoring destination required')
    ledger = common.absolute(plan['local_ledger'])
    archive_path = common.binding_path(plan['archive'])
    common.require(archive_path.stat().st_size <= 2 * 1024 * 1024, 'tool archive too large')
    payload = archive_path.read_bytes()
    common.require(common.digest(payload) == plan['archive']['sha256'], 'archive SHA mismatch')
    common.require(isinstance(plan['tool_files'], dict) and set(plan['tool_files']) == TOOLS
                   and all(common.checksum(value) for value in plan['tool_files'].values()), 'pure tool inventory required')
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive.getmembers():
            common.require(member.isfile() and member.name in TOOLS and member.name not in seen
                           and member.size <= 512 * 1024, 'unsafe tool archive member')
            common.require(common.digest(archive.extractfile(member).read()) == plan['tool_files'][member.name],
                           'tool SHA mismatch')
            seen.add(member.name)
    common.require(seen == TOOLS, 'incomplete tool inventory')
    common.require(isinstance(plan['metadata_files'], list) and plan['metadata_files'], 'metadata required')
    destinations = {str(root / name) for name in TOOLS}
    inputs = {archive_path, common.binding_path(binding)}
    authority = None
    for entry in plan['metadata_files']:
        common.require(isinstance(entry, dict) and set(entry) == {'binding', 'destination'}, 'strict metadata entry required')
        path = common.binding_path(entry['binding'])
        destination = common.absolute(entry['destination'])
        common.require(path.suffix in {'.json', '.md'} and destination.suffix == path.suffix
                       and any(destination.is_relative_to(prefix) for prefix in METADATA_PREFIXES)
                       and str(destination) not in destinations, 'metadata destination required')
        data = path.read_bytes()
        common.require(len(data) <= 2 * 1024 * 1024 and common.digest(data) == entry['binding']['sha256'], 'metadata SHA mismatch')
        inputs.add(path)
        destinations.add(str(destination))
        if entry['destination'] == plan['remote_authority']:
            authority = entry['binding']['sha256']
            common.require(common.parse(data)['schema_version'] == 'protocol_replication_queue_v1', 'queue authority required')
    common.require(authority is not None and ledger not in inputs, 'authority/ledger binding required')
    return dict(plan, manifest_sha256=binding['sha256'], authority_sha256=authority,
                staging=str(root) + '.incoming-' + binding['sha256'], python=PYTHON)


def append(path, value, mode='ab'):
    with open(path, mode) as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode())
        stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def once(binding):
    plan = validate(binding)
    ledger = common.absolute(plan['local_ledger'])
    if ledger.exists():
        entries = [common.parse(line) for line in ledger.read_bytes().splitlines() if line.strip()]
        common.require(entries and all(entry.get('manifest_sha256') == binding['sha256'] for entry in entries)
                       and entries[-1].get('status') == 'started' and type(entries[-1].get('pid')) is int
                       and entries[-1]['pid'] > 0, 'ambiguous or conflicting deployment intent; manual reconciliation required')
        return dict(entries[-1], status='already_started')
    try:
        probe = subprocess.run(['ssh', *OPTIONS, HOST, 'true'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return {'status': 'waiting', 'reason': 'fixed-host connection unavailable'}
    if probe.returncode:
        return {'status': 'waiting', 'reason': 'fixed-host connection unavailable'}
    validate(binding)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    append(ledger, {'status': 'intent', 'manifest_sha256': binding['sha256']}, 'xb')
    try:
        def run(command):
            return subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        run(['ssh', *OPTIONS, HOST, shlex.join([PYTHON, '-c',
            'import pathlib,sys; target=pathlib.Path(sys.argv[1]); '
            'assert not any(path.is_symlink() for path in (target,*target.parents)); target.mkdir()', plan['staging']])])
        run(['scp', *OPTIONS, '--', plan['archive']['path'], HOST + ':' + shlex.quote(plan['staging'] + '/tools.tar')])
        for index, entry in enumerate(plan['metadata_files']):
            run(['scp', *OPTIONS, '--', entry['binding']['path'], HOST + ':' + shlex.quote(plan['staging'] + '/input-' + str(index))])
        response = run(['ssh', *OPTIONS, HOST, shlex.join([PYTHON, '-c', INSTALLER, json.dumps(plan)])])
        result = common.parse(response.stdout)
        common.require(set(result) == {'status', 'pid', 'manifest_sha256', 'remote_ledger', 'remote_pid'}
                       and result['status'] == 'started' and type(result['pid']) is int and result['pid'] > 0
                       and result['manifest_sha256'] == binding['sha256']
                       and result['remote_ledger'] == plan['remote_root'] + '.launch.jsonl'
                       and result['remote_pid'] == plan['remote_root'] + '.pid', 'invalid launch receipt')
        append(ledger, result)
        return result
    except BaseException:
        append(ledger, {'status': 'failed_or_unknown', 'manifest_sha256': binding['sha256']})
        raise


def watch(binding, poll_seconds=300, timeout_seconds=72000, *, sleeper=time.sleep, clock=time.monotonic):
    common.require(poll_seconds == 300 and 0 < timeout_seconds <= 72000, 'poll 300 seconds; deadline at most 20 hours')
    deadline = clock() + timeout_seconds
    while clock() < deadline:
        result = once(binding)
        print(json.dumps(result), flush=True)
        if result['status'] != 'waiting':
            return result
        sleeper(min(poll_seconds, max(0, deadline - clock())))
    print(json.dumps({'status': 'timeout', 'manifest_sha256': binding['sha256']}), flush=True)
    raise TimeoutError('fixed-host restoration deadline expired; nothing deployed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--once', action='store_true')
    mode.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    binding = {'path': args.manifest, 'sha256': args.manifest_sha256}
    if args.watch:
        watch(binding)
    else:
        print(json.dumps(once(binding)), flush=True)


if __name__ == '__main__':
    main()
