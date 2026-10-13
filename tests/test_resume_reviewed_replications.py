import importlib.util
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def launcher():
    path = ROOT / 'scripts/resume_reviewed_replications.py'
    assert path.is_file(), 'passive reviewed bootstrap is missing'
    specification = importlib.util.spec_from_file_location('passive_replication_bootstrap', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_bootstrap_interface():
    module = launcher()
    assert callable(module.once) and callable(module.watch)


def bound(path, payload):
    path.write_bytes(payload)
    return {'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest()}


@pytest.fixture
def prepared(tmp_path):
    module = launcher()
    output = io.BytesIO()
    inventory = {}
    with tarfile.open(fileobj=output, mode='w') as archive:
        for name in sorted(module.TOOLS):
            payload = ('"""synthetic metadata tool ' + name + '"""\n').encode()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
            inventory[name] = hashlib.sha256(payload).hexdigest()
    archive_binding = bound(tmp_path / 'tools.tar', output.getvalue())
    authority = bound(tmp_path / 'authority.json', json.dumps({'schema_version': 'protocol_replication_queue_v1'}).encode())
    root = str(module.PREFIX / 'bootstrap-synthetic')
    plan = {'schema_version': 'reviewed_replication_resume_v1', 'archive': archive_binding,
            'tool_files': inventory, 'metadata_files': [{'binding': authority, 'destination': root + '/authority.json'}],
            'remote_root': root, 'remote_authority': root + '/authority.json',
            'local_ledger': str(tmp_path / 'new' / 'deployment.jsonl')}
    manifest = bound(tmp_path / 'manifest.json', json.dumps(plan).encode())
    return module, plan, manifest


def receipt(plan, manifest):
    return {'status': 'started', 'pid': 12345, 'manifest_sha256': manifest['sha256'],
            'remote_ledger': plan['remote_root'] + '.launch.jsonl', 'remote_pid': plan['remote_root'] + '.pid'}


def test_network_failure_has_no_writes_or_deployment(prepared, monkeypatch):
    module, plan, manifest = prepared
    calls = []
    def unavailable(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=255, stdout='', stderr='DNS unavailable')
    monkeypatch.setattr(module.subprocess, 'run', unavailable)
    assert module.once(manifest)['status'] == 'waiting'
    assert calls == [['ssh', *module.OPTIONS, module.HOST, 'true']]
    assert not Path(plan['local_ledger']).parent.exists()


@pytest.mark.parametrize('artifact', ['archive', 'metadata', 'manifest'])
def test_tamper_rejected_before_probe(prepared, monkeypatch, artifact):
    module, plan, manifest = prepared
    binding = {'archive': plan['archive'], 'metadata': plan['metadata_files'][0]['binding'], 'manifest': manifest}[artifact]
    Path(binding['path']).write_bytes(b'tampered')
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: pytest.fail('network before validation'))
    with pytest.raises(ValueError, match='SHA'):
        module.once(manifest)
    assert not Path(plan['local_ledger']).exists()


def test_success_fixed_host_and_exclusive_intent_then_idempotent(prepared, monkeypatch):
    module, plan, manifest = prepared
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[0] in {'ssh', 'scp'}
        assert command[1:5] == module.OPTIONS
        if command[0] == 'ssh':
            assert command[5] == module.HOST
        else:
            assert command[-1].startswith(module.HOST + ':')
        if command[-1] == 'true':
            assert not Path(plan['local_ledger']).exists()
            return SimpleNamespace(returncode=0, stdout='')
        assert json.loads(Path(plan['local_ledger']).read_text())['status'] == 'intent'
        return SimpleNamespace(returncode=0, stdout=json.dumps(receipt(plan, manifest)))
    monkeypatch.setattr(module.subprocess, 'run', run)
    result = module.once(manifest)
    assert result == receipt(plan, manifest)
    assert len(calls) == 5
    command = shlex.split(calls[-1][-1])
    assert command[:2] == [module.PYTHON, '-c']
    deployment = json.loads(command[-1])
    assert deployment['authority_sha256'] == plan['metadata_files'][0]['binding']['sha256']
    assert deployment['remote_authority'] == plan['remote_authority']
    assert deployment['tool_files'] == plan['tool_files']
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: pytest.fail('duplicate network operation'))
    assert module.once(manifest)['status'] == 'already_started'


@pytest.mark.parametrize('stage', ['scp', 'receipt'])
def test_ambiguous_partial_deployment_never_retries(prepared, monkeypatch, stage):
    module, plan, manifest = prepared
    def run(command, **kwargs):
        if stage == 'scp' and command[0] == 'scp':
            raise subprocess.CalledProcessError(255, command)
        return SimpleNamespace(returncode=0, stdout='not a receipt')
    monkeypatch.setattr(module.subprocess, 'run', run)
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        module.once(manifest)
    assert 'failed_or_unknown' in Path(plan['local_ledger']).read_text()
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: pytest.fail('ambiguous retry'))
    with pytest.raises(ValueError, match='manual reconciliation'):
        module.once(manifest)


def test_watch_bounded_timeout_no_real_wait(prepared, monkeypatch, capsys):
    module, plan, manifest = prepared
    now = [0]
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=255))
    with pytest.raises(TimeoutError):
        module.watch(manifest, timeout_seconds=600, sleeper=sleep, clock=lambda: now[0])
    assert sleeps == [300, 300]
    assert '"status": "timeout"' in capsys.readouterr().out
    assert not Path(plan['local_ledger']).exists()


def test_remote_install_hashes_existing_destinations_and_exact_once(prepared, tmp_path, monkeypatch, capsys):
    module, plan, manifest = prepared
    deployment = module.validate(manifest)
    root = tmp_path / 'remote' / 'tools'
    staging = tmp_path / 'staging'
    staging.mkdir()
    (staging / 'tools.tar').write_bytes(Path(plan['archive']['path']).read_bytes())
    authority = plan['metadata_files'][0]['binding']
    (staging / 'input-0').write_bytes(Path(authority['path']).read_bytes())
    deployment.update(remote_root=str(root), staging=str(staging), remote_authority=str(root / 'authority.json'))
    deployment['metadata_files'] = [{'binding': authority, 'destination': str(root / 'authority.json')}]
    root.mkdir(parents=True)
    destination = root / 'authority.json'
    destination.write_bytes(b'wrong original destination')
    launches = []
    def spawn(command, **kwargs):
        launches.append(command)
        assert Path(str(root) + '.launch.jsonl').exists()
        assert kwargs['env']['PATH'].startswith('/opt/slurm/bin:')
        assert kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'
        assert command == ['nohup', module.PYTHON, str(root / 'scripts/queue_protocol_replications.py'),
                           '--authority', str(destination), '--authority-sha256', authority['sha256'],
                           '--watch', '--poll-seconds', '300', '--timeout-seconds', '72000']
        return SimpleNamespace(pid=4321)
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    monkeypatch.setattr(module.sys, 'argv', ['installer', json.dumps(deployment)])
    with pytest.raises(ValueError, match='existing destination mismatch'):
        exec(module.INSTALLER, {})
    assert not launches and not (root / 'scripts').exists()
    destination.unlink()
    try:
        exec(module.INSTALLER, {})
        result = json.loads(capsys.readouterr().out)
        assert result['pid'] == 4321
        assert destination.read_bytes() == Path(authority['path']).read_bytes()
        assert root.stat().st_mode & 0o777 == 0o555
        with pytest.raises(ValueError, match='already reserved'):
            exec(module.INSTALLER, {})
        assert len(launches) == 1
    finally:
        for path in [root, root / 'scripts']:
            if path.exists():
                path.chmod(0o755)


@pytest.mark.parametrize('kind', ['symlink', 'traversal', 'extra'])
def test_non_tool_archive_rejected_before_network(prepared, monkeypatch, kind):
    module, plan, manifest = prepared
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w') as archive:
        member = tarfile.TarInfo('../escape.py' if kind == 'traversal' else
                                 ('weights.pt' if kind == 'extra' else sorted(module.TOOLS)[0]))
        if kind == 'symlink':
            member.type = tarfile.SYMTYPE
            member.linkname = '/etc/passwd'
        archive.addfile(member)
    plan['archive'] = bound(Path(plan['archive']['path']), output.getvalue())
    manifest = bound(Path(manifest['path']), json.dumps(plan).encode())
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: pytest.fail('unsafe archive probed'))
    with pytest.raises(ValueError, match='unsafe tool archive'):
        module.once(manifest)


def test_original_metadata_paths_preserved(prepared):
    module, plan, manifest = prepared
    destination = str(module.METADATA_PREFIXES[1] / 'review.md')
    local = Path(manifest['path']).parent / 'review.md'
    plan['metadata_files'].append({'binding': bound(local, b'reviewed metadata only'), 'destination': destination})
    plan['remote_authority'] = str(module.PREFIX / 'replication-queue-reviewed-b1-5.json')
    plan['metadata_files'][0]['destination'] = plan['remote_authority']
    manifest = bound(Path(manifest['path']), json.dumps(plan).encode())
    deployment = module.validate(manifest)
    assert deployment['remote_authority'] == plan['remote_authority']
    assert deployment['metadata_files'][1]['destination'] == destination
