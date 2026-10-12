import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / 'scripts/cluster/download_transfer_archives.sh'
ARCHIVES = {
    'fantasyid': ('fantasyid', 'FantasyID.tgz', '17063366', '5a6e608762f640db026ac122b4400a72'),
    'az': ('idnet', 'AZ.zip', '13852757', 'd51cdee3f3aaf4592039ba6623af7df3'),
    'wv': ('idnet', 'WV.zip', '13854938', 'a51cbc873c5cddd1b28b242f8bb96496'),
}


@pytest.fixture
def download_job(tmp_path):
    raw = tmp_path / 'raw with spaces'
    log = tmp_path / 'calls.log'
    environment_file = tmp_path / 'bash_env'
    environment_file.write_text(f'''source() {{
    printf 'source %s\\n' "$1" >> "$TEST_LOG"
    return 99
}}
conda() {{
    printf 'conda %s %s\\n' "$1" "$2" >> "$TEST_LOG"
    return 98
}}
command() {{
    if [[ "$1" == -v && "$2" == "${{MOCK_MISSING_TOOL:-}}" ]]; then return 127; fi
    builtin command "$@"
}}
curl() {{
    printf 'curl' >> "$TEST_LOG"
    printf ' <%s>' "$@" >> "$TEST_LOG"
    printf '\\n' >> "$TEST_LOG"
    [[ "$PWD" == "$ROC_TRANSFER_RAW_ROOT/fantasyid" || "$PWD" == "$ROC_TRANSFER_RAW_ROOT/idnet" ]] || return 96
    local output=''
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == --output ]]; then output="$2"; shift; fi
        shift
    done
    [[ "$output" == *.part ]] || return 95
    [[ -e "$output" ]] && printf 'resume %s\\n' "$(cat "$output")" >> "$TEST_LOG"
    case "${{MOCK_DOWNLOAD:-valid}}" in
        valid) printf 'verified archive' > "$output" ;;
        mismatch) printf 'bad archive' > "$output" ;;
        interrupted) printf 'partial archive' > "$output"; return 22 ;;
    esac
}}
md5sum() {{
    local expected path
    read -r expected path
    printf 'md5 %s %s\\n' "$expected" "$path" >> "$TEST_LOG"
    [[ "$(cat "$path")" == 'verified archive' ]]
}}
''')
    environment = dict(os.environ, BASH_ENV=str(environment_file), SLURM_JOB_ID='1234',
                       SLURMD_NODENAME='cpu01', SLURM_JOB_PARTITION='cpu', SLURM_SUBMIT_DIR=str(tmp_path),
                       ROC_TRANSFER_RAW_ROOT=str(raw), TEST_LOG=str(log))
    environment.pop('MOCK_DOWNLOAD', None)
    environment.pop('MOCK_MISSING_TOOL', None)
    environment.pop('CONDA_PREFIX', None)
    return dict(raw=raw, log=log, environment=environment, tmp=tmp_path)


def run_wrapper(job, arguments=('fantasyid',)):
    assert WRAPPER.is_file(), 'allocated transfer downloader missing'
    return subprocess.run(['bash', str(WRAPPER), *arguments], env=job['environment'],
                          capture_output=True, text=True, timeout=10)


def calls(job):
    return job['log'].read_text() if job['log'].exists() else ''


@pytest.mark.parametrize('choice', ARCHIVES)
def test_download_mapping_resumes_part_and_promotes_only_verified_archive(download_job, choice):
    job = download_job
    folder, name, record, checksum = ARCHIVES[choice]
    destination = job['raw'] / folder
    destination.mkdir(parents=True)
    (destination / (name + '.part')).write_text('previous partial')
    result = run_wrapper(job, (choice,))
    assert result.returncode == 0, result.stderr
    assert (destination / name).read_text() == 'verified archive'
    assert not (destination / (name + '.part')).exists()
    assert not (destination / (name + '.lock')).exists()
    log = calls(job)
    assert f'https://zenodo.org/records/{record}/files/{name}?download=1' in log
    assert '<--continue-at> <->' in log and '<--fail>' in log and '<--location>' in log
    assert f'<--output> <{name}.part>' in log
    assert 'resume previous partial' in log
    assert f'md5 {checksum} {name}.part' in log
    assert 'source ' not in log and 'conda ' not in log
    assert len(list(destination.iterdir())) == 1


@pytest.mark.parametrize('valid', [True, False])
def test_existing_final_is_verified_and_never_overwritten(download_job, valid):
    job = download_job
    destination = job['raw'] / 'fantasyid'
    destination.mkdir(parents=True)
    final = destination / 'FantasyID.tgz'
    content = 'verified archive' if valid else 'existing unverified archive'
    final.write_text(content)
    result = run_wrapper(job)
    assert (result.returncode == 0) is valid
    assert final.read_text() == content
    assert 'curl' not in calls(job)
    assert 'md5 5a6e608762f640db026ac122b4400a72 FantasyID.tgz' in calls(job)
    assert not (destination / 'FantasyID.tgz.lock').exists()


@pytest.mark.parametrize('mode', ['mismatch', 'interrupted'])
def test_failed_transfer_retains_part_without_final_and_releases_lock(download_job, mode):
    job = download_job
    job['environment']['MOCK_DOWNLOAD'] = mode
    result = run_wrapper(job)
    destination = job['raw'] / 'fantasyid'
    assert result.returncode != 0
    assert not (destination / 'FantasyID.tgz').exists()
    assert (destination / 'FantasyID.tgz.part').is_file()
    assert not (destination / 'FantasyID.tgz.lock').exists()
    job['environment'].pop('MOCK_DOWNLOAD')
    second = run_wrapper(job)
    assert second.returncode == 0, second.stderr
    assert (destination / 'FantasyID.tgz').read_text() == 'verified archive'


def test_duplicate_lock_rejects_without_download_or_removing_owner_lock(download_job):
    job = download_job
    lock = job['raw'] / 'fantasyid/FantasyID.tgz.lock'
    lock.mkdir(parents=True)
    result = run_wrapper(job)
    assert result.returncode != 0
    assert lock.is_dir()
    assert 'curl' not in calls(job) and 'md5' not in calls(job)


@pytest.mark.parametrize('location', ['final', 'part', 'folder'])
def test_symlinks_are_not_followed_or_modified(download_job, location):
    job = download_job
    destination = job['raw'] / 'fantasyid'
    destination.mkdir(parents=True)
    target = job['tmp'] / 'protected'
    if location == 'folder':
        destination.rmdir()
        target.mkdir()
        destination.symlink_to(target, target_is_directory=True)
    else:
        target.write_text('protected content')
        name = 'FantasyID.tgz' if location == 'final' else 'FantasyID.tgz.part'
        (destination / name).symlink_to(target)
    result = run_wrapper(job)
    assert result.returncode != 0
    assert 'curl' not in calls(job) and 'md5' not in calls(job)
    if location == 'folder':
        assert list(target.iterdir()) == []
    else:
        assert target.read_text() == 'protected content'


@pytest.mark.parametrize('field,value', [
    ('SLURM_JOB_ID', None), ('SLURM_JOB_ID', 'invalid'), ('SLURMD_NODENAME', None),
    ('SLURMD_NODENAME', 'login01'), ('SLURMD_NODENAME', 'head01'),
    ('SLURM_JOB_PARTITION', 'gpu'), ('SLURM_JOB_PARTITION', None), ('SLURM_SUBMIT_DIR', None),
])
def test_cpu_allocation_guards_fail_before_download_or_writes(download_job, field, value):
    job = download_job
    if value is None:
        job['environment'].pop(field, None)
    else:
        job['environment'][field] = value
    result = run_wrapper(job)
    assert result.returncode != 0
    assert calls(job) == '' and not job['raw'].exists()


@pytest.mark.parametrize('arguments', [(), ('unknown',), ('fantasyid', 'az'), ('../wv',)])
def test_only_one_approved_archive_selector_is_accepted(download_job, arguments):
    result = run_wrapper(download_job, arguments)
    assert result.returncode != 0
    assert calls(download_job) == '' and not download_job['raw'].exists()


def test_wrapper_reuses_cpu_allocation_and_default_shared_destination():
    assert WRAPPER.is_file(), 'allocated transfer downloader missing'
    text = WRAPPER.read_text()
    original = (ROOT / 'scripts/cluster/prepare_anchor_enrollment.sh').read_text()
    for directive in ('account', 'partition', 'nodes', 'ntasks', 'cpus-per-task', 'mem', 'output'):
        expected = next(line for line in original.splitlines() if line.startswith(f'#SBATCH --{directive}='))
        assert expected in text
    assert '#SBATCH --gres=' not in text and '#SBATCH --gpus' not in text
    assert '/shared/b00090279/refer-or-clear/data/raw' in text
    assert 'tar ' not in text and 'unzip ' not in text and 'sbatch ' not in text
    for forbidden in ('miniconda', 'conda activate', 'CONDA_PREFIX', 'python'):
        assert forbidden not in text
    result = subprocess.run(['bash', '-n', str(WRAPPER)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('prefix', [None, '/unrelated/environment'])
def test_pure_download_requires_no_conda_bootstrap_or_prefix(download_job, prefix):
    job = download_job
    if prefix is not None:
        job['environment']['CONDA_PREFIX'] = prefix
    result = run_wrapper(job)
    assert result.returncode == 0, result.stderr
    assert (job['raw'] / 'fantasyid/FantasyID.tgz').read_text() == 'verified archive'
    assert 'source ' not in calls(job) and 'conda ' not in calls(job)


def test_download_uses_public_get_without_authentication(download_job):
    job = download_job
    result = run_wrapper(job, ('wv',))
    assert result.returncode == 0, result.stderr
    invocation = next(line for line in calls(job).splitlines() if line.startswith('curl'))
    assert invocation.startswith('curl <--disable>')
    assert '<--request> <GET>' in invocation
    assert '<--user>' not in invocation and '<--header>' not in invocation
    assert 'Authorization' not in invocation and 'Bearer' not in invocation


@pytest.mark.parametrize('tool', ['curl', 'md5sum'])
def test_missing_download_tool_fails_before_destination_or_download(download_job, tool):
    job = download_job
    job['environment']['MOCK_MISSING_TOOL'] = tool
    result = run_wrapper(job)
    assert result.returncode != 0
    assert not job['raw'].exists()
    assert 'curl' not in calls(job) and 'md5' not in calls(job)
    assert tool in result.stderr
