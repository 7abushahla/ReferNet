import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from refer_or_clear import metrics


def binding(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def document(path, value):
    path.write_text(json.dumps(value, allow_nan=False))
    return binding(path)


def csv_file(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return binding(path)


@pytest.fixture
def reporter():
    path = ROOT / 'scripts/report_current_matched.py'
    assert path.is_file(), 'matched metadata reporter is not implemented'
    specification = importlib.util.spec_from_file_location('report_current_matched', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path):
    records, members = [], {}
    for role in ('val', 'test'):
        members[role] = []
        for position, (label, family) in enumerate([(0, 'real'), (0, 'real'), (1, 'text_copy'), (1, 'text_inpaint')]):
            row = dict(image_id=f'{role}-{position}', dataset='sidtd', design_id='design-0',
                       component_id=f'{role}-component-{position // 2}', label=label, attack_family=family)
            records.append(row)
            members[role].append(dict(row, role=role))
    request = dict(schema_version='current_matched_report_request_v1', protocol='B', index=0,
                   records=csv_file(tmp_path / 'records.csv', records),
                   memberships={role: csv_file(tmp_path / f'{role}-members.csv', rows) for role, rows in members.items()}, arms={})
    for arm in ('off', 'on'):
        checkpoint = dict(path=str(tmp_path / f'{arm}-best.pt'), sha256=('a' if arm == 'off' else 'b') * 64)
        roles = {}
        for role in ('val', 'test'):
            scores = [.9, .5, .2, .7] if arm == 'off' else [.8, .6, .1, .3]
            predictions = [dict(image_id=row['image_id'], label=row['label'], design_id=row['design_id'],
                                attack_family=row['attack_family'], bonafide_score=score, attack_probability=1 - score)
                           for row, score in zip(members[role], scores)]
            predictions_binding = csv_file(tmp_path / f'{arm}-{role}-predictions.csv', predictions)
            summary = dict(role=role, selected_epoch=12, n_images=4, n_genuine=2, n_altered=2,
                           accuracy=metrics.deepid_acc(scores, [0, 0, 1, 1]), auc=metrics.auc(scores, [0, 0, 1, 1]))
            if role == 'test':
                summary.update(source_checkpoint=checkpoint, predictions=predictions_binding)
            roles[role] = dict(predictions=predictions_binding, summary=document(tmp_path / f'{arm}-{role}-summary.json', summary))
        receipt = dict(passed=True, phase='train', diagnostic_only=False, arm=arm, epochs_completed=100, selected_epoch=12,
                       checkpoint=checkpoint, predictions=roles['val']['predictions'],
                       replay=dict(passed=True),
                       identities=dict(source_sha256='c' * 64, records_sha256=request['records']['sha256'],
                                       split_sha256='d' * 64, enrollment_sha256='e' * 64))
        request['arms'][arm] = dict(completion=document(tmp_path / f'{arm}-train.json', receipt), **roles)
    return dict(request=request, path=tmp_path / 'request.json', members=members, root=tmp_path)


def request_binding(bundle):
    return document(bundle['path'], bundle['request'])


def rewrite(bundle, arm, role, artifact, change):
    path = Path(bundle['request']['arms'][arm][role][artifact]['path'])
    if artifact == 'summary':
        value = json.loads(path.read_text())
        change(value)
        updated = document(path, value)
    else:
        with path.open(newline='') as stream:
            value = list(csv.DictReader(stream))
        change(value)
        updated = csv_file(path, value)
    bundle['request']['arms'][arm][role][artifact] = updated
    if artifact == 'predictions' and role == 'val':
        receipt_binding = bundle['request']['arms'][arm]['completion']
        receipt_path = Path(receipt_binding['path'])
        receipt = json.loads(receipt_path.read_text())
        receipt['predictions'] = updated
        bundle['request']['arms'][arm]['completion'] = document(receipt_path, receipt)
    if artifact == 'predictions' and role == 'test':
        summary_binding = bundle['request']['arms'][arm][role]['summary']
        summary_path = Path(summary_binding['path'])
        summary = json.loads(summary_path.read_text())
        summary['predictions'] = updated
        bundle['request']['arms'][arm][role]['summary'] = document(summary_path, summary)


@pytest.mark.parametrize('protocol', ['A', 'B', 'C'])
def test_saved_metrics_and_source_threshold_transport(reporter, bundle, protocol):
    bundle['request']['protocol'] = protocol
    result = reporter.build_report(request_binding(bundle))
    assert result['protocol'] == protocol
    assert result['arms']['off']['test']['native']['accuracy'] == .75
    assert result['arms']['on']['test']['native']['accuracy'] == 1
    assert result['arms']['off']['test']['native']['genuine_false_positive_count'] == 0
    assert result['arms']['off']['test']['native']['subgroups']['copy']['n_altered'] == 1
    assert result['arms']['off']['test']['native']['subgroups']['inpainting']['n_genuine'] == 2
    threshold = metrics.far_threshold([.9, .5, .2, .7], [0, 0, 1, 1], .05)
    assert result['arms']['off']['operating_point']['threshold'] == threshold
    assert result['arms']['off']['test']['operational']['apcer'] == 0
    assert result['arms']['off']['test']['operational']['bpcer'] == .5
    assert result['authorization_granted'] is False


def test_test_scores_never_choose_threshold(reporter, bundle, monkeypatch):
    calls = []
    original = metrics.far_threshold
    def recorded(scores, labels, target):
        calls.append(list(scores))
        return original(scores, labels, target)
    monkeypatch.setattr(metrics, 'far_threshold', recorded)
    rewrite(bundle, 'off', 'test', 'predictions', lambda rows: rows[0].update(bonafide_score='.8', attack_probability='.2'))
    result = reporter.build_report(request_binding(bundle))
    assert calls == [[.9, .5, .2, .7], [.8, .6, .1, .3]]
    assert result['arms']['off']['operating_point']['frozen_on'] == 'source_val'


@pytest.mark.parametrize('failure', ['hash', 'duplicate', 'missing', 'label', 'design', 'family', 'role', 'nan', 'probability', 'summary', 'subgroup', 'checkpoint'])
def test_invalid_role_artifacts_fail_closed(reporter, bundle, failure):
    if failure == 'hash':
        bundle['request']['arms']['on']['test']['predictions']['sha256'] = '0' * 64
    elif failure == 'summary':
        rewrite(bundle, 'on', 'test', 'summary', lambda value: value.update(accuracy=.2))
    elif failure == 'subgroup':
        rewrite(bundle, 'on', 'test', 'summary', lambda value: value.update(subgroups=dict(copy=dict(n_altered=22))))
    elif failure == 'checkpoint':
        rewrite(bundle, 'on', 'test', 'summary', lambda value: value['source_checkpoint'].update(sha256='0' * 64))
    else:
        def corrupt(rows):
            if failure == 'duplicate':
                rows[1] = dict(rows[0])
            elif failure == 'missing':
                rows.pop()
            elif failure == 'label':
                rows[0]['label'] = '1'
            elif failure == 'design':
                rows[0]['design_id'] = 'foreign'
            elif failure == 'family':
                rows[0]['attack_family'] = 'text_copy'
            elif failure == 'role':
                for row in rows:
                    row['role'] = 'val'
            elif failure == 'nan':
                rows[0]['bonafide_score'] = 'nan'
            elif failure == 'probability':
                rows[0]['attack_probability'] = '.9'
        rewrite(bundle, 'on', 'test', 'predictions', corrupt)
    with pytest.raises(ValueError):
        reporter.build_report(request_binding(bundle))


@pytest.mark.parametrize('failure', ['unfinished', 'source', 'record-hash', 'selected', 'arm', 'replay', 'registration'])
def test_completed_matched_training_required(reporter, bundle, failure):
    receipt_path = Path(bundle['request']['arms']['on']['completion']['path'])
    value = json.loads(receipt_path.read_text())
    if failure == 'unfinished':
        value['epochs_completed'] = 99
    elif failure == 'source':
        value['identities']['source_sha256'] = 'f' * 64
    elif failure == 'record-hash':
        value['identities']['records_sha256'] = 'f' * 64
    elif failure == 'selected':
        value['selected_epoch'] = True
    elif failure == 'replay':
        value['replay']['passed'] = False
    elif failure == 'registration':
        value['identities']['registration_sha256'] = 'f' * 64
    else:
        value['arm'] = 'off'
    bundle['request']['arms']['on']['completion'] = document(receipt_path, value)
    with pytest.raises(ValueError):
        reporter.build_report(request_binding(bundle))


def test_prediction_order_is_explicitly_joined_by_id(reporter, bundle):
    original = reporter.build_report(request_binding(bundle))
    rewrite(bundle, 'on', 'test', 'predictions', lambda rows: rows.reverse())
    reordered = reporter.build_report(request_binding(bundle))
    for arm in ('off', 'on'):
        for role in ('val', 'test'):
            for key in ('native', 'operational'):
                assert reordered['arms'][arm][role][key] == original['arms'][arm][role][key]
    assert reordered['membership_order'] == original['membership_order']
    assert reordered['metadata_sha256'] != original['metadata_sha256']


def test_bootstrap_is_paired_deterministic_fixed_threshold_and_conditional(reporter, bundle, monkeypatch):
    calls = []
    original = metrics.far_threshold
    def recorded(scores, labels, target):
        calls.append(target)
        return original(scores, labels, target)
    monkeypatch.setattr(metrics, 'far_threshold', recorded)
    first = reporter.build_report(request_binding(bundle), bootstrap=True)
    assert calls == [.05, .05]
    assert first['paired_bootstrap'] == reporter.build_report(request_binding(bundle), bootstrap=True)['paired_bootstrap']
    bootstrap = first['paired_bootstrap']
    assert bootstrap['seed'] == 42 and bootstrap['replicates'] == 1000
    assert bootstrap['cluster_fields'] == ['dataset', 'design_id', 'component_id']
    assert bootstrap['roles']['test']['n_clusters'] == 2
    assert 'fixed trained checkpoints' in bootstrap['interpretation']
    assert bootstrap['roles']['test']['metrics']['accuracy']['estimate_on_minus_off'] == .25
    assert bootstrap['roles']['test']['metrics']['accuracy']['interval_95'] == [0, .5]
    assert bootstrap['roles']['test']['metrics']['auc']['undefined_replicates'] > 0


def test_unresolved_lineage_skips_bootstrap_not_native_metrics(reporter, bundle):
    path = Path(bundle['request']['records']['path'])
    with path.open(newline='') as stream:
        records = list(csv.DictReader(stream))
    records[0]['component_id'] = ''
    bundle['request']['records'] = csv_file(path, records)
    membership_path = Path(bundle['request']['memberships']['val']['path'])
    csv_file(membership_path, [dict(row, role='val') for row in records[:4]])
    bundle['request']['memberships']['val'] = binding(membership_path)
    for arm in ('off', 'on'):
        receipt_path = Path(bundle['request']['arms'][arm]['completion']['path'])
        receipt = json.loads(receipt_path.read_text())
        receipt['identities']['records_sha256'] = bundle['request']['records']['sha256']
        bundle['request']['arms'][arm]['completion'] = document(receipt_path, receipt)
    result = reporter.build_report(request_binding(bundle), bootstrap=True)
    assert result['paired_bootstrap']['roles']['val']['status'] == 'skipped'
    assert result['paired_bootstrap']['roles']['test']['status'] == 'complete'
    assert result['arms']['off']['val']['native']['accuracy'] == .75


def test_metadata_only_cli_preserves_inputs_and_existing_output(reporter, bundle):
    request = request_binding(bundle)
    before = {path: path.read_bytes() for path in bundle['root'].iterdir() if path.is_file()}
    output = bundle['root'] / 'report.json'
    command = [sys.executable, str(ROOT / 'scripts/report_current_matched.py'), '--request', request['path'],
               '--request-sha256', request['sha256'], '--out', str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())['authorization_granted'] is False
    assert all(path.read_bytes() == content for path, content in before.items())
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_import_never_imports_scientific_models_or_pixel_readers():
    probe = '''import builtins, importlib.util
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'cv2', 'PIL') or name.startswith(('refer_or_clear.models', 'refer_or_clear.pilots')):
        raise AssertionError('forbidden import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location('report_current_matched', 'scripts/report_current_matched.py')
spec.loader.exec_module(importlib.util.module_from_spec(spec))
'''
    result = subprocess.run([sys.executable, '-c', probe], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('failure', ['schema', 'protocol', 'index', 'extra-arm', 'extra-role', 'member-label', 'member-role', 'overlap'])
def test_explicit_request_and_membership_bounds(reporter, bundle, failure):
    request = bundle['request']
    if failure == 'schema':
        request['schema_version'] = 'unknown'
    elif failure == 'protocol':
        request['protocol'] = 'unknown'
    elif failure == 'index':
        request['index'] = True
    elif failure == 'extra-arm':
        request['arms']['alternative'] = request['arms']['on']
    elif failure == 'extra-role':
        request['memberships']['train'] = request['memberships']['val']
    else:
        path = Path(request['memberships']['val']['path'])
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if failure == 'member-label':
            rows[0]['label'] = '1'
        elif failure == 'member-role':
            rows[0]['role'] = 'test'
        else:
            rows = [dict(row, role='val') for row in bundle['members']['test']]
        request['memberships']['val'] = csv_file(path, rows)
    with pytest.raises(ValueError):
        reporter.build_report(request_binding(bundle))


def test_absent_altered_source_validation_has_no_operational_threshold(reporter):
    rows = [dict(image_id='clean', label=0, attack_family='real', bonafide_score=.5)]
    assert metrics.far_threshold([.5], [0], .05) is None
    assert reporter.operational_rates(rows, None) == dict(apcer=None, bpcer=None, hter=None)
    assert reporter.native_metrics(rows)['accuracy'] == 1
    assert reporter.native_metrics(rows)['auc'] is None
    assert reporter.native_metrics(rows)['subgroups']['copy']['accuracy'] is None


def test_report_does_not_open_receipt_checkpoint_or_record_pixel_paths(reporter, bundle, monkeypatch):
    original = Path.read_bytes
    def guarded(path):
        assert path.suffix in ('.csv', '.json'), 'nonmetadata access'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', guarded)
    reporter.build_report(request_binding(bundle))
