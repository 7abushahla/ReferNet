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
                                attack_family=row['attack_family'], bonafide_score=score, attack_probability=1 - score,
                                context_available=True, referral_required=False)
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


def test_mandatory_referral_changes_deployment_not_saved_score_metrics_or_threshold(reporter, bundle):
    def scores(rows):
        rows[3].update(bonafide_score='.7', attack_probability='.3')
    rewrite(bundle, 'on', 'test', 'predictions', scores)
    rewrite(bundle, 'on', 'test', 'summary', lambda value: value.update(accuracy=.75, auc=.75))
    before = reporter.build_report(request_binding(bundle))
    def unavailable(rows):
        for row in rows:
            row['correction_logit'] = '0.0'
        for index in (0, 3):
            rows[index].update(context_available='False', referral_required='True')
    rewrite(bundle, 'on', 'test', 'predictions', unavailable)
    after = reporter.build_report(request_binding(bundle))
    original, changed = before['arms']['on'], after['arms']['on']
    assert changed['operating_point'] == original['operating_point']
    assert changed['test']['native'] == original['test']['native']
    assert changed['test']['operational'] == original['test']['operational'] == dict(apcer=.5, bpcer=0, hter=.25)
    deployment = changed['test']['deployment_with_mandatory_referral']
    assert deployment['apcer'] == 0
    assert deployment['bpcer'] == .5
    assert deployment['hter'] == .25
    assert deployment['n_cleared'] == 1 and deployment['n_referred'] == 3
    assert deployment['mandatory_referral_count'] == 2
    assert deployment['withheld_due_to_mandatory_referral_count'] == 2
    assert deployment['cleared_ids'] == ['test-1']
    assert deployment['referred_ids'] == ['test-0', 'test-2', 'test-3']
    assert 'classifier_only' in changed['test']['operational_policy']
    assert 'context_available is True' in deployment['clearance_policy']


def test_missing_context_is_unresolved_and_never_authorizes_clearance(reporter, bundle):
    def missing(rows):
        for row in rows:
            del row['context_available']
            del row['referral_required']
    rewrite(bundle, 'off', 'test', 'predictions', missing)
    result = reporter.build_report(request_binding(bundle))
    block = result['arms']['off']['test']['deployment_with_mandatory_referral']
    assert block['unknown_context_count'] == 4
    assert block['mandatory_referral_count'] == 4
    assert block['n_cleared'] == 0
    assert block['apcer'] == 0 and block['bpcer'] == 1
    assert result['arms']['off']['test']['operational']['bpcer'] == .5


@pytest.mark.parametrize('failure', ['context', 'contradiction'])
def test_malformed_or_contradictory_availability_is_rejected(reporter, bundle, failure):
    def corrupt(rows):
        if failure == 'context':
            rows[0]['context_available'] = 'unknown'
        else:
            rows[0]['referral_required'] = 'True'
    rewrite(bundle, 'on', 'test', 'predictions', corrupt)
    with pytest.raises(ValueError):
        reporter.build_report(request_binding(bundle))


def test_undefined_threshold_fails_closed_in_deployment(reporter):
    rows = [dict(image_id='clean', label=0, bonafide_score=1., context_available=True)]
    block = reporter.deployment_with_mandatory_referral(rows, None)
    assert block['status'] == 'undefined_threshold_all_referred'
    assert block['n_cleared'] == 0 and block['bpcer'] == 1 and block['apcer'] is None


def test_operational_bootstrap_is_explicitly_classifier_only(reporter):
    rows = [dict(image_id='genuine', label=0, bonafide_score=.9, context_available=False,
                 dataset='sidtd', design_id='design', component_id='same'),
            dict(image_id='altered', label=1, bonafide_score=.8, context_available=False,
                 dataset='sidtd', design_id='design', component_id='same')]
    value = reporter.paired_bootstrap(dict(test=dict(off=rows, on=rows)), dict(off=.85, on=.85))
    assert value['metric_policies']['apcer'] == 'classifier_only_score_threshold'
    assert value['metric_policies']['bpcer'] == 'classifier_only_score_threshold'
    assert 'mandatory-referral deployment' in value['interpretation']


def test_table_labels_classifier_and_mandatory_referral_separately(reporter, bundle):
    table = reporter.render_table(reporter.build_report(request_binding(bundle)))
    assert 'Classifier-only test APCER' in table
    assert 'Deployment test APCER' in table
    assert 'Deployment test BPCER' in table


def producer_shaped_bundle(bundle, protocol):
    request = bundle['request']
    request['protocol'] = protocol
    if protocol == 'C':
        records_path = Path(request['records']['path'])
        with records_path.open(newline='') as stream:
            records = list(csv.DictReader(stream))
        for row in records:
            if row['image_id'].startswith('test-'):
                row['dataset'] = 'fantasyid'
                if row['attack_family'] == 'text_copy':
                    row['attack_family'] = 'face_swap'
        request['records'] = csv_file(records_path, records)
        members = [dict(row, role='test') for row in records if row['image_id'].startswith('test-')]
        request['memberships']['test'] = csv_file(Path(request['memberships']['test']['path']), members)
        for arm in ('off', 'on'):
            rewrite(bundle, arm, 'test', 'predictions', lambda rows: rows[2].update(attack_family='face_swap'))
            path = Path(request['arms'][arm]['completion']['path'])
            receipt = json.loads(path.read_text())
            receipt['identities']['records_sha256'] = request['records']['sha256']
            request['arms'][arm]['completion'] = document(path, receipt)
    parent_arms = {}
    for arm in ('off', 'on'):
        inputs = request['arms'][arm]
        receipt = json.loads(Path(inputs['completion']['path']).read_text())
        summary_path = Path(inputs['test']['summary']['path'])
        summary = json.loads(summary_path.read_text())
        del summary['source_checkpoint']
        del summary['predictions']
        summary.update(protocol=protocol, aggregate=False, selected_on='source_validation')
        if protocol == 'A':
            summary['fold'] = 0
        else:
            summary.update(source_protocol='A', source_fold=0)
        inputs['test']['summary'] = document(summary_path, summary)
        parent_arms[arm] = dict(checkpoint=receipt['checkpoint'], train_result=inputs['completion'],
            completion=dict(path=str(bundle['root'] / f'{arm}-control.json'), sha256='1' * 64),
            fresh_replay=dict(path=str(bundle['root'] / f'{arm}-fresh-replay.json'), sha256='2' * 64),
            training_identities=receipt['identities'], predictions=inputs['test']['predictions'], summary=inputs['test']['summary'])
    parent = dict(schema_version=f'protocol_{protocol.lower()}_current_score_result_v1', passed=True, protocol=protocol,
        role='test', n_images=4, aggregate=False, seed=42, eval_batch_size=1,
        request=dict(path=str(bundle['root'] / 'scorer-request.json'), sha256='3' * 64),
        evaluation_source_sha256='4' * 64, training_source_sha256='c' * 64,
        evaluation=dict(path=str(bundle['root'] / 'evaluation.json'), sha256='5' * 64),
        registration=dict(path=str(bundle['root'] / 'test-registration.json'), sha256='6' * 64),
        image_ids=[f'test-{index}' for index in range(4)], arms=parent_arms, learning_performed=False)
    if protocol == 'A':
        parent.update(fold=0, threshold_tuning_performed=False)
    else:
        parent.update(deployment='locked_a_selected_models_no_target_training', source_protocol='A', source_fold=0,
            n_source_train=1778, n_source_val=222, published_target_count=6, n_target_support=1, n_target_excluded=1,
            threshold_tuning=False, bonafide_threshold=.5, decision_rule='bonafide_score >= 0.5 is genuine; ties genuine')
    request['scorer_result'] = document(bundle['root'] / 'score-result.json', parent)
    return parent


@pytest.mark.parametrize('protocol', ['A', 'C'])
def test_plain_actual_producer_shaped_summary_uses_bound_parent_without_artifact_edits(reporter, bundle, protocol):
    producer_shaped_bundle(bundle, protocol)
    bound_request = request_binding(bundle)
    before = {path: path.read_bytes() for path in bundle['root'].iterdir() if path.is_file()}
    report = reporter.build_report(bound_request)
    assert report['arms']['off']['test']['native']['accuracy'] == .75
    assert report['arms']['on']['test']['native']['accuracy'] == 1
    assert report['scorer_result'] == bundle['request']['scorer_result']
    assert report['metadata_sha256'][report['scorer_result']['path']] == report['scorer_result']['sha256']
    assert all(path.read_bytes() == content for path, content in before.items())
    for arm in ('off', 'on'):
        summary = json.loads(Path(bundle['request']['arms'][arm]['test']['summary']['path']).read_text())
        assert 'source_checkpoint' not in summary and 'predictions' not in summary
    if protocol == 'C':
        assert report['arms']['off']['test']['native']['other_altered_count'] == 1
        assert report['arms']['off']['test']['native']['subgroups']['copy']['n_altered'] == 0
        assert report['arms']['off']['test']['native']['subgroups']['inpainting']['n_altered'] == 1
    specification = importlib.util.spec_from_file_location('plot_current_matched_bridge', ROOT / 'scripts/plot_current_matched.py')
    plotter = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(plotter)
    report_binding = document(bundle['root'] / 'bound-report.json', report)
    validated = plotter.load_inputs([(report_binding, bound_request)])
    assert validated[0]['report']['scorer_result'] == bundle['request']['scorer_result']


@pytest.mark.parametrize('failure', ['missing-parent', 'hash', 'schema', 'passed', 'protocol', 'role', 'aggregate',
    'count', 'ids', 'training-source', 'checkpoint', 'predictions', 'summary', 'train-result', 'identities',
    'arms', 'fold', 'learning', 'tuning'])
def test_parent_result_provenance_is_not_optional_or_self_asserted(reporter, bundle, failure):
    parent = producer_shaped_bundle(bundle, 'A')
    if failure == 'missing-parent':
        del bundle['request']['scorer_result']
    elif failure == 'hash':
        bundle['request']['scorer_result']['sha256'] = '0' * 64
    else:
        if failure == 'schema':
            parent['schema_version'] = 'unknown'
        elif failure == 'passed':
            parent['passed'] = False
        elif failure == 'protocol':
            parent['protocol'] = 'C'
        elif failure == 'role':
            parent['role'] = 'val'
        elif failure == 'aggregate':
            parent['aggregate'] = True
        elif failure == 'count':
            parent['n_images'] = 99
        elif failure == 'ids':
            parent['image_ids'][1] = parent['image_ids'][0]
        elif failure == 'training-source':
            parent['training_source_sha256'] = 'f' * 64
        elif failure == 'checkpoint':
            parent['arms']['on']['checkpoint']['sha256'] = 'f' * 64
        elif failure in ('predictions', 'summary'):
            parent['arms']['on'][failure] = parent['arms']['off'][failure]
        elif failure == 'train-result':
            parent['arms']['on']['train_result'] = parent['arms']['off']['train_result']
        elif failure == 'identities':
            parent['arms']['on']['training_identities']['source_sha256'] = 'f' * 64
        elif failure == 'arms':
            del parent['arms']['on']
        elif failure == 'fold':
            parent['fold'] = True
        elif failure == 'learning':
            parent['learning_performed'] = True
        else:
            parent['threshold_tuning_performed'] = True
        bundle['request']['scorer_result'] = document(Path(bundle['request']['scorer_result']['path']), parent)
    with pytest.raises(ValueError):
        reporter.build_report(request_binding(bundle))


@pytest.mark.parametrize('field,value', [('source_protocol', 'B'), ('source_fold', True)])
def test_c_parent_retains_original_a_source_selection(reporter, bundle, field, value):
    parent = producer_shaped_bundle(bundle, 'C')
    parent[field] = value
    bundle['request']['scorer_result'] = document(Path(bundle['request']['scorer_result']['path']), parent)
    with pytest.raises(ValueError):
        reporter.build_report(request_binding(bundle))


def test_parent_does_not_override_conflicting_inline_provenance(reporter, bundle):
    parent = producer_shaped_bundle(bundle, 'A')
    summary_path = Path(bundle['request']['arms']['on']['test']['summary']['path'])
    summary = json.loads(summary_path.read_text())
    summary['source_checkpoint'] = parent['arms']['off']['checkpoint']
    updated = document(summary_path, summary)
    bundle['request']['arms']['on']['test']['summary'] = updated
    parent['arms']['on']['summary'] = updated
    bundle['request']['scorer_result'] = document(Path(bundle['request']['scorer_result']['path']), parent)
    with pytest.raises(ValueError, match='contradicts parent'):
        reporter.build_report(request_binding(bundle))
