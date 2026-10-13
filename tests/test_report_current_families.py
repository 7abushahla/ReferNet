import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_report_current_matched import (
    binding, bundle, csv_file, document, reporter, request_binding, rewrite,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def families():
    path = ROOT / 'scripts/report_current_families.py'
    assert path.is_file(), 'family reporting extension is not implemented'
    specification = importlib.util.spec_from_file_location('report_current_families', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize('family', ['face_swap', 'face_morph', 'mixed', '', 'unrecognized_family'])
def test_every_recorded_family_is_separate(families, reporter, bundle, family):
    records = reporter.csv_rows(Path(bundle['request']['records']['path']).read_bytes())
    for row in records:
        if row['attack_family'] == 'text_copy':
            row['attack_family'] = family
    bundle['request']['records'] = csv_file(bundle['root'] / 'records.csv', records)
    for role in ('val', 'test'):
        members = bundle['members'][role]
        members[2]['attack_family'] = family
        bundle['request']['memberships'][role] = csv_file(bundle['root'] / f'{role}-members.csv', members)
        for arm in ('off', 'on'):
            rewrite(bundle, arm, role, 'predictions', lambda rows: rows[2].update(attack_family=family))
    for arm in ('off', 'on'):
        parent = bundle['request']['arms'][arm]['completion']
        receipt = json.loads(Path(parent['path']).read_text())
        receipt['identities']['records_sha256'] = bundle['request']['records']['sha256']
        bundle['request']['arms'][arm]['completion'] = document(Path(parent['path']), receipt)
    result = families.build_family_report(request_binding(bundle))
    test = result['arms']['off']['test']
    assert set(test['families']) == {family, 'text_inpaint'}
    assert test['families'][family]['n_altered'] == 1
    assert test['families'][family]['n_genuine_shared'] == 2
    assert test['families'][family]['family_vs_all_genuine_auc'] == 1.
    assert test['families']['text_inpaint']['native']['altered_miss_count'] == 1
    assert test['genuine']['native']['false_positive_count'] == 0


def test_ties_orientation_and_single_source_threshold(families, reporter, bundle, monkeypatch):
    calls = []
    original = reporter.metrics.far_threshold
    def capture(scores, labels, budget):
        calls.append((scores, labels, budget))
        return original(scores, labels, budget)
    monkeypatch.setattr(reporter.metrics, 'far_threshold', capture)
    result = families.build_family_report(request_binding(bundle))
    baseline = result['arms']['off']['test']
    assert len(calls) == 2 and all(call[2] == .05 for call in calls)
    assert baseline['families']['text_copy']['native']['altered_miss_rate'] == 0.
    assert baseline['families']['text_inpaint']['native']['altered_miss_rate'] == 1.
    assert baseline['genuine']['native']['false_positive_count'] == 0
    assert result['authorization_granted'] is False
    tied = families._errors([dict(bonafide_score=.5)], .5, altered=True)
    assert tied == dict(altered_miss_count=1, altered_miss_rate=1.)
    assert families._errors([dict(bonafide_score=.5)], .5, altered=False)['false_positive_count'] == 0


def test_undefined_threshold_and_auc_are_not_zero(families):
    rows = [dict(image_id='attack', label=1, attack_family='face_swap', bonafide_score=.8)]
    result = families._population(rows, None)
    face = result['families']['face_swap']
    assert face['n_altered'] == 1 and face['n_genuine_shared'] == 0
    assert face['family_vs_all_genuine_auc'] is None
    assert face['transported'] == dict(altered_miss_count=None, altered_miss_rate=None)
    assert result['genuine']['native']['false_positive_rate'] is None
    only_genuine = families._population([dict(label=0, bonafide_score=.4)], .5)
    assert only_genuine['families'] == {}
    assert only_genuine['genuine']['native']['false_positive_rate'] == 1.


def test_c_preserves_113_faces_and_120_text_attacks(families, reporter, bundle):
    records = reporter.csv_rows(Path(bundle['request']['records']['path']).read_bytes())[:4]
    targets = [dict(image_id=f'target-{position:03}', dataset='fantasyid', design_id='target-design',
        component_id=f'target-component-{position}', label=int(position >= 234),
        attack_family='real' if position < 234 else 'face_swap' if position < 347 else 'text_inpaint')
        for position in range(467)]
    bundle['request'].update(protocol='C', records=csv_file(bundle['root'] / 'records.csv', records + targets))
    bundle['request']['memberships']['test'] = csv_file(bundle['root'] / 'test-members.csv',
        [dict(row, role='test') for row in targets])
    for arm in ('off', 'on'):
        parent = bundle['request']['arms'][arm]['completion']
        receipt = json.loads(Path(parent['path']).read_text())
        receipt['identities']['records_sha256'] = bundle['request']['records']['sha256']
        bundle['request']['arms'][arm]['completion'] = document(Path(parent['path']), receipt)
        predictions = [dict(image_id=row['image_id'], design_id=row['design_id'], attack_family=row['attack_family'],
            label=row['label'], bonafide_score=.9 if row['label'] == 0 else .5,
            context_available=True, referral_required=False) for row in targets]
        predicted = csv_file(bundle['root'] / f'{arm}-test-predictions.csv', predictions)
        summary = dict(role='test', selected_epoch=12, n_images=467, accuracy=234 / 467, auc=1.,
                       predictions=predicted, source_checkpoint=receipt['checkpoint'])
        bundle['request']['arms'][arm]['test'] = dict(predictions=predicted,
            summary=document(bundle['root'] / f'{arm}-test-summary.json', summary))
    result = families.build_family_report(request_binding(bundle))
    test = result['arms']['off']['test']
    assert test['n_images'] == 467 and test['genuine']['n_genuine'] == 234
    assert test['families']['face_swap']['n_altered'] == 113
    assert test['families']['text_inpaint']['n_altered'] == 120
    assert test['families']['face_swap']['native']['altered_miss_rate'] == 1.
    assert test['families']['face_swap']['family_vs_all_genuine_auc'] == 1.


def test_complete_frozen_validation_remains_mandatory(families, bundle):
    rewrite(bundle, 'off', 'test', 'summary', lambda value: value.update(accuracy=.01))
    with pytest.raises(ValueError, match='accuracy'):
        families.build_family_report(request_binding(bundle))


def test_metadata_change_after_validation_rejected(families, bundle, monkeypatch):
    original = families.reporter.build_report
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with Path(bundle['request']['arms']['off']['test']['predictions']['path']).open('a') as stream:
            stream.write('\n')
        return result
    monkeypatch.setattr(families.reporter, 'build_report', mutate)
    with pytest.raises(ValueError, match='SHA256'):
        families.build_family_report(request_binding(bundle))


def test_cli_output_is_exclusive_and_inputs_unchanged(families, bundle):
    authority = request_binding(bundle)
    before = {path: path.read_bytes() for path in bundle['root'].iterdir() if path.is_file()}
    output = bundle['root'] / 'families.json'
    command = [sys.executable, str(ROOT / 'scripts/report_current_families.py'),
        '--request', authority['path'], '--request-sha256', authority['sha256'], '--out', str(output)]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())['schema_version'] == 'current_matched_family_report_v1'
    original = output.read_bytes()
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode == 1 and output.read_bytes() == original
    assert all(path.read_bytes() == contents for path, contents in before.items())
