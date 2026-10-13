import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_report_current_matched import binding, bundle, csv_file, document, reporter


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def preparer():
    path = ROOT / 'scripts/prepare_current_report.py'
    assert path.is_file(), 'metadata report preparer is not implemented'
    specification = importlib.util.spec_from_file_location('prepare_current_report', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load(value):
    return json.loads(Path(value['path']).read_text())


def update(value, change):
    contents = load(value)
    change(contents)
    return document(Path(value['path']), contents)


@pytest.fixture(params=['A0', 'A', 'B', 'C'])
def inputs(request, bundle, reporter):
    kind = request.param
    protocol = 'A' if kind == 'A0' else kind
    index = 0 if kind in ('A0', 'C') else 4
    root = bundle['root']
    report = bundle['request']
    report.update(protocol=protocol, index=index)
    if protocol == 'C':
        records = reporter.csv_rows(Path(report['records']['path']).read_bytes())[:4]
        targets = []
        for position in range(467):
            family = 'real' if position < 234 else 'face_swap' if position < 347 else 'text_inpaint'
            targets.append(dict(image_id=f'target-{position:03}', dataset='fantasyid', design_id='fantasyid/chinese2',
                component_id=f'target-component-{position}', label=int(position >= 234), attack_family=family))
        report['records'] = csv_file(root / 'records.csv', records + targets)
        bundle['members']['test'] = [dict(row, role='test') for row in targets]
        for arm in ('off', 'on'):
            predictions = [dict(image_id=row['image_id'], label=row['label'], design_id=row['design_id'],
                attack_family=row['attack_family'], bonafide_score=.9 if row['label'] == 0 else .1,
                context_available=True, referral_required=False) for row in targets]
            predicted = csv_file(root / f'{arm}-test-predictions.csv', predictions)
            summary = dict(role='test', selected_epoch=12, n_images=467, n_genuine=234, n_altered=233,
                           accuracy=1., auc=1.)
            report['arms'][arm]['test'] = dict(predictions=predicted,
                summary=document(root / f'{arm}-test-summary.json', summary))
    split = csv_file(root / 'split.csv', [dict(image_id=row['image_id'], role=role)
        for role, rows in bundle['members'].items() for row in rows])
    enrollment = document(root / 'enrollment.json', dict(schema_version='fixture_enrollment'))
    authority = dict(schema_version='current_report_preparation_v1', protocol=protocol, index=index,
        records=report['records'], split=split, enrollment=enrollment, arms={})
    source = dict(records=report['records'], split=split, enrollment=enrollment, arms={},
                  output_root=str(root / 'scientific-run'))
    legacy = kind in ('A0', 'C')
    for arm in ('off', 'on'):
        item = report['arms'][arm]
        item['completion'] = update(item['completion'], lambda value: value['identities'].update(
            records_sha256=report['records']['sha256'], split_sha256=split['sha256'],
            enrollment_sha256=enrollment['sha256']))
        train = load(item['completion'])
        fresh = document(root / f'{arm}-fresh.json', dict(passed=True, fresh_process=True, pid=123,
            train_result=item['completion'], checkpoint=train['checkpoint'], identities=train['identities'],
            summary=load(item['val']['summary']), replay=dict(passed=True)))
        authority['arms'][arm] = dict(train_result=item['completion'], val_summary=item['val']['summary'], fresh_replay=fresh)
        completion = document(root / f'{arm}-completion.json', dict(passed=True, arm=arm, epochs_completed=100,
            train_result=item['completion'], checkpoint=train['checkpoint'], fresh_replay=fresh))
        source['arms'][arm] = dict(train_result=item['completion'], completion=completion, fresh_replay=fresh)
        if legacy:
            item['test']['summary'] = update(item['test']['summary'], lambda value: (
                value.pop('source_checkpoint', None), value.pop('predictions', None),
                value.update(protocol=protocol, aggregate=False, selected_on='source_validation')))
    if legacy:
        parent_arms = {arm: dict(source['arms'][arm], checkpoint=load(report['arms'][arm]['completion'])['checkpoint'],
            training_identities=load(report['arms'][arm]['completion'])['identities'], **report['arms'][arm]['test'])
            for arm in ('off', 'on')}
        source.update(schema_version='protocol_a_current_score_v1', protocol='A', fold=0,
                      training_source_sha256='c' * 64)
        if protocol == 'C':
            authority['target_split'] = csv_file(root / 'target-split.csv', [dict(image_id=row['image_id'])
                for row in bundle['members']['test']])
            authority['target_enrollment'] = document(root / 'target-enrollment.json', dict(query_ids=[
                row['image_id'] for row in bundle['members']['test']]))
            source = dict(schema_version='protocol_c_current_score_v1', protocol='C',
                training_source_sha256='c' * 64, source_selection={key: source[key]
                for key in ('records', 'split', 'enrollment', 'arms')},
                output_root=str(root / 'scientific-run'),
                target_split=authority['target_split'], target_enrollment=authority['target_enrollment'])
        source_binding = document(root / 'source-request.json', source)
        result = dict(schema_version=f'protocol_{protocol.lower()}_current_score_result_v1', protocol=protocol,
            passed=True, aggregate=False, role='test', seed=42, eval_batch_size=1, learning_performed=False,
            training_source_sha256='c' * 64, request=source_binding, n_images=len(bundle['members']['test']),
            image_ids=sorted(row['image_id'] for row in bundle['members']['test']), arms=parent_arms)
        result.update({'fold': 0, 'threshold_tuning_performed': False} if protocol == 'A' else
                      {'source_fold': 0, 'source_protocol': 'A', 'threshold_tuning': False})
    else:
        source.update(schema_version=f'protocol_{protocol.lower()}_request_v1', protocol=protocol,
                      source_sha256='c' * 64, **{('fold' if protocol == 'A' else 'group'): index})
        if protocol == 'B':
            authority['target_enrollment'] = document(root / 'target-enrollment.json',
                dict(query_ids=sorted(row['image_id'] for row in bundle['members']['test'])))
            source['target_enrollment'] = authority['target_enrollment']
        source_binding = document(root / 'source-request.json', source)
        authorization = document(root / 'evaluation.json', dict(schema_version=f'protocol_{protocol.lower()}_evaluation_v1',
            protocol=protocol, **{('fold' if protocol == 'A' else 'group'): index}, source_validation_selected=True,
            authorized_image_ids=sorted(row['image_id'] for row in bundle['members']['test']),
            records_sha256=report['records']['sha256'], split_sha256=split['sha256'],
            enrollment_manifest_sha256=enrollment['sha256'],
            checkpoints={arm: load(report['arms'][arm]['completion'])['checkpoint'] for arm in ('off', 'on')}))
        for arm in ('off', 'on'):
            item = report['arms'][arm]
            item['completion'] = update(item['completion'], lambda value: value.update(request=source_binding,
                summary=item['val']['summary'], protocol=protocol, **{('fold' if protocol == 'A' else 'group'): index}))
            authority['arms'][arm]['train_result'] = item['completion']
            fresh_binding = authority['arms'][arm]['fresh_replay']
            authority['arms'][arm]['fresh_replay'] = update(fresh_binding, lambda value: (
                value.pop('train_result'), value.pop('replay'), value.update(parent_result=item['completion'],
                request=source_binding, comparison=dict(passed=True))))
            item['test']['summary'] = update(item['test']['summary'], lambda value: value.update(
                protocol=protocol, authorization=authorization, **{('fold' if protocol == 'A' else 'group'): index}))
        result = dict(schema_version=f'protocol_{protocol.lower()}_evaluation_result_v1', protocol=protocol,
            **{('fold' if protocol == 'A' else 'group'): index}, passed=True, same_query_population=True,
            request=source_binding, authorization=authorization,
            summaries={arm: report['arms'][arm]['test']['summary'] for arm in ('off', 'on')})
    authority['scorer_result'] = document(root / 'result.json', result)
    return dict(authority=authority, root=root, members=bundle['members'], kind=kind, source=source_binding)


def mock_contracts(preparer, inputs, monkeypatch):
    contract = SimpleNamespace(source_val=tuple(SimpleNamespace(image_id=row['image_id'])
        for row in inputs['members']['val']), target_test=tuple(SimpleNamespace(image_id=row['image_id'])
        for row in inputs['members']['test']))
    for name in ('load_protocol_a_contract', 'load_protocol_b_contract', 'load_protocol_c_contract'):
        monkeypatch.setattr(preparer, name, lambda *args, **kwargs: contract)
    target = SimpleNamespace(query_ids=tuple(row.image_id for row in contract.target_test))
    monkeypatch.setattr(preparer, 'load_b_target', lambda *args, **kwargs: target)
    monkeypatch.setattr(preparer, 'load_c_target', lambda *args, **kwargs: target)


def prepare(preparer, inputs, monkeypatch, output=None):
    mock_contracts(preparer, inputs, monkeypatch)
    authority = document(inputs['root'] / 'preparation-request.json', inputs['authority'])
    return preparer.prepare_report(authority, output or inputs['root'] / 'prepared')


def test_actual_producer_shapes_preserve_native_files(preparer, inputs, monkeypatch, reporter):
    before = {path: path.read_bytes() for path in inputs['root'].iterdir() if path.is_file()}
    result = prepare(preparer, inputs, monkeypatch)
    assert result['authorization_granted'] is False
    assert result['schema_version'] == 'current_report_preparation_result_v1'
    report_request = load(result['report_request'])
    assert ('scorer_result' in report_request) == (inputs['kind'] in ('A0', 'C'))
    for arm in ('off', 'on'):
        assert report_request['arms'][arm]['val']['summary'] == inputs['authority']['arms'][arm]['val_summary']
    report = reporter.build_report(result['report_request'])
    assert report['protocol'] == inputs['authority']['protocol']
    assert report['index'] == inputs['authority']['index']
    assert report['arms']['off']['operating_point']['target_apcer'] == .05
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert binding(Path(result['report_request']['path'])) == result['report_request']
    assert Path(result['report_request']['path'] + '.sha256').read_text().strip() == result['report_request']['sha256']


@pytest.mark.parametrize('failure', ['partial', 'source', 'split', 'enrollment', 'fresh', 'summary', 'population', 'hash'])
def test_invalid_chains_fail_closed(preparer, inputs, monkeypatch, failure):
    authority = inputs['authority']
    if failure in ('partial', 'source', 'split', 'enrollment'):
        train = authority['arms']['off']['train_result']
        def change(value):
            if failure == 'partial':
                value['epochs_completed'] = 99
            else:
                value['identities'][{'source': 'source_sha256', 'split': 'split_sha256',
                                     'enrollment': 'enrollment_sha256'}[failure]] = 'f' * 64
        authority['arms']['off']['train_result'] = update(train, change)
    elif failure == 'fresh':
        authority['arms']['off']['fresh_replay'] = update(authority['arms']['off']['fresh_replay'],
                                                       lambda value: value.update(passed=False))
    elif failure == 'summary':
        authority['arms']['off']['val_summary'] = update(authority['arms']['off']['val_summary'],
                                                       lambda value: value.update(accuracy=.123))
    elif failure == 'population':
        inputs['members']['test'].pop()
    else:
        authority['scorer_result']['sha256'] = '0' * 64
    with pytest.raises((ValueError, KeyError)):
        prepare(preparer, inputs, monkeypatch)
    assert not (inputs['root'] / 'prepared').exists()


def test_existing_output_never_overwritten(preparer, inputs, monkeypatch):
    output = inputs['root'] / 'prepared'
    output.mkdir()
    sentinel = output / 'keep'
    sentinel.write_text('original')
    with pytest.raises((ValueError, FileExistsError)):
        prepare(preparer, inputs, monkeypatch, output)
    assert sentinel.read_text() == 'original'


def test_output_under_input_parent_rejected(preparer, inputs, monkeypatch):
    with pytest.raises(ValueError, match='output'):
        prepare(preparer, inputs, monkeypatch, inputs['root'] / 'scientific-run' / 'reports')


def test_missing_native_val_summary_rejected_without_reconstruction(preparer, inputs, monkeypatch):
    Path(inputs['authority']['arms']['off']['val_summary']['path']).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare(preparer, inputs, monkeypatch)


def test_cli_invalid_returns_one_without_output(tmp_path, preparer):
    authority = document(tmp_path / 'bad.json', dict(schema_version='unknown'))
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/prepare_current_report.py'),
        '--request', authority['path'], '--request-sha256', authority['sha256'],
        '--output-dir', str(tmp_path / 'out')], capture_output=True, text=True)
    assert result.returncode == 1
    assert not (tmp_path / 'out').exists()


def test_import_does_not_load_scientific_stack(preparer):
    code = f"import sys; sys.path.insert(0, {str(ROOT / 'scripts')!r}); import prepare_current_report; assert not any(name in sys.modules for name in ('torch', 'cv2', 'PIL', 'refer_or_clear.models.direct_reference'))"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_fresh_native_metrics_do_not_invent_an_exact_loss_gate(preparer, inputs):
    authority = inputs['authority']['arms']['off']
    train = load(authority['train_result'])
    fresh = load(authority['fresh_replay'])
    fresh['summary']['image_loss'] = .2 + 1e-8
    fresh_binding = document(inputs['root'] / 'loss-fresh.json', fresh)
    summary = dict(load(authority['val_summary']), image_loss=.2)
    preparer._fresh(preparer.Metadata(), fresh_binding, authority['train_result'], train,
                    inputs['source'], inputs['kind'] in ('A0', 'C'), summary)


@pytest.mark.parametrize('fold,n_val,n_test', [(0, 222, 222), (4, 223, 223), (9, 222, 223)])
def test_real_official_a_memberships_without_pixels(preparer, fold, n_val, n_test):
    request = dict(protocol='A', index=fold, records=binding(ROOT / 'data/processed/records.csv'),
        split=binding(ROOT / f'data/processed/splits/protocol_a_it{fold}.csv'))
    populations = preparer._populations(preparer.Metadata(), request)
    assert len(populations['val']) == n_val
    assert len(populations['test']) == n_test
    assert not set(populations['val']) & set(populations['test'])


def test_real_wrong_fold_and_foreign_records_rejected(preparer, tmp_path):
    request = dict(protocol='A', index=9, records=binding(ROOT / 'data/processed/records.csv'),
        split=binding(ROOT / 'data/processed/splits/protocol_a_it4.csv'))
    with pytest.raises(ValueError, match='hash'):
        preparer._populations(preparer.Metadata(), request)
    altered = tmp_path / 'records.csv'
    altered.write_bytes((ROOT / 'data/processed/records.csv').read_bytes() + b'\n')
    request.update(index=4, records=binding(altered))
    with pytest.raises(ValueError, match='records'):
        preparer._populations(preparer.Metadata(), request)


def test_cli_success_emits_bound_report_request(preparer, inputs, monkeypatch, capsys):
    mock_contracts(preparer, inputs, monkeypatch)
    authority = document(inputs['root'] / 'prepare.json', inputs['authority'])
    status = preparer.main(['--request', authority['path'], '--request-sha256', authority['sha256'],
                           '--output-dir', str(inputs['root'] / 'prepared')])
    assert status == 0
    result = json.loads(capsys.readouterr().out)
    assert result['metadata_ready'] is True and result['authorization_granted'] is False
    assert load(result['report_request'])['records'] == inputs['authority']['records']


def test_metadata_mutation_during_validation_removes_only_new_output(preparer, inputs, monkeypatch):
    original = preparer.reporter.build_report
    def mutate(*args, **kwargs):
        report = original(*args, **kwargs)
        with Path(inputs['authority']['split']['path']).open('a') as stream:
            stream.write('\n')
        return report
    monkeypatch.setattr(preparer.reporter, 'build_report', mutate)
    with pytest.raises(ValueError, match='SHA256'):
        prepare(preparer, inputs, monkeypatch)
    assert not (inputs['root'] / 'prepared').exists()
    assert Path(inputs['authority']['records']['path']).is_file()


@pytest.mark.parametrize('index', [True, -1, 10, '4'])
def test_invalid_index_rejected(preparer, inputs, monkeypatch, index):
    inputs['authority']['index'] = index
    with pytest.raises(ValueError):
        prepare(preparer, inputs, monkeypatch)


def test_symlink_metadata_rejected(preparer, inputs, monkeypatch):
    records = inputs['authority']['records']
    alias = inputs['root'] / 'alias.csv'
    alias.symlink_to(records['path'])
    inputs['authority']['records'] = dict(records, path=str(alias))
    with pytest.raises(ValueError, match='regular'):
        prepare(preparer, inputs, monkeypatch)


@pytest.mark.parametrize('inputs', ['C'], indirect=True)
def test_no_generic_c_schema_invented(preparer, inputs, monkeypatch):
    inputs['authority']['scorer_result'] = update(inputs['authority']['scorer_result'],
        lambda value: value.update(schema_version='protocol_c_evaluation_result_v1'))
    with pytest.raises(ValueError, match='unsupported current producer'):
        prepare(preparer, inputs, monkeypatch)
