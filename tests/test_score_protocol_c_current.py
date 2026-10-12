import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from test_score_protocol_a_current import ready as a_ready, put


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/score_protocol_c_current.py'


def launcher():
    assert SCRIPT.is_file(), 'Protocol C scorer is missing'
    spec = importlib.util.spec_from_file_location('protocol_c_score_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ready(a_ready, monkeypatch):
    module = launcher()
    module.driver = a_ready['module']
    source = {name: a_ready['request'][name] for name in ('records', 'split', 'enrollment', 'weights', 'config', 'arms')}
    tmp = a_ready['tmp']
    target_split = put(tmp / 'c-split.json', {'synthetic': True})
    monkeypatch.setattr(module.contract_api, 'TARGET_SPLIT_SHA256', target_split['sha256'])
    identifiers = [f'query-{index:04d}' for index in range(467)]
    target = put(tmp / 'c-target.json', {'query_ids': identifiers})
    enrollment = put(tmp / 'c-enrollment.json', {'output_root': str(tmp / 'c-original')})
    root = put(tmp / 'c-root.json', {'original_root': str(tmp / 'c-original'),
        'relocated_root': str(tmp / 'c-relocated')})
    request = dict(schema_version='protocol_c_current_score_v1', protocol='C', seed=42,
        eval_batch_size=1, training_source_sha256=module.driver.TRAINING_SOURCE_SHA256,
        evaluation_source_sha256=a_ready['request']['evaluation_source_sha256'],
        critical_source_hashes=a_ready['request']['critical_source_hashes'], source_selection=source,
        target_split=target_split, target_enrollment=target, enrollment=enrollment, root=root,
        raw_root=str(tmp / 'never-open-c-raw'), output_root=str(tmp / 'c-score'))
    binding = put(tmp / 'c-score-request.json', request)
    monkeypatch.setattr(module.driver, 'load_request', lambda *args: pytest.fail('fake A authority'))
    monkeypatch.setattr(module.driver, 'score', lambda *args: pytest.fail('A test scoring'))
    return dict(module=module, request=request, binding=binding, original=a_ready, tmp=tmp)


def test_interface_exists():
    module = launcher()
    assert all(callable(getattr(module, name)) for name in ('load_request', 'score', 'main'))


def test_allocation_before_authorities(monkeypatch):
    module = launcher()
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(ValueError, match='allocated'):
        module.main(['--request', '/never-open.json', '--request-sha256', 'a' * 64])


def test_source_selection_is_not_a_test_authority(ready):
    module = ready['module']
    request, config = module.load_request(ready['binding'])
    source = request['source_selection']
    assert not {'protocol', 'fold', 'schema_version', 'root', 'raw_root'} & set(source)
    original = ready['original']
    for arm in ('off', 'on'):
        result = module.driver.validate_arm(source, config, arm, original['val_rows'], original['train_rows'])
        assert result['result']['selected_epoch'] == 2


@pytest.mark.parametrize('defect', ['inventory', 'missing_critical', 'changed_critical'])
def test_current_c_inventory_and_original_critical_sources_required(ready, defect):
    request = ready['request']
    if defect == 'inventory':
        request['evaluation_source_sha256'] = 'a' * 64
    elif defect == 'missing_critical':
        request['critical_source_hashes'].pop(next(iter(request['critical_source_hashes'])))
    else:
        request['critical_source_hashes'][next(iter(request['critical_source_hashes']))] = 'a' * 64
    with pytest.raises(ValueError, match='source|critical'):
        ready['module'].load_request(put(Path(ready['binding']['path']), request))


@pytest.mark.parametrize('name', ['source_sha256', 'weights_sha256', 'records_sha256', 'split_sha256',
    'enrollment_sha256', 'registration_sha256', 'phase_authority_sha256', 'preprocess_sha256',
    'fresh_initial_state_sha256', 'diagnostic_request_sha256', 'diagnostic_gate'])
def test_all_matched_parent_checks_retained(name):
    module = launcher()
    original = dict(identities={key: 'a' * 64 for key in ('source_sha256', 'weights_sha256',
        'records_sha256', 'split_sha256', 'enrollment_sha256', 'registration_sha256',
        'phase_authority_sha256', 'preprocess_sha256')},
        result=dict(fresh_initial_state_sha256='a' * 64, diagnostic_gate={'same': True}),
        authority=dict(diagnostic_request_sha256='a' * 64))
    selected = dict(off=copy.deepcopy(original), on=copy.deepcopy(original))
    container = 'identities' if name in original['identities'] else 'authority' if name in original['authority'] else 'result'
    selected['on'][container][name] = 'b' * 64
    with pytest.raises(ValueError, match='matched'):
        module.validate_pair(selected)


@pytest.mark.parametrize('change', [lambda r: r.update(protocol='A'),
    lambda r: r.update(fold=0), lambda r: r.update(eval_batch_size=True),
    lambda r: r['source_selection'].update(raw_root='/fake-a-test'),
    lambda r: r['source_selection']['arms'].pop('on'),
    lambda r: r['source_selection']['records'].update(sha256='a' * 64)])
def test_strict_source_request(ready, change):
    change(ready['request'])
    binding = put(Path(ready['binding']['path']), ready['request'])
    with pytest.raises(ValueError):
        ready['module'].load_request(binding)


def test_parent_failure_precedes_any_c_authority(ready, monkeypatch):
    module = ready['module']
    monkeypatch.setattr(module.driver, 'require_gpu_allocation', lambda: None)
    source = ready['request']['source_selection']
    on_result = module.driver.document(source['arms']['on']['train_result'])
    on_result['epochs_completed'] = 99
    source['arms']['on']['train_result'] = put(Path(source['arms']['on']['train_result']['path']), on_result)
    ready['request']['target_enrollment']['path'] = '/never-open-c-authority.json'
    binding = put(Path(ready['binding']['path']), ready['request'])
    monkeypatch.setattr(module.consumer, 'load_reference_context', lambda *args, **kwargs: pytest.fail('C access before parent gates'))
    with pytest.raises(ValueError, match='complete'):
        module.score(binding)
    assert not Path(ready['request']['output_root']).exists()


def predictions():
    rows = []
    for index in range(467):
        label = index % 2
        logit = 2. if label else -2.
        probability = torch.tensor(logit).sigmoid().item()
        rows.append(dict(image_id=f'query-{index:04d}', design_id=f'design-{index % 3}',
            attack_family=('face_swap' if index % 4 == 1 else 'text_inpaint') if label else '', label=label, attack_logit=logit,
            query_logit=logit, correction_logit=0., attack_probability=probability,
            bonafide_score=1. - probability, context_available=True, referral_required=False))
    return rows


@pytest.mark.parametrize('defect', ['count', 'order', 'label', 'nan', 'polarity', 'decomposition', 'referral'])
def test_exact_query_prediction_validation(defect):
    module = launcher()
    actual = predictions()
    expected = pd.DataFrame(copy.deepcopy(actual))
    if defect == 'count':
        actual.pop()
    elif defect == 'order':
        actual.reverse()
    elif defect == 'label':
        actual[0]['label'] = True
    elif defect == 'nan':
        actual[0]['query_logit'] = float('nan')
    elif defect == 'polarity':
        actual[0]['bonafide_score'] = actual[0]['attack_probability']
    elif defect == 'decomposition':
        actual[0]['correction_logit'] = 1.
    else:
        actual[0]['referral_required'] = True
    with pytest.raises(ValueError):
        module.validate_predictions(actual, expected)


@pytest.mark.parametrize('defect', [None, 'source_changed', 'wrong_membership', 'input_overlap', 'wrong_summary'])
def test_matched_scoring_without_a_test_or_target_learning(ready, monkeypatch, defect):
    module = ready['module']
    monkeypatch.setattr(module.driver, 'require_gpu_allocation', lambda: None)
    monkeypatch.setattr(module.driver.pilots, 'configure_precision', lambda *args: None)
    expected = predictions()
    rows = pd.DataFrame(expected)[['image_id', 'design_id', 'attack_family', 'label']]
    contract = SimpleNamespace(records_sha256=module.driver.RECORDS_SHA256,
        source_split_sha256=module.driver.SPLIT_SHA256,
        target_split_sha256=ready['request']['target_split']['sha256'])
    target = SimpleNamespace(query_ids=tuple(rows.image_id))
    monkeypatch.setattr(module.contract_api, 'load_protocol_c_contract', lambda *args, **kwargs: contract)
    monkeypatch.setattr(module.contract_api, 'load_target_enrollment', lambda *args, **kwargs: target)
    calls = []
    def context(*args, **kwargs):
        evaluation = module.driver.document(kwargs['evaluation_binding'])
        assert evaluation['authorized_image_ids'] == rows.image_id.tolist()
        assert evaluation['checkpoints'] == {arm: module.driver.document(
            ready['request']['source_selection']['arms'][arm]['train_result'])['checkpoint'] for arm in ('off', 'on')}
        calls.append('context')
        return SimpleNamespace(rows=lambda: rows.iloc[::-1] if defect == 'wrong_membership' else rows)
    monkeypatch.setattr(module.consumer, 'load_reference_context', context)
    def build(context, *, output_path):
        put(output_path, {'synthetic': True})
        calls.append('build')
        return SimpleNamespace(path=str(output_path))
    monkeypatch.setattr(module.consumer, 'build_protocol_c_registration', build)
    shared = object()
    monkeypatch.setattr(module.consumer, 'load_protocol_c_registration', lambda *args: shared)
    def dataset(context, table, *, reference_enabled):
        assert table is shared
        calls.append(('dataset', reference_enabled))
        return reference_enabled
    monkeypatch.setattr(module.consumer, 'ProtocolCReferenceDataset', dataset)
    monkeypatch.setattr(module, 'DataLoader', lambda data, **kwargs: data)
    def evaluate(model, loader, device):
        assert device == 'cuda'
        summary = module.summarize_predictions(expected)
        summary.update(image_loss=torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([row['attack_logit'] for row in expected]),
            torch.tensor([row['label'] for row in expected], dtype=torch.float32)).item(),
            buffers_unchanged=True, n_context_available=467, n_context_unavailable=0, referral_ids=[])
        if defect == 'wrong_summary':
            summary['accuracy'] = 0.
        if defect == 'source_changed':
            Path(ready['request']['source_selection']['config']['path']).write_text('changed after evaluation')
        return summary, copy.deepcopy(expected)
    monkeypatch.setattr(module, 'evaluate', evaluate, raising=False)
    monkeypatch.setattr(module.driver, 'fresh_model', lambda *args: object())
    if defect == 'input_overlap':
        ready['request']['output_root'] = str(ready['tmp'])
        ready['binding'] = put(Path(ready['binding']['path']), ready['request'])
    if defect is not None:
        with pytest.raises(ValueError):
            module.score(ready['binding'])
        assert not (Path(ready['request']['output_root']) / 'result.json').exists()
        return
    result = module.score(ready['binding'])
    assert result['n_images'] == 467 and result['protocol'] == 'C'
    assert result['learning_performed'] is False and result['threshold_tuning'] is False
    assert result['image_ids'] == rows.image_id.tolist()
    assert calls == ['context', 'build', ('dataset', False), ('dataset', True)]


def test_fixed_half_threshold_ties_are_genuine():
    module = launcher()
    actual = predictions()
    for row in actual:
        row.update(attack_logit=0., query_logit=0., correction_logit=0., attack_probability=.5, bonafide_score=.5)
    module.validate_predictions(actual, pd.DataFrame(actual))
    summary = module.summarize_predictions(actual)
    assert summary['genuine_false_positives'] == []
    assert len(summary['altered_misses']) == 233


@pytest.mark.parametrize('enabled', [False, True])
def test_c_inference_reuses_core_inputs_logits_loss_and_keeps_face_taxonomy(enabled):
    from test_direct_reference_pilot import SyntheticClassifier, samples
    from torch.utils.data import DataLoader
    module = launcher()
    data = samples(enabled)
    baseline_summary, baseline = module.driver.core.evaluate_direct_reference(
        SyntheticClassifier(enabled), DataLoader(data, batch_size=1), 'cpu')
    data[1]['meta']['attack_family'] = 'face_swap'
    summary, actual = module.evaluate(SyntheticClassifier(enabled), DataLoader(data, batch_size=1), 'cpu')
    baseline[1]['attack_family'] = 'face_swap'
    assert actual == baseline
    assert summary['accuracy'] == baseline_summary['accuracy']
    assert summary['auc'] == baseline_summary['auc']
    assert summary['image_loss'] == baseline_summary['image_loss']
    assert summary['subgroups']['face_swap']['n_altered'] == 1
    assert summary['subgroups']['text_inpaint']['n_altered'] == 2
    assert summary['referral_ids'] == ['1', '3']
    assert summary['buffers_unchanged'] is True
    with pytest.raises(ValueError, match='SIDTD'):
        module.driver.core.evaluate_direct_reference(SyntheticClassifier(enabled), DataLoader(data, batch_size=1), 'cpu')


def test_c_inference_buffer_mutation_is_rejected():
    from test_direct_reference_pilot import SyntheticClassifier, samples
    from torch.utils.data import DataLoader
    module = launcher()
    model = SyntheticClassifier(False)
    model.mutate_eval = True
    with pytest.raises(ValueError, match='buffers'):
        module.evaluate(model, DataLoader(samples(False), batch_size=1), 'cpu')
