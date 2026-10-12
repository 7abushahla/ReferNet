import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

from test_direct_reference_pilot import SyntheticClassifier
from refer_or_clear import direct_reference_pilot as core, pilots


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/score_protocol_a_current.py'


def launcher():
    assert SCRIPT.is_file(), 'Protocol A current scoring driver is missing'
    spec = importlib.util.spec_from_file_location('protocol_a_score_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_driver_exists_and_exposes_bounded_interface():
    module = launcher()
    for name in ('load_request', 'validate_arm', 'fresh_model', 'score', 'main'):
        assert callable(getattr(module, name))


def test_allocation_required_before_authority_or_pixels(monkeypatch):
    module = launcher()
    for name in ('SLURM_JOB_ID', 'SLURMD_NODENAME', 'SLURM_JOB_PARTITION', 'CUDA_VISIBLE_DEVICES'):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match='allocated'):
        module.main(['--request', '/never-open-request.json', '--request-sha256', 'a' * 64])


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return dict(path=str(path), sha256=pilots.file_sha256(path))


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return dict(path=str(path), sha256=pilots.file_sha256(path))


@pytest.fixture
def ready(tmp_path, monkeypatch):
    module = launcher()
    weights = tmp_path / 'synthetic-offline-initialization.bin'
    weights.write_bytes(b'no real model weights')
    weights_binding = module.bind(weights)
    monkeypatch.setattr(module, 'WEIGHTS_SHA256', weights_binding['sha256'])
    monkeypatch.setitem(pilots.OFFICIAL_WEIGHT_SHA256, 'D_running_pilot', weights_binding['sha256'])
    records, splits = [], []
    for role, count in (('train', 1778), ('val', 222), ('test', 222)):
        for index in range(count):
            identity = f'{role}-{index:04d}'
            label = index % 2
            record = dict(image_id=identity, path=f'{identity}.png', label=label, design_id='design',
                          attack_family='text_copy' if label and index % 4 == 1 else 'text_inpaint' if label else 'real',
                          component_id=f'identity-{identity}', dataset='sidtd')
            records.append(record)
            splits.append(dict(image_id=identity, label=label, design_id='design', component_id=record['component_id'],
                               role=role, protocol='A', ref_group='it0'))
    records_binding = table(tmp_path / 'records.csv', records)
    split_binding = table(tmp_path / 'split.csv', splits)
    monkeypatch.setattr(module, 'RECORDS_SHA256', records_binding['sha256'])
    monkeypatch.setattr(module, 'SPLIT_SHA256', split_binding['sha256'])
    enrollment = put(tmp_path / 'enrollment.json', dict(output_root=str(tmp_path / 'original-bank')))
    monkeypatch.setattr(module.consumer, 'ENROLLMENT_SHA256', enrollment['sha256'])
    root = put(tmp_path / 'root.json', dict(schema_version='direct_reference_product_root_v1',
               enrollment_manifest_sha256=enrollment['sha256'], original_root=str(tmp_path / 'original-bank'),
               relocated_root=str(tmp_path / 'relocated-bank')))
    config = yaml.safe_load((ROOT / 'configs/direct_reference_pilot.yaml').read_text())
    config['weights_sha256'] = weights_binding['sha256']
    config['enrollment_sha256'] = enrollment['sha256']
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    config_binding = module.bind(config_path)
    registration = put(tmp_path / 'training-registration.json', dict(schema_version='synthetic-table'))
    diagnostic = put(tmp_path / 'diagnostic.json', dict(passed=True))
    phase_authority = put(tmp_path / 'training-phase.json', dict(phase='train', source_sha256=module.TRAINING_SOURCE_SHA256))
    val_rows = module.load_role_rows(Path(records_binding['path']), Path(split_binding['path']), 'val')
    train_rows = module.load_role_rows(Path(records_binding['path']), Path(split_binding['path']), 'train')

    def model_builder(path, checksum, *, reference_enabled, head_seed, adapter_seed):
        assert checksum == weights_binding['sha256'] and head_seed == adapter_seed == 42
        model = SyntheticClassifier(reference_enabled)
        model.initialization_provenance['base']['checkpoint_path'] = str(path)
        return model

    monkeypatch.setattr(module, 'build_direct_reference_model', model_builder)
    arms = {}
    for arm in ('off', 'on'):
        enabled = arm == 'on'
        directory = tmp_path / 'training' / arm
        directory.mkdir(parents=True)
        model = model_builder(weights, weights_binding['sha256'], reference_enabled=enabled, head_seed=42, adapter_seed=42)
        initial = pilots.tensor_sha256(model.state_dict())
        model.initialization_provenance['base']['checkpoint_path'] = '/original-allocation/offline-weights.pth'
        effective = dict(config, arm=arm, reference_enabled=enabled)
        identities = {name: 'b' * 64 for name in core.IDENTITY_FIELDS}
        identities.update(source_sha256=module.TRAINING_SOURCE_SHA256, config_sha256=pilots.json_sha256(effective),
                          records_sha256=records_binding['sha256'], split_sha256=split_binding['sha256'],
                          enrollment_sha256=enrollment['sha256'], weights_sha256=weights_binding['sha256'],
                          registration_sha256=registration['sha256'],
                          phase_authority_sha256=phase_authority['sha256'],
                          initialization_sha256=pilots.json_sha256(model.initialization_provenance))
        with torch.no_grad():
            model.head.fill_(.125)
        predictions = []
        for row in val_rows.to_dict('records'):
            logit = 2.125 if row['label'] else -1.875
            probability = torch.tensor(logit).sigmoid().item()
            predictions.append(dict(image_id=row['image_id'], design_id=row['design_id'], attack_family=row['attack_family'],
                               label=row['label'], attack_logit=logit, query_logit=logit, correction_logit=0.,
                               attack_probability=probability, bonafide_score=1. - probability,
                               context_available=True, referral_required=False, availability_reason='fit'))
        summary = pilots.summarize_predictions(predictions)
        summary.update(image_loss=torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([row['attack_logit'] for row in predictions]),
            torch.tensor([row['label'] for row in predictions], dtype=torch.float32)).item(),
            buffers_unchanged=True, n_context_available=222, n_context_unavailable=0, referral_ids=[])
        key = (summary['accuracy'], summary['auc'])
        core.save_direct_reference_checkpoint(directory / 'best.pt', model, effective, identities, 2, key, 'train')
        checkpoint = module.bind(directory / 'best.pt')
        prediction_binding = table(directory / 'val_predictions.csv', predictions)
        history = table(directory / 'log.csv', [dict(epoch=epoch, n_images=1776, optimizer_steps=222,
                         val_accuracy=.5 if epoch == 1 else key[0], val_auc=.5 if epoch == 1 else key[1],
                         val_image_loss=.1, loss=.1) for epoch in range(1, 101)])
        original_authority = put(directory / 'authority.json', dict(source_sha256=module.TRAINING_SOURCE_SHA256,
                                 schema_version='direct_reference_control_request_v1' if arm == 'off' else 'direct_reference_on_study_request_v1',
                                 arm=arm, epochs=100, fold=0, seed=42, config_sha256=config_binding['sha256'],
                                 diagnostic_request_sha256='e' * 64, diagnostic_gate=diagnostic, registration=registration,
                                 phase_authority=phase_authority, off_completion=arms.get('off', {}).get('completion'),
                                 launcher_sha256='d' * 64))
        result = dict(passed=True, phase='train', diagnostic_only=False, epochs_completed=100, optimizer_steps=22200,
                      role='val', arm=arm, selected_epoch=2, selection_key=pilots._selection_record(key),
                      scheduler=dict(total_updates=22200, warmup_updates=1000), identities=identities,
                      fresh_initial_state_sha256=initial, checkpoint=checkpoint, predictions=prediction_binding,
                      registration=registration, diagnostic_gate=diagnostic,
                      replay=dict(passed=True, fresh_process=False, n_images=222, phase='train'),
                      omitted_train_ids=[dict(epoch=epoch, image_ids=train_rows.image_id.tolist()[:2]) for epoch in range(1, 101)])
        train_binding = put(directory / 'train.json', result)
        authority_key = 'control_request' if arm == 'off' else 'on_request'
        comparison = pilots.compare_replay(predictions, predictions)
        replay = dict(schema_version='direct_reference_control_replay_v1' if arm == 'off' else 'direct_reference_on_study_replay_v1',
                      passed=True, fresh_process=True, phase='train', arm=arm, n_images=222, buffers_unchanged=True,
                      pid=99999, train_result=train_binding, checkpoint=checkpoint, identities=identities,
                      history=history, image_ids=val_rows.image_id.tolist(), predictions=predictions, summary=summary,
                      replay=comparison, max_abs_difference=comparison['max_abs_difference'], launcher_sha256='d' * 64,
                      **{authority_key: original_authority})
        replay_binding = put(directory / 'fresh-replay.json', replay)
        completion = dict(schema_version='direct_reference_control_result_v1' if arm == 'off' else 'direct_reference_on_study_result_v1',
                          passed=True, phase='train', arm=arm, epochs_completed=100, train_result=train_binding,
                          fresh_replay=replay_binding, checkpoint=checkpoint, identities=identities,
                          scientific_source_sha256=module.TRAINING_SOURCE_SHA256, launcher_sha256='d' * 64,
                          **{authority_key: original_authority})
        arms[arm] = dict(train_result=train_binding, fresh_replay=replay_binding,
                         completion=put(directory / 'completion.json', completion))
    inventory = module.source_entries(ROOT)
    request = dict(schema_version='protocol_a_current_score_v1', protocol='A', fold=0, seed=42, eval_batch_size=1,
                   training_source_sha256=module.TRAINING_SOURCE_SHA256, evaluation_source_sha256=module.object_sha256(inventory),
                   critical_source_hashes={entry['path']: entry['sha256'] for entry in inventory
                                           if entry['path'] in module.CRITICAL_SOURCE_PATHS},
                   records=records_binding, split=split_binding, enrollment=enrollment, root=root,
                   raw_root=str(tmp_path / 'never-decode-raw'), weights=weights_binding, config=config_binding,
                   output_root=str(tmp_path / 'scored-test'), arms=arms)
    request_binding = put(tmp_path / 'score-request.json', request)
    return dict(module=module, request=request, binding=request_binding, config=config,
                val_rows=val_rows, train_rows=train_rows, tmp=tmp_path)


def mutate(ready, arm, artifact, change):
    binding = ready['request']['arms'][arm][artifact]
    value = ready['module'].document(binding)
    change(value)
    ready['request']['arms'][arm][artifact] = put(Path(binding['path']), value)
    ready['binding'] = put(Path(ready['binding']['path']), ready['request'])


def selected(ready, arm='off'):
    return ready['module'].validate_arm(ready['request'], ready['config'], arm, ready['val_rows'], ready['train_rows'])


@pytest.mark.parametrize('arm', ['off', 'on'])
def test_exact_completed_artifacts_accepted(ready, arm):
    module = ready['module']
    request, config = module.load_request(ready['binding'])
    assert request == ready['request'] and config == ready['config']
    actual = selected(ready, arm)
    assert actual['result']['selected_epoch'] == 2
    assert actual['identities']['source_sha256'] == module.TRAINING_SOURCE_SHA256
    assert actual['identities']['source_sha256'] != request['evaluation_source_sha256']


@pytest.mark.parametrize('field,value', [('epochs_completed', 99), ('optimizer_steps', 22199), ('passed', False),
                                       ('diagnostic_only', True), ('role', 'test'), ('epochs_completed', True)])
def test_incomplete_train_rejected_before_checkpoint_or_model(ready, monkeypatch, field, value):
    mutate(ready, 'on', 'train_result', lambda result: result.update({field: value}))
    monkeypatch.setattr(core, 'read_direct_reference_checkpoint', lambda *args: pytest.fail('checkpoint read before completion'))
    monkeypatch.setattr(ready['module'], 'build_direct_reference_model', lambda *args, **kwargs: pytest.fail('model before completion'))
    with pytest.raises(ValueError, match='complete OFF100/ON100'):
        selected(ready, 'on')


@pytest.mark.parametrize('field,value', [('fresh_process', False), ('n_images', 221), ('buffers_unchanged', False),
                                       ('arm', 'on'), ('pid', 0)])
def test_fresh_replay_contract_rejected(ready, field, value):
    mutate(ready, 'off', 'fresh_replay', lambda replay: replay.update({field: value}))
    completion = ready['request']['arms']['off']['completion']
    body = ready['module'].document(completion)
    body['fresh_replay'] = ready['request']['arms']['off']['fresh_replay']
    ready['request']['arms']['off']['completion'] = put(Path(completion['path']), body)
    with pytest.raises(ValueError, match='fresh replay'):
        selected(ready)


@pytest.mark.parametrize('field', ['image_id', 'label', 'attack_family', 'design_id'])
def test_hash_consistent_official_validation_provenance_rejected(ready, field):
    arm = ready['request']['arms']['off']
    result = ready['module'].document(arm['train_result'])
    predictions = ready['module'].csv_predictions(result['predictions'])
    predictions[0][field] = 1 if field == 'label' else 'forged-provenance'
    result['predictions'] = table(Path(result['predictions']['path']), predictions)
    arm['train_result'] = put(Path(arm['train_result']['path']), result)
    completion = ready['module'].document(arm['completion'])
    completion['train_result'] = arm['train_result']
    arm['completion'] = put(Path(arm['completion']['path']), completion)
    with pytest.raises(ValueError, match='official prediction metadata'):
        selected(ready)


def test_history_earliest_tie_not_loss_selection(ready):
    arm = ready['request']['arms']['off']
    replay = ready['module'].document(arm['fresh_replay'])
    rows = list(csv.DictReader(Path(replay['history']['path']).open()))
    rows[0]['val_accuracy'] = rows[1]['val_accuracy']
    rows[0]['val_auc'] = rows[1]['val_auc']
    rows[0]['val_image_loss'] = '100'
    replay['history'] = table(Path(replay['history']['path']), rows)
    arm['fresh_replay'] = put(Path(arm['fresh_replay']['path']), replay)
    completion = ready['module'].document(arm['completion'])
    completion['fresh_replay'] = arm['fresh_replay']
    arm['completion'] = put(Path(arm['completion']['path']), completion)
    with pytest.raises(ValueError, match='earliest accuracy/AUC'):
        selected(ready)


@pytest.mark.parametrize('mutation', ['extra', 'source', 'critical', 'weights', 'output'])
def test_strict_authority_errors_before_model_or_test(ready, mutation):
    request = ready['request']
    if mutation == 'extra':
        request['override_epochs'] = 1
    elif mutation == 'source':
        request['evaluation_source_sha256'] = '0' * 64
    elif mutation == 'critical':
        request['critical_source_hashes'][next(iter(request['critical_source_hashes']))] = '0' * 64
    elif mutation == 'weights':
        Path(request['weights']['path']).write_bytes(b'altered weights')
    else:
        request['output_root'] = request['raw_root']
    ready['binding'] = put(Path(ready['binding']['path']), request)
    with pytest.raises(ValueError):
        ready['module'].load_request(ready['binding'])


@pytest.mark.parametrize('arm', ['off', 'on'])
def test_relocation_only_provenance_and_strict_selected_load(ready, arm):
    module = ready['module']
    chosen = selected(ready, arm)
    model = module.fresh_model(ready['request'], chosen, arm, device='cpu')
    assert model.initialization_provenance == chosen['provenance']
    assert model.initialization_provenance['base']['checkpoint_path'] == '/original-allocation/offline-weights.pth'
    assert model.head.item() == .125
    core._validate_model(model, chosen['config'], chosen['identities'])
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, chosen['checkpoint']['state_dict'][name], rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['adapter_seed', 'base_hash', 'initial_state'])
def test_provenance_cannot_be_reassigned_before_equality(ready, monkeypatch, kind):
    module = ready['module']
    chosen = selected(ready)
    builder = module.build_direct_reference_model
    made = []

    def changed(*args, **kwargs):
        model = builder(*args, **kwargs)
        if kind == 'adapter_seed':
            model.initialization_provenance['adapter_seed'] = 43
        elif kind == 'base_hash':
            model.initialization_provenance['base']['pretrained_tensor_sha256'] = 'e' * 64
        else:
            with torch.no_grad():
                model.head.fill_(3)
        made.append(model)
        return model

    monkeypatch.setattr(module, 'build_direct_reference_model', changed)
    with pytest.raises(ValueError, match='initialization provenance|initial tensor'):
        module.fresh_model(ready['request'], chosen, 'off', device='cpu')
    assert made[0].initialization_provenance['base']['checkpoint_path'] == ready['request']['weights']['path']


def test_missing_on_completion_never_authorizes_test(ready, monkeypatch):
    module = ready['module']
    monkeypatch.setattr(module, 'require_gpu_allocation', lambda: None)
    mutate(ready, 'on', 'completion', lambda result: result.update(passed=False))
    for name in ('load_evaluation_context', 'build_evaluation_registration', 'load_evaluation_registration', 'ProtocolAEvaluationDataset'):
        monkeypatch.setattr(module.consumer, name, lambda *args, **kwargs: pytest.fail('test consumer accessed before completion'))
    monkeypatch.setattr(module, 'fresh_model', lambda *args, **kwargs: pytest.fail('fresh model before both completion'))
    with pytest.raises(ValueError, match='completed arm receipt'):
        module.score(ready['binding'])
    assert not Path(ready['request']['output_root']).exists()


def test_scoring_uses_all_four_consumer_apis_and_current_core_without_learning(ready, monkeypatch):
    module = ready['module']
    calls = []
    monkeypatch.setattr(module, 'require_gpu_allocation', lambda: None)
    monkeypatch.setattr(pilots, 'configure_precision', lambda *args: calls.append('precision'))
    original_fresh = module.fresh_model
    monkeypatch.setattr(module, 'fresh_model', lambda request, chosen, arm: original_fresh(request, chosen, arm, device='cpu'))
    joined, _ = module.consumer._membership(module.content(ready['request']['records']), module.content(ready['request']['split']))
    rows = joined.loc[joined.role == 'test'].sort_values('image_id').reset_index(drop=True)

    def context(*args, evaluation_binding):
        authority = module.document(evaluation_binding)
        assert authority['test_image_ids'] == rows.image_id.tolist()
        assert authority['checkpoints'] == {arm: module.document(ready['request']['arms'][arm]['train_result'])['checkpoint']
                                            for arm in ('off', 'on')}
        calls.append('context')
        return rows

    def registration(context_rows, path):
        assert context_rows is rows
        calls.append('build-table')
        put(path, dict(image_ids=rows.image_id.tolist()))
        return type('Table', (), {'path': str(path)})()

    def load(context_rows, binding):
        assert context_rows is rows
        assert module.document(binding)['image_ids'] == rows.image_id.tolist()
        calls.append('load-table')
        return 'authenticated-table'

    def dataset(context_rows, table, *, reference_enabled):
        assert context_rows is rows and table == 'authenticated-table'
        calls.append('on-data' if reference_enabled else 'off-data')
        values = []
        for index, row in enumerate(rows.to_dict('records')):
            query = torch.zeros(4, 32, 32)
            query[0].fill_(2. if row['label'] else -2.)
            available = index != 0
            sample = dict(query=query, overlap=torch.ones(1, 32, 32) if available else torch.zeros(1, 32, 32),
                          context_available=torch.tensor(available), label=row['label'],
                          meta={key: row[key] for key in ('image_id', 'design_id', 'attack_family')})
            sample['meta']['availability_reason'] = 'fit' if available else 'registration_failed'
            if reference_enabled:
                sample['reference'] = torch.zeros(3, 32, 32)
            values.append(sample)
        return values

    actual_evaluate = core.evaluate_direct_reference

    def evaluate(model, loader, device):
        assert device == 'cuda' and loader.batch_size == 1 and loader.drop_last is False and loader.num_workers == 0
        assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
        assert all(parameter.grad is None for parameter in model.parameters())
        before = {name: value.clone() for name, value in model.state_dict().items()}
        result = actual_evaluate(model, loader, 'cpu')
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        calls.append('evaluate')
        return result

    monkeypatch.setattr(module.consumer, 'load_evaluation_context', context)
    monkeypatch.setattr(module.consumer, 'build_evaluation_registration', registration)
    monkeypatch.setattr(module.consumer, 'load_evaluation_registration', load)
    monkeypatch.setattr(module.consumer, 'ProtocolAEvaluationDataset', dataset)
    monkeypatch.setattr(core, 'evaluate_direct_reference', evaluate)
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *args, **kwargs: pytest.fail('optimizer created'))
    result = module.score(ready['binding'])
    assert calls == ['precision', 'context', 'build-table', 'load-table', 'off-data', 'evaluate', 'on-data', 'evaluate']
    assert result['role'] == 'test' and result['fold'] == 0 and result['aggregate'] is False
    assert result['n_images'] == 222 and result['image_ids'] == rows.image_id.tolist()
    assert result['learning_performed'] is False and result['threshold_tuning_performed'] is False
    for arm in ('off', 'on'):
        report = result['arms'][arm]
        assert report['training_identities']['source_sha256'] == module.TRAINING_SOURCE_SHA256
        summary = module.document(report['summary'])
        assert summary['role'] == 'test' and summary['n_images'] == 222 and summary['selected_epoch'] == 2
        assert summary['n_context_unavailable'] == 1 and summary['referral_ids'] == [rows.image_id.iloc[0]]
        predictions = module.csv_predictions(report['predictions'])
        module.validate_predictions(predictions, rows)
    assert module.document(module.bind(Path(ready['request']['output_root']) / 'result.json')) == result


def refresh_replay(ready, arm, replay):
    authorities = ready['request']['arms'][arm]
    authorities['fresh_replay'] = put(Path(authorities['fresh_replay']['path']), replay)
    completion = ready['module'].document(authorities['completion'])
    completion['fresh_replay'] = authorities['fresh_replay']
    authorities['completion'] = put(Path(authorities['completion']['path']), completion)


@pytest.mark.parametrize('mutation', ['nonfinite', 'score', 'summary_auc', 'summary_loss', 'availability', 'history_count', 'history_steps'])
def test_hash_consistent_replay_evidence_tampering_rejected(ready, mutation):
    module = ready['module']
    replay = module.document(ready['request']['arms']['off']['fresh_replay'])
    if mutation == 'nonfinite':
        replay['predictions'][0]['attack_logit'] = 'nan'
    elif mutation == 'score':
        replay['predictions'][0]['attack_probability'] = .5
    elif mutation == 'summary_auc':
        replay['summary']['auc'] = .8
    elif mutation == 'summary_loss':
        replay['summary']['image_loss'] += .1
    elif mutation == 'availability':
        replay['predictions'][0].update(context_available=False, referral_required=True)
    else:
        with Path(replay['history']['path']).open() as stream:
            history = list(csv.DictReader(stream))
        if mutation == 'history_count':
            history.pop()
        else:
            history[60]['optimizer_steps'] = '221'
        replay['history'] = table(Path(replay['history']['path']), history)
    refresh_replay(ready, 'off', replay)
    with pytest.raises(ValueError):
        selected(ready)


def test_replay_image_loss_rounding_tolerance_only(ready):
    module = ready['module']
    replay = module.document(ready['request']['arms']['off']['fresh_replay'])
    replay['summary']['image_loss'] += pilots.ATOL / 2
    refresh_replay(ready, 'off', replay)
    assert selected(ready)['result']['selected_epoch'] == 2


@pytest.mark.parametrize('field,value', [('phase', 'smoke'), ('epoch', 3), ('selected', False),
                                       ('arm', 'on'), ('architecture_id', 'obsolete-map-head')])
def test_hash_consistent_checkpoint_header_tampering_rejected(ready, field, value):
    module = ready['module']
    authorities = ready['request']['arms']['off']
    result = module.document(authorities['train_result'])
    checkpoint = torch.load(result['checkpoint']['path'], weights_only=True)
    checkpoint[field] = value
    torch.save(checkpoint, result['checkpoint']['path'])
    result['checkpoint'] = module.bind(Path(result['checkpoint']['path']))
    authorities['train_result'] = put(Path(authorities['train_result']['path']), result)
    replay = module.document(authorities['fresh_replay'])
    replay.update(checkpoint=result['checkpoint'], train_result=authorities['train_result'])
    refresh_replay(ready, 'off', replay)
    completion = module.document(authorities['completion'])
    completion.update(checkpoint=result['checkpoint'], train_result=authorities['train_result'])
    authorities['completion'] = put(Path(authorities['completion']['path']), completion)
    with pytest.raises(ValueError, match='checkpoint|architecture'):
        selected(ready)


def test_checkpoint_bytes_changed_rejected_before_deserialization(ready, monkeypatch):
    module = ready['module']
    result = module.document(ready['request']['arms']['off']['train_result'])
    Path(result['checkpoint']['path']).write_bytes(b'not the externally pinned checkpoint')
    monkeypatch.setattr(core, 'read_direct_reference_checkpoint', lambda *args: pytest.fail('untrusted checkpoint loaded'))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        selected(ready)


@pytest.mark.parametrize('field', ['adapter_tensor_sha256', 'reference_enabled'])
def test_all_nonlocation_provenance_fields_protected(ready, monkeypatch, field):
    module = ready['module']
    chosen = selected(ready)
    builder = module.build_direct_reference_model

    def changed(*args, **kwargs):
        model = builder(*args, **kwargs)
        model.initialization_provenance[field] = 'f' * 64 if field == 'adapter_tensor_sha256' else True
        return model

    monkeypatch.setattr(module, 'build_direct_reference_model', changed)
    with pytest.raises(ValueError, match='scientific initialization provenance'):
        module.fresh_model(ready['request'], chosen, 'off', device='cpu')


def test_relocated_weights_bytes_rechecked_before_construction(ready, monkeypatch):
    module = ready['module']
    chosen = selected(ready)
    Path(ready['request']['weights']['path']).write_bytes(b'changed after authority validation')
    monkeypatch.setattr(module, 'build_direct_reference_model', lambda *args, **kwargs: pytest.fail('unverified weights loaded'))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        module.fresh_model(ready['request'], chosen, 'off', device='cpu')


@pytest.mark.parametrize('field,value', [('fold', True), ('eval_batch_size', 8), ('seed', 43),
                                       ('training_source_sha256', '0' * 64), ('protocol', 'B')])
def test_fixed_request_scientific_settings_not_overridable(ready, field, value):
    ready['request'][field] = value
    binding = put(Path(ready['binding']['path']), ready['request'])
    with pytest.raises(ValueError, match='fixed Protocol A'):
        ready['module'].load_request(binding)


def test_duplicate_authority_json_keys_rejected(tmp_path):
    module = launcher()
    path = tmp_path / 'duplicate.json'
    path.write_text('{"fold":0,"fold":1}')
    with pytest.raises(ValueError, match='duplicate JSON key'):
        module.document(module.bind(path))
