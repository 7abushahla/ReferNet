"""Score only the official Protocol A fold-zero test after both completed arms.

The externally pinned protocol_a_current_score_v1 request has fields
schema_version, protocol, fold, seed, eval_batch_size, evaluation_source_sha256,
training_source_sha256, critical_source_hashes, records, split, enrollment, root,
raw_root, weights, config, output_root, and arms. Each arm binds train_result,
completion, and fresh_replay. All file bindings are absolute path/SHA256 pairs.
critical_source_hashes maps original-training source paths to SHA256 values,
attested by the external request issuer against the original training tree.
Embedded training artifact paths must be restored unchanged by staging.
The root binding separately authorizes enrollment-product relocation.
No training authorization service, fitting of score thresholds, or learning is
performed here. Outputs are fold-zero TEST measurements, not aggregate results.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import torch
from torch.utils.data import DataLoader
import yaml

from anchor_job_manifest import source_entries, object_sha256
from refer_or_clear import direct_reference_pilot as core, pilots
from refer_or_clear.data import protocol_a_evaluation as consumer
from refer_or_clear.data.query import load_role_rows
from refer_or_clear.models.direct_reference import build_direct_reference_model

TRAINING_SOURCE_SHA256 = 'a5ff28ae2a332b977551241f2d07641d51df78d54a4cc1b58169c7338ce11292'
WEIGHTS_SHA256 = 'da3f569306dadd1fac5b64abc2a3d484571cd0ecf6410e1f152275cfe49a3618'
RECORDS_SHA256 = 'afd661819fe9be97c83d2ee5123a7ce60869437c61d1e9029c4b8a96175aa276'
SPLIT_SHA256 = '985d10b47461e0fa41d5502fbb51cd6fc03f8c7e8ec8cd1040b21be318efc1d9'
CRITICAL_SOURCE_PATHS = frozenset({
    'src/refer_or_clear/models/direct_reference.py',
    'src/refer_or_clear/models/anchor_changer.py',
    'src/refer_or_clear/models/changerex.py',
    'src/refer_or_clear/anchor_contract.py',
    'src/refer_or_clear/direct_reference_pilot.py',
    'src/refer_or_clear/pilots.py',
    'src/refer_or_clear/train.py',
    'src/refer_or_clear/data/direct_reference.py',
    'src/refer_or_clear/data/dataset.py',
})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute(value):
    require(isinstance(value, str), 'absolute path required')
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts
            and not any(parent.is_symlink() for parent in (path, *path.parents)),
            'absolute non-symlink path required')
    return path


def digest(path):
    return pilots.file_sha256(path)


def bind(path):
    return dict(path=str(path), sha256=digest(path))


def content(binding):
    require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}
            and isinstance(binding['sha256'], str)
            and re.fullmatch('[0-9a-f]{64}', binding['sha256']), 'path/SHA256 binding required')
    path = absolute(binding['path'])
    require(digest(path) == binding['sha256'], 'artifact SHA256 mismatch')
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == binding['sha256'], 'artifact changed during read')
    return payload


def document(binding):
    def pairs(entries):
        value = {}
        for key, item in entries:
            require(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    value = json.loads(content(binding), object_pairs_hook=pairs)
    json.dumps(value, allow_nan=False)
    require(isinstance(value, dict), 'JSON object required')
    return value


def authenticate(binding):
    require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}
            and isinstance(binding['sha256'], str)
            and re.fullmatch('[0-9a-f]{64}', binding['sha256']), 'path/SHA256 binding required')
    require(digest(absolute(binding['path'])) == binding['sha256'], 'artifact SHA256 mismatch')


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return bind(path)


def require_gpu_allocation():
    node = os.environ.get('SLURMD_NODENAME', '').lower()
    job = os.environ.get('SLURM_JOB_ID', '')
    require(job.isdecimal() and int(job) > 0 and node
            and not any(token in node for token in ('login', 'head', 'frontend', 'lgn'))
            and os.environ.get('SLURM_JOB_PARTITION') == 'gpu'
            and os.environ.get('CUDA_VISIBLE_DEVICES', '') not in ('', '-1', 'NoDevFiles'),
            'allocated GPU node required')
    require(torch.cuda.is_available(), 'allocated CUDA device unavailable')


def validate_source(request):
    entries = source_entries(ROOT)
    require(request['evaluation_source_sha256'] == object_sha256(entries), 'evaluation source inventory mismatch')
    expected = request['critical_source_hashes']
    require(isinstance(expected, dict) and CRITICAL_SOURCE_PATHS <= set(expected), 'critical original-training hashes required')
    inventory = {entry['path']: entry['sha256'] for entry in entries}
    for path, checksum in expected.items():
        require(path.startswith('src/') and path.endswith('.py') and '..' not in Path(path).parts
                and isinstance(checksum, str) and re.fullmatch('[0-9a-f]{64}', checksum)
                and inventory.get(path) == checksum, 'critical scientific source differs from original training')
    for name, module in tuple(sys.modules.items()):
        if name == 'refer_or_clear' or name.startswith('refer_or_clear.'):
            location = getattr(module, '__file__', None)
            require(location is not None and Path(location).resolve().is_relative_to(ROOT / 'src'),
                    'foreign scientific module imported')


def load_request(binding):
    request = document(binding)
    fixed = dict(schema_version='protocol_a_current_score_v1', protocol='A', fold=0, seed=42,
                 eval_batch_size=1, training_source_sha256=TRAINING_SOURCE_SHA256)
    fields = {'evaluation_source_sha256', 'critical_source_hashes', 'records', 'split', 'enrollment',
              'root', 'raw_root', 'weights', 'config', 'output_root', 'arms'}
    require(set(request) == set(fixed) | fields, 'strict scoring request fields required')
    require(all(type(request[key]) is type(value) and request[key] == value for key, value in fixed.items()),
            'fixed Protocol A scoring contract mismatch')
    validate_source(request)
    for name, expected in (('records', RECORDS_SHA256), ('split', SPLIT_SHA256),
                           ('enrollment', consumer.ENROLLMENT_SHA256), ('weights', WEIGHTS_SHA256)):
        require(isinstance(request[name], dict) and request[name].get('sha256') == expected,
                f'official {name} identity required')
        authenticate(request[name])
    document(request['root'])
    config = yaml.safe_load(content(request['config']))
    require(isinstance(config, dict), 'original config object required')
    fixed_config = dict(epochs=100, seed=42, head_seed=42, adapter_seed=42, fold=0,
                        batch_size=8, eval_batch_size=1, accumulation_steps=1, drop_last=True,
                        eval_drop_last=False, input_shape=[4, 512, 800], mask_weight=0.0)
    require(all(type(config.get(key)) is type(value) and config[key] == value for key, value in fixed_config.items()),
            'original full-training recipe required')
    require(config.get('selection') == dict(primary='accuracy', tie_break='auc', exact_tie='earlier_epoch'),
            'accuracy/AUC/earlier selection required')
    require(isinstance(request['arms'], dict) and set(request['arms']) == {'off', 'on'}, 'both completed arms required')
    protected = [ROOT, absolute(binding['path']), absolute(request['raw_root'])]
    relocation = document(request['root'])
    protected += [absolute(relocation[name]) for name in ('original_root', 'relocated_root')]
    for name in ('records', 'split', 'enrollment', 'root', 'weights', 'config'):
        protected.append(absolute(request[name]['path']))
    for arm in ('off', 'on'):
        authorities = request['arms'][arm]
        require(isinstance(authorities, dict) and set(authorities) == {'train_result', 'completion', 'fresh_replay'},
                'externally bound training/completion/fresh replay required')
        for authority in authorities.values():
            document(authority)
            protected.append(absolute(authority['path']))
    output = absolute(request['output_root'])
    require(not output.exists(), 'new isolated scoring output required')
    require(all(not output.is_relative_to(path) and not path.is_relative_to(output) for path in protected),
            'output overlaps immutable input')
    return request, config


def number(value):
    require(type(value) is not bool, 'finite numeric field required')
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError('finite numeric field required') from None
    require(math.isfinite(value), 'finite numeric field required')
    return value


def validate_predictions(predictions, rows):
    require(isinstance(predictions, list) and len(predictions) == len(rows) == 222,
            'exact official 222-image population required')
    for actual, expected in zip(predictions, rows.to_dict('records')):
        require(all(type(actual.get(key)) is type(expected[key]) and actual[key] == expected[key]
                    for key in ('image_id', 'design_id', 'attack_family', 'label')), 'official prediction metadata/order mismatch')
        for key in ('attack_logit', 'query_logit', 'correction_logit', 'attack_probability', 'bonafide_score'):
            number(actual.get(key))
        require(type(actual.get('context_available')) is bool and type(actual.get('referral_required')) is bool
                and actual['referral_required'] is not actual['context_available'], 'availability/referral mismatch')
        probability = torch.tensor(actual['attack_logit'], dtype=torch.float32).sigmoid().item()
        require(abs(actual['attack_probability'] - probability) <= 1e-7
                and abs(actual['bonafide_score'] - (1. - probability)) <= 1e-7,
                'prediction score/logit mismatch')
        require(math.isclose(actual['attack_logit'], actual['query_logit'] + actual['correction_logit'],
                             abs_tol=pilots.ATOL, rel_tol=pilots.RTOL), 'prediction logit decomposition mismatch')


def csv_predictions(binding):
    rows = list(csv.DictReader(StringIO(content(binding).decode())))
    for row in rows:
        require(row.get('label') in ('0', '1'), 'binary prediction label required')
        row['label'] = int(row['label'])
        for name in ('attack_logit', 'query_logit', 'correction_logit', 'attack_probability', 'bonafide_score'):
            row[name] = number(row.get(name))
        for name in ('context_available', 'referral_required'):
            require(row.get(name) in ('True', 'False'), 'boolean prediction field required')
            row[name] = row[name] == 'True'
    return rows


def validate_summary(summary, predictions):
    computed = pilots.summarize_predictions(predictions)
    computed.update(buffers_unchanged=True,
                    n_context_available=sum(row['context_available'] for row in predictions),
                    n_context_unavailable=sum(not row['context_available'] for row in predictions),
                    referral_ids=[row['image_id'] for row in predictions if row['referral_required']])
    require(all(type(summary.get(key)) is type(value) and summary[key] == value for key, value in computed.items()),
            'recomputed summary/buffers mismatch')
    require(number(summary.get('image_loss')) >= 0, 'nonnegative image loss required')
    labels = torch.tensor([row['label'] for row in predictions], dtype=torch.float32)
    logits = torch.tensor([row['attack_logit'] for row in predictions], dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    require(math.isclose(summary['image_loss'], loss, abs_tol=pilots.ATOL, rel_tol=pilots.RTOL),
            'recomputed image loss mismatch')


def validate_arm(request, config, arm, val_rows, train_rows):
    authorities = request['arms'][arm]
    result = document(authorities['train_result'])
    completion = document(authorities['completion'])
    replay = document(authorities['fresh_replay'])
    fixed = dict(passed=True, phase='train', diagnostic_only=False, epochs_completed=100,
                 optimizer_steps=22200, role='val', arm=arm)
    require(all(type(result.get(key)) is type(value) and result[key] == value for key, value in fixed.items()),
            'complete OFF100/ON100 training required before test access')
    require(result.get('scheduler') == dict(total_updates=22200, warmup_updates=1000), 'full scheduler accounting required')
    schema = 'direct_reference_control_result_v1' if arm == 'off' else 'direct_reference_on_study_result_v1'
    for key, value in dict(schema_version=schema, passed=True, arm=arm, epochs_completed=100,
                           train_result=authorities['train_result'], fresh_replay=authorities['fresh_replay'],
                           checkpoint=result.get('checkpoint'), identities=result.get('identities')).items():
        require(type(completion.get(key)) is type(value) and completion[key] == value, 'completed arm receipt mismatch')
    require(arm != 'off' or completion.get('phase') == 'train', 'OFF completion phase mismatch')
    require(arm != 'on' or completion.get('scientific_source_sha256') == TRAINING_SOURCE_SHA256,
            'ON original scientific source mismatch')
    identities = result.get('identities', {})
    effective = dict(config, arm=arm, reference_enabled=arm == 'on')
    expected = dict(source_sha256=TRAINING_SOURCE_SHA256, config_sha256=pilots.json_sha256(effective),
                    weights_sha256=WEIGHTS_SHA256, records_sha256=request['records']['sha256'],
                    split_sha256=request['split']['sha256'], enrollment_sha256=request['enrollment']['sha256'],
                    registration_sha256=result['registration']['sha256'])
    require(set(identities) == set(core.IDENTITY_FIELDS)
            and all(identities.get(key) == value for key, value in expected.items()), 'original training identities mismatch')
    core._validate_contract(effective, identities)
    authenticate(result['registration'])
    authenticate(result['checkpoint'])
    predictions = csv_predictions(result['predictions'])
    validate_predictions(predictions, val_rows)
    summary = pilots.summarize_predictions(predictions)
    history_binding = replay.get('history')
    history = list(csv.DictReader(StringIO(content(history_binding).decode())))
    require(len(history) == 100, '100 ordered history rows required')
    best_key, best_epoch = None, None
    for epoch, row in enumerate(history, 1):
        require(row.get('epoch') == str(epoch) and row.get('n_images') == '1776'
                and row.get('optimizer_steps') == '222', 'full ordered epoch accounting required')
        key = (number(row.get('val_accuracy')), number(row.get('val_auc')))
        require(all(0 <= value <= 1 for value in key), 'bounded selection metrics required')
        require(number(row.get('val_image_loss')) >= 0 and number(row.get('loss')) >= 0, 'valid history losses required')
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
    require(type(result.get('selected_epoch')) is int and result['selected_epoch'] == best_epoch
            and result.get('selection_key') == pilots._selection_record(best_key)
            and (summary['accuracy'], summary['auc']) == best_key, 'earliest accuracy/AUC checkpoint selection mismatch')
    omissions = result.get('omitted_train_ids')
    require(isinstance(omissions, list) and len(omissions) == 100, 'full drop-last omission history required')
    train_ids = set(train_rows.image_id)
    for epoch, row in enumerate(omissions, 1):
        ids = row.get('image_ids', [])
        require(type(row.get('epoch')) is int and row['epoch'] == epoch and isinstance(ids, list)
                and len(ids) == len(set(ids)) == 2 and set(ids) <= train_ids, 'official drop-last omissions mismatch')
    fixed_replay = dict(schema_version='direct_reference_control_replay_v1' if arm == 'off' else 'direct_reference_on_study_replay_v1',
                        passed=True, fresh_process=True, phase='train', arm=arm, n_images=222,
                        buffers_unchanged=True, train_result=authorities['train_result'], checkpoint=result['checkpoint'],
                        identities=identities, image_ids=val_rows.image_id.tolist())
    require(all(type(replay.get(key)) is type(value) and replay[key] == value for key, value in fixed_replay.items()),
            'full fresh replay binding required')
    require(type(replay.get('pid')) is int and replay['pid'] > 0, 'fresh replay process identity required')
    authority_key = 'control_request' if arm == 'off' else 'on_request'
    require(completion.get(authority_key) == replay.get(authority_key) and isinstance(replay.get(authority_key), dict),
            'completion/replay authority mismatch')
    original_authority = document(replay[authority_key])
    require(original_authority.get('source_sha256') == TRAINING_SOURCE_SHA256, 'original training authority source mismatch')
    authority_fixed = dict(schema_version='direct_reference_control_request_v1' if arm == 'off' else 'direct_reference_on_study_request_v1',
                           arm=arm, epochs=100, fold=0, seed=42, config_sha256=request['config']['sha256'])
    require(all(type(original_authority.get(key)) is type(value) and original_authority[key] == value
                for key, value in authority_fixed.items()), 'original training authority recipe mismatch')
    require(isinstance(original_authority.get('diagnostic_request_sha256'), str)
            and re.fullmatch('[0-9a-f]{64}', original_authority['diagnostic_request_sha256']),
            'original diagnostic request identity required')
    if arm == 'off':
        require(original_authority.get('diagnostic_gate') == result.get('diagnostic_gate'), 'original diagnostic gate mismatch')
    if arm == 'on':
        require(completion.get('launcher_sha256') == replay.get('launcher_sha256') == original_authority.get('launcher_sha256'),
                'ON launcher provenance mismatch')
        require(original_authority.get('off_completion') == request['arms']['off']['completion']
                and original_authority.get('registration') == result['registration'], 'ON original OFF/table authority mismatch')
        phase_authority = original_authority.get('phase_authority')
        authenticate(phase_authority)
        require(phase_authority['sha256'] == identities['phase_authority_sha256'], 'original phase authority mismatch')
    actual = replay.get('predictions')
    validate_predictions(actual, val_rows)
    comparison = pilots.compare_replay(predictions, actual)
    require(comparison['passed'] and comparison == replay.get('replay')
            and replay.get('max_abs_difference') == comparison['max_abs_difference'], 'fresh replay predictions mismatch')
    require(all(actual_row[key] == expected_row[key] for actual_row, expected_row in zip(actual, predictions)
                for key in ('context_available', 'referral_required')), 'fresh replay context mismatch')
    validate_summary(replay.get('summary', {}), actual)
    in_process = result.get('replay', {})
    require(in_process.get('passed') is True and in_process.get('fresh_process') is False
            and type(in_process.get('n_images')) is int and in_process['n_images'] == 222
            and in_process.get('phase') == 'train', 'full in-process replay required')
    checkpoint = core.read_direct_reference_checkpoint(Path(result['checkpoint']['path']), effective, identities)
    require(checkpoint['phase'] == 'train' and checkpoint['epoch'] == best_epoch
            and checkpoint['selection_key'] == result['selection_key'], 'selected core checkpoint header mismatch')
    provenance = checkpoint['initialization_provenance']
    require(isinstance(provenance.get('base', {}).get('checkpoint_path'), str), 'original initialization location required')
    return dict(result=result, config=effective, identities=identities, provenance=provenance, authority=original_authority,
                checkpoint=checkpoint, history=history_binding)


def fresh_model(request, selected, arm, device='cuda'):
    authenticate(request['weights'])
    model = build_direct_reference_model(absolute(request['weights']['path']), WEIGHTS_SHA256,
                                        reference_enabled=arm == 'on', head_seed=42, adapter_seed=42)
    current = copy.deepcopy(model.initialization_provenance)
    original = copy.deepcopy(selected['provenance'])
    require(current['base'].get('checkpoint_path') == request['weights']['path']
            and current['base'].get('checkpoint_sha256') == WEIGHTS_SHA256, 'current official weights provenance mismatch')
    current['base'].pop('checkpoint_path', None)
    original['base'].pop('checkpoint_path', None)
    require(current == original, 'fresh scientific initialization provenance mismatch')
    require(pilots.tensor_sha256(model.state_dict()) == selected['result']['fresh_initial_state_sha256'],
            'fresh initial tensor hash mismatch')
    model.initialization_provenance = copy.deepcopy(selected['provenance'])
    core._validate_model(model, selected['config'], selected['identities'])
    model.load_state_dict(selected['checkpoint']['state_dict'], strict=True)
    return model.to(device)


def all_bindings(value):
    if isinstance(value, dict):
        if set(value) == {'path', 'sha256'}:
            yield value
        else:
            for item in value.values():
                yield from all_bindings(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_bindings(item)


def score(binding):
    require_gpu_allocation()
    request, config = load_request(binding)
    rows = {role: load_role_rows(Path(request['records']['path']), Path(request['split']['path']), role)
            for role in ('train', 'val')}
    selected = {arm: validate_arm(request, config, arm, rows['val'], rows['train']) for arm in ('off', 'on')}
    for name in ('source_sha256', 'weights_sha256', 'records_sha256', 'split_sha256', 'enrollment_sha256',
                 'registration_sha256', 'phase_authority_sha256', 'preprocess_sha256'):
        require(selected['off']['identities'][name] == selected['on']['identities'][name], 'matched arm identity mismatch')
    require(selected['off']['result']['fresh_initial_state_sha256'] == selected['on']['result']['fresh_initial_state_sha256'],
            'matched arm initial tensor state required')
    require(selected['off']['authority']['diagnostic_request_sha256'] == selected['on']['authority']['diagnostic_request_sha256']
            and selected['off']['result']['diagnostic_gate'] == selected['on']['result']['diagnostic_gate'],
            'matched original diagnostic authority required')
    protected = list(all_bindings(request))
    for arm in ('off', 'on'):
        protected.extend(all_bindings(selected[arm]['result']))
        protected.extend(all_bindings(selected[arm]['authority']))
        protected.append(selected[arm]['history'])
        protected.extend(all_bindings(document(request['arms'][arm]['completion'])))
        protected.extend(all_bindings(document(request['arms'][arm]['fresh_replay'])))
    output = absolute(request['output_root'])
    require(all(not output.is_relative_to(absolute(item['path'])) and not absolute(item['path']).is_relative_to(output)
                for item in protected), 'output overlaps embedded training input')
    for item in protected:
        authenticate(item)
    pilots.configure_precision('cuda', 'D_running_pilot')
    models = {arm: fresh_model(request, selected[arm], arm) for arm in ('off', 'on')}
    joined, _ = consumer._membership(content(request['records']), content(request['split']))
    test_rows = joined.loc[joined.role == 'test'].sort_values('image_id').reset_index(drop=True)
    evaluation = dict(schema_version='protocol_a_evaluation_v1', protocol='A', fold=0,
                      records_sha256=request['records']['sha256'], split_sha256=request['split']['sha256'],
                      enrollment_manifest_sha256=request['enrollment']['sha256'],
                      test_image_ids=test_rows.image_id.tolist(), source_validation_selected=True,
                      checkpoints={arm: selected[arm]['result']['checkpoint'] for arm in ('off', 'on')})
    output.mkdir(parents=True, exist_ok=False)
    evaluation_binding = write(output / 'evaluation.json', evaluation)
    context = consumer.load_evaluation_context(request['records'], request['split'], request['enrollment'], request['root'],
                                               Path(request['raw_root']), evaluation_binding=evaluation_binding)
    built = consumer.build_evaluation_registration(context, output / 'test-registration.json')
    registration = bind(built.path)
    table = consumer.load_evaluation_registration(context, registration)
    reports = {}
    for arm in ('off', 'on'):
        data = consumer.ProtocolAEvaluationDataset(context, table, reference_enabled=arm == 'on')
        loader = DataLoader(data, batch_size=1, shuffle=False, drop_last=False, num_workers=0,
                            generator=torch.Generator().manual_seed(3043))
        summary, predictions = core.evaluate_direct_reference(models[arm], loader, 'cuda')
        validate_predictions(predictions, test_rows)
        validate_summary(summary, predictions)
        directory = output / arm
        directory.mkdir()
        pilots._write_csv(directory / 'test_predictions.csv', predictions)
        summary_binding = write(directory / 'test_summary.json', dict(summary, protocol='A', fold=0, role='test',
                               aggregate=False, selected_on='source_validation', selected_epoch=selected[arm]['result']['selected_epoch']))
        reports[arm] = dict(checkpoint=selected[arm]['result']['checkpoint'],
                            train_result=request['arms'][arm]['train_result'], completion=request['arms'][arm]['completion'],
                            fresh_replay=request['arms'][arm]['fresh_replay'], training_identities=selected[arm]['identities'],
                            predictions=bind(directory / 'test_predictions.csv'), summary=summary_binding)
    validate_source(request)
    for item in (binding, *protected):
        authenticate(item)
    result = dict(schema_version='protocol_a_current_score_result_v1', passed=True, protocol='A', fold=0,
                  role='test', n_images=222, aggregate=False, seed=42, eval_batch_size=1,
                  request=binding, evaluation_source_sha256=request['evaluation_source_sha256'],
                  training_source_sha256=TRAINING_SOURCE_SHA256, evaluation=evaluation_binding,
                  registration=registration, image_ids=test_rows.image_id.tolist(), arms=reports,
                  learning_performed=False, threshold_tuning_performed=False)
    write(output / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args(argv)
    return score(dict(path=args.request, sha256=args.request_sha256))


if __name__ == '__main__':
    main()
