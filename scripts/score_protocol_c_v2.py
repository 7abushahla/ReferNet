"""Evaluate the locked Protocol C queries with fold-0 Protocol A study arms.

The externally bound protocol_c_v2_score_v1 request has protocol=C, seed=42,
eval_batch_size=1, source_sha256 (frozen scientific source inventory hash),
a_request (path/SHA256 binding of the fold-0 protocol_a_request_v1 study
request), a_gate_sha256 (the study's gate/gate.json hash), target_split,
target_enrollment, enrollment, root, raw_root and output_root.

Both completed A arms are authenticated through the study's own receipt chain
(protocol_a_study.load_request, completed_arm and the fresh-replay receipts)
before any C authority or product is opened. A test data is never opened.
Only the locked 467 target queries are evaluated, with a shared registration
artifact, no augmentation or learning, and the fixed bonafide_score >= 0.5
genuine decision (ties genuine). Outputs are written below
output_root/evaluation, which must not already exist.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import torch
from torch.utils.data import DataLoader

from anchor_job_manifest import source_entries, object_sha256
import protocol_a_study as study
import score_protocol_a_current as driver
import score_protocol_c_current as current
from refer_or_clear import direct_reference_pilot as core, pilots
from refer_or_clear import protocol_c_contract as contract_api
from refer_or_clear.data import protocol_c_reference as consumer
from refer_or_clear.data.direct_reference import _bytes, _json
from refer_or_clear.models.direct_reference import build_direct_reference_model

SCHEMA = 'protocol_c_v2_score_v1'
RESULT_SCHEMA = 'protocol_c_v2_score_result_v1'
FIELDS = {'source_sha256', 'a_request', 'a_gate_sha256', 'target_split', 'target_enrollment',
          'enrollment', 'root', 'raw_root', 'output_root'}
# optional: the frozen source this scorer runs from, when it is a later revision than the A training source
# (source_sha256 stays the training source that the A request, receipts and checkpoints are bound to)
OPTIONAL_FIELDS = {'scoring_source_sha256'}
MATCHED_IDENTITIES = ('source_sha256', 'weights_sha256', 'records_sha256', 'split_sha256', 'enrollment_sha256',
                      'registration_sha256', 'phase_authority_sha256', 'preprocess_sha256')

require = driver.require
validate_predictions = current.validate_predictions
summarize_predictions = current.summarize_predictions
validate_summary = current.validate_summary
evaluate = current.evaluate


def checksum(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None, 'lowercase SHA256 required')
    return value


def binding_shape(value):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'path/SHA256 binding required')
    checksum(value['sha256'])
    driver.absolute(value['path'])
    return value


def scoring_source_sha256(request):
    return checksum(request.get('scoring_source_sha256', request['source_sha256']))


def validate_source(request):
    checksum(request['source_sha256'])
    require(scoring_source_sha256(request) == object_sha256(source_entries(ROOT)),
            'frozen scientific source differs from request')
    for name, module in tuple(sys.modules.items()):
        if name == 'refer_or_clear' or name.startswith('refer_or_clear.'):
            location = getattr(module, '__file__', None)
            require(location is not None and Path(location).resolve().is_relative_to(ROOT / 'src'),
                    'foreign scientific module imported')


def load_request(binding):
    """Metadata-only request validation: no A receipt, C authority or product is opened."""
    request = driver.document(binding)
    fixed = dict(schema_version=SCHEMA, protocol='C', seed=42, eval_batch_size=1)
    require(set(request) - OPTIONAL_FIELDS == set(fixed) | FIELDS and set(request) <= set(fixed) | FIELDS | OPTIONAL_FIELDS,
            'strict C v2 scoring request fields required')
    require(all(type(request.get(key)) is type(value) and request[key] == value for key, value in fixed.items()),
            'fixed C transfer contract mismatch')
    validate_source(request)
    for name in ('a_request', 'target_split', 'target_enrollment', 'enrollment', 'root'):
        binding_shape(request[name])
    checksum(request['a_gate_sha256'])
    driver.absolute(request['raw_root'])
    output = driver.absolute(request['output_root'])
    require(not (output / 'evaluation').exists(), 'locked C evaluation already exists')
    return request


def validate_pair(arms):
    for name in MATCHED_IDENTITIES:
        require(arms['off']['identities'][name] == arms['on']['identities'][name], 'matched arm identity mismatch')
    require(arms['off']['receipt']['fresh_initial_state_sha256'] == arms['on']['receipt']['fresh_initial_state_sha256'],
            'matched arm initial tensor state required')


def authenticate_arms(request):
    """Authenticate the fold-0 A study and both completed arms through the study's receipt chain."""
    a_binding = request['a_request']
    a_request, context, base, a_output = study.load_request(a_binding)
    require(type(a_request.get('fold')) is int and a_request['fold'] == 0 and context.contract.fold == 0,
            'fold-0 Protocol A study required')
    require(a_request.get('source_sha256') == request['source_sha256'], 'A study source differs from C request source')
    gate_binding = study.pinned_gate(a_output, request['a_gate_sha256'])
    require(_json(gate_binding).get('request') == a_binding, 'training gate belongs to a different request')
    arms = {}
    for arm in ('off', 'on'):
        receipt, config = study.completed_arm(a_request, context, base, a_output, arm, gate_binding, a_binding)
        directory = a_output / 'training' / arm
        fresh_binding = driver.bind(directory / 'fresh-replay.json')
        fresh = _json(fresh_binding)
        require(fresh.get('passed') is True and fresh.get('fresh_process') is True
                and fresh.get('checkpoint') == receipt['checkpoint'] and fresh.get('request') == a_binding,
                'passed fresh-process replay of the selected checkpoint required')
        require(type(receipt.get('selected_epoch')) is int and 1 <= receipt['selected_epoch'] <= 100,
                'selected epoch required')
        arms[arm] = dict(receipt=receipt, config=config, identities=receipt['identities'],
                         checkpoint=receipt['checkpoint'], selected_epoch=receipt['selected_epoch'],
                         train_result=driver.bind(directory / 'train.json'), fresh_replay=fresh_binding)
    validate_pair(arms)
    return dict(binding=a_binding, request=a_request, context=context, base=base, output=a_output,
                gate=gate_binding, arms=arms)


def fresh_model(a_request, config, device):
    """arm_model-style fresh construction with the A study's fixed seeds and arm recipe."""
    recipe = dict(core.ARM_RECIPE_DEFAULTS, **{key: config[key] for key in core.ARM_RECIPE_DEFAULTS if key in config})
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    pilots.configure_precision(device, 'D_running_pilot')
    _bytes(a_request['weights'])
    return build_direct_reference_model(Path(a_request['weights']['path']), a_request['weights']['sha256'],
                                        reference_enabled=config['reference_enabled'], head_seed=42, adapter_seed=42,
                                        local_enabled=recipe['local_branch'], reference_bn=recipe['reference_bn'],
                                        fused_query_weight=recipe['fused_query_weight']).to(device)


def arm_model(a_request, arm, device):
    receipt, config = arm['receipt'], arm['config']
    model = fresh_model(a_request, config, device)
    require(pilots.tensor_sha256(model.state_dict()) == receipt['fresh_initial_state_sha256'],
            'fresh initial tensor hash mismatch')
    core._validate_model(model, config, receipt['identities'])
    checkpoint = core.read_direct_reference_checkpoint(Path(receipt['checkpoint']['path']), config, receipt['identities'])
    require(checkpoint['phase'] == 'train' and checkpoint['epoch'] == receipt['selected_epoch']
            and checkpoint['selection_key'] == receipt['selection_key'], 'selected checkpoint header mismatch')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model


def source_inputs(request, a_request):
    """Records and source-split bindings for the C consumer: the paths the C enrollment manifest was built with,
    required to hash to the A study's records and split (same content, possibly another copy; the consumer compares
    the manifest's contract paths verbatim)."""
    manifest = driver.document(request['enrollment'])
    contract = manifest.get('contract')
    require(isinstance(contract, dict) and isinstance(contract.get('records_path'), str)
            and isinstance(contract.get('source_split_path'), str), 'C enrollment manifest contract paths required')
    bindings = []
    for name, key in (('records', 'records_path'), ('split', 'source_split_path')):
        path = driver.absolute(contract[key])
        require(path.is_file() and driver.digest(path) == a_request[name]['sha256'],
                f'C enrollment {key} must hash to the A study {name}')
        bindings.append(dict(path=str(path), sha256=a_request[name]['sha256']))
    return tuple(bindings)


def score(binding, device='cuda'):
    request = load_request(binding)
    study_run = authenticate_arms(request)
    a_request, a_context, arms = study_run['request'], study_run['context'], study_run['arms']
    require(len(a_context.contract.source_train) == 1778 and len(a_context.contract.source_val) == 222,
            'actual A source 1778/222 required')
    require(request['target_split']['sha256'] == contract_api.TARGET_SPLIT_SHA256,
            'original C target split identity required')
    protected = list(driver.all_bindings(request)) + list(driver.all_bindings(a_request)) + [study_run['gate']]
    for arm in ('off', 'on'):
        protected.extend(driver.all_bindings(arms[arm]['receipt']))
        protected.append(arms[arm]['train_result'])
        protected.append(arms[arm]['fresh_replay'])
    for item in protected:
        driver.authenticate(item)
    records, split = source_inputs(request, a_request)
    contract = contract_api.load_protocol_c_contract(records['path'], split['path'], request['target_split']['path'],
                                                     records_sha256=records['sha256'])
    target = contract_api.load_target_enrollment(contract, request['target_enrollment']['path'],
                                                 expected_sha256=request['target_enrollment']['sha256'])
    identifiers = sorted(target.query_ids)
    require(len(identifiers) == len(set(identifiers)) == 467, 'locked 467 C queries required')
    published = len(target.support_ids) + len(target.query_ids) + len(target.excluded_ids)
    require(published == contract_api.TARGET_COUNT, 'published C target count mismatch')
    output = driver.absolute(request['output_root'])
    evaluation_root = output / 'evaluation'
    relocation = driver.document(request['root'])
    manifest = driver.document(request['enrollment'])
    protected_paths = [ROOT, driver.absolute(binding['path']), driver.absolute(request['raw_root']),
                       driver.absolute(manifest['output_root']), driver.absolute(relocation['original_root']),
                       driver.absolute(relocation['relocated_root']), study_run['output'],
                       Path(a_context.raw_root), Path(a_context.manifest['output_root']), Path(a_context.product_root)]
    protected_paths.extend(driver.absolute(item['path']) for item in protected)
    require(not evaluation_root.exists() and all(not output.is_relative_to(path) and not path.is_relative_to(output)
                                                 for path in protected_paths),
            'new isolated C output outside protected inputs required')
    models = {arm: arm_model(a_request, arms[arm], device) for arm in ('off', 'on')}
    evaluation = dict(schema_version='protocol_c_evaluation_v1', protocol='C', deployment=consumer.DEPLOYMENT,
        source_protocol='A', source_fold=0, source_validation_selected=True, source_sha256=request['source_sha256'],
        scoring_source_sha256=scoring_source_sha256(request),
        records_sha256=contract.records_sha256, source_split_sha256=contract.source_split_sha256,
        target_split_sha256=contract.target_split_sha256, target_enrollment_sha256=request['target_enrollment']['sha256'],
        enrollment_manifest_sha256=request['enrollment']['sha256'], authorized_image_ids=identifiers,
        a_request=study_run['binding'], a_gate=study_run['gate'],
        checkpoints={arm: arms[arm]['checkpoint'] for arm in ('off', 'on')},
        consumer_inputs=dict(records=records, split=split),
        selected_epochs={arm: arms[arm]['selected_epoch'] for arm in ('off', 'on')},
        fresh_replays={arm: arms[arm]['fresh_replay'] for arm in ('off', 'on')})
    evaluation_root.mkdir(parents=True, exist_ok=False)
    evaluation_binding = driver.write(evaluation_root / 'authorization.json', evaluation)
    context = consumer.load_reference_context(records['path'], split['path'], request['target_split']['path'],
        records_sha256=records['sha256'], target_binding=request['target_enrollment'],
        enrollment_binding=request['enrollment'], raw_root=Path(request['raw_root']),
        evaluation_binding=evaluation_binding, root_binding=request['root'])
    test_rows = context.rows()
    require(test_rows.image_id.tolist() == identifiers, 'exact ordered C consumer query membership required')
    built = consumer.build_protocol_c_registration(context, output_path=evaluation_root / 'registration.json')
    registration = driver.bind(built.path)
    table = consumer.load_protocol_c_registration(context, registration)
    reports = {}
    for arm in ('off', 'on'):
        data = consumer.ProtocolCReferenceDataset(context, table, reference_enabled=arm == 'on')
        loader = DataLoader(data, batch_size=1, shuffle=False, drop_last=False, num_workers=0,
                            generator=torch.Generator().manual_seed(3043))
        summary, predictions = evaluate(models[arm], loader, device)
        validate_predictions(predictions, test_rows)
        validate_summary(summary, predictions)
        directory = evaluation_root / arm
        directory.mkdir()
        pilots._write_csv(directory / 'test_predictions.csv', predictions)
        summary_binding = driver.write(directory / 'test_summary.json', dict(summary, protocol='C', role='test',
            aggregate=False, source_protocol='A', source_fold=0, selected_on='source_validation',
            selected_epoch=arms[arm]['selected_epoch'], source_checkpoint=arms[arm]['checkpoint'],
            authorization=evaluation_binding, registration=registration, published_target_count=published,
            n_target_support=len(target.support_ids), n_target_excluded=len(target.excluded_ids),
            query_count=len(identifiers), threshold_policy='fixed_bonafide_threshold_0.5',
            predictions=driver.bind(directory / 'test_predictions.csv')))
        reports[arm] = dict(checkpoint=arms[arm]['checkpoint'], selected_epoch=arms[arm]['selected_epoch'],
            train_result=arms[arm]['train_result'], fresh_replay=arms[arm]['fresh_replay'],
            training_identities=copy.deepcopy(arms[arm]['identities']),
            predictions=driver.bind(directory / 'test_predictions.csv'), summary=summary_binding)
    validate_source(request)
    for item in (binding, *protected, evaluation_binding, registration):
        driver.authenticate(item)
    result = dict(schema_version=RESULT_SCHEMA, passed=True, protocol='C', role='test',
        deployment=consumer.DEPLOYMENT, source_protocol='A', source_fold=0, n_source_train=1778, n_source_val=222,
        n_images=467, published_target_count=published, n_target_support=len(target.support_ids),
        n_target_excluded=len(target.excluded_ids), aggregate=False, seed=42, eval_batch_size=1, request=binding,
        source_sha256=request['source_sha256'], a_request=study_run['binding'], a_gate=study_run['gate'],
        authorization=evaluation_binding, registration=registration, image_ids=identifiers, arms=reports,
        learning_performed=False, threshold_tuning=False, bonafide_threshold=0.5,
        decision_rule='bonafide_score >= 0.5 is genuine; ties genuine')
    driver.write(evaluation_root / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--allow-cpu', action='store_true', help='test-only: skip the GPU allocation guard')
    args = parser.parse_args(argv)
    if args.allow_cpu:
        device = 'cpu'
    else:
        driver.require_gpu_allocation()
        device = 'cuda'
    return score(dict(path=str(Path(args.request).resolve()), sha256=args.request_sha256), device)


if __name__ == '__main__':
    main()
