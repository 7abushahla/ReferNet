"""Evaluate locked C queries using authenticated, validation-selected A arms.

The externally bound protocol_c_current_score_v1 request has protocol=C,
seed=42, eval_batch_size=1, training_source_sha256, evaluation_source_sha256,
critical_source_hashes, source_selection, target_split, target_enrollment,
enrollment, root, raw_root and output_root. source_selection is an ordinary
metadata bundle with records, split, enrollment, weights, config and arms.
Each arm binds train_result, completion and fresh_replay. It is not A test
authorization. All file bindings are absolute path/SHA256 pairs. The current
C source inventory must retain the original nine critical scientific files.

Both original A completion/selection/replay chains and matched identities are
checked before C authorities or pixels. Only the locked 467 target queries are
evaluated, with a shared registration artifact, no augmentation or learning,
and the fixed bonafide_score >= 0.5 genuine decision (ties genuine). Products
use C's explicit full-bank and relocation schemas. Source counts remain
1778 train/222 validation; no legacy C source-training claim is made.
Inference reuses the core FP32/input/forward/loss primitives, with a local
summary for the recorded C face_swap/text_inpaint taxonomy. The original
SIDTD-only core summarizer is not modified or given relabeled C examples.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import torch
from torch.utils.data import DataLoader
import yaml

import score_protocol_a_current as driver
from refer_or_clear import protocol_c_contract as contract_api
from refer_or_clear.data import protocol_c_reference as consumer


def load_request(binding):
    request = driver.document(binding)
    fixed = dict(schema_version='protocol_c_current_score_v1', protocol='C', seed=42,
                 eval_batch_size=1, training_source_sha256=driver.TRAINING_SOURCE_SHA256)
    fields = {'evaluation_source_sha256', 'critical_source_hashes', 'source_selection',
              'target_split', 'target_enrollment', 'enrollment', 'root', 'raw_root', 'output_root'}
    driver.require(set(request) == set(fixed) | fields, 'strict C scoring request fields required')
    driver.require(all(type(request.get(key)) is type(value) and request[key] == value
                       for key, value in fixed.items()), 'fixed C transfer contract mismatch')
    driver.validate_source(request)
    source = request['source_selection']
    driver.require(isinstance(source, dict)
                   and set(source) == {'records', 'split', 'enrollment', 'weights', 'config', 'arms'},
                   'strict original A source-selection bundle required, not A test authority')
    for name, expected in (('records', driver.RECORDS_SHA256), ('split', driver.SPLIT_SHA256),
                           ('enrollment', driver.consumer.ENROLLMENT_SHA256), ('weights', driver.WEIGHTS_SHA256)):
        driver.require(isinstance(source[name], dict) and source[name].get('sha256') == expected,
                       f'original A {name} identity required')
        driver.authenticate(source[name])
    config = yaml.safe_load(driver.content(source['config']))
    fixed_config = dict(epochs=100, seed=42, head_seed=42, adapter_seed=42, fold=0,
                        batch_size=8, eval_batch_size=1, accumulation_steps=1, drop_last=True,
                        eval_drop_last=False, input_shape=[4, 512, 800], mask_weight=0.0)
    driver.require(isinstance(config, dict) and all(type(config.get(key)) is type(value)
                   and config[key] == value for key, value in fixed_config.items()),
                   'original full-training recipe required')
    driver.require(config.get('selection') == dict(primary='accuracy', tie_break='auc', exact_tie='earlier_epoch'),
                   'accuracy/AUC/earlier selection required')
    driver.require(isinstance(source['arms'], dict) and set(source['arms']) == {'off', 'on'},
                   'both completed A arms required')
    for arm in ('off', 'on'):
        authorities = source['arms'][arm]
        driver.require(isinstance(authorities, dict)
                       and set(authorities) == {'train_result', 'completion', 'fresh_replay'},
                       'original training/completion/fresh replay bindings required')
    return request, config


def validate_pair(selected):
    for name in ('source_sha256', 'weights_sha256', 'records_sha256', 'split_sha256', 'enrollment_sha256',
                 'registration_sha256', 'phase_authority_sha256', 'preprocess_sha256'):
        driver.require(selected['off']['identities'][name] == selected['on']['identities'][name],
                       'matched arm identity mismatch')
    driver.require(selected['off']['result']['fresh_initial_state_sha256']
                   == selected['on']['result']['fresh_initial_state_sha256'],
                   'matched arm initial tensor state required')
    driver.require(selected['off']['authority']['diagnostic_request_sha256']
                   == selected['on']['authority']['diagnostic_request_sha256']
                   and selected['off']['result']['diagnostic_gate'] == selected['on']['result']['diagnostic_gate'],
                   'matched original diagnostic authority required')


def validate_predictions(predictions, rows):
    driver.require(isinstance(predictions, list) and len(predictions) == len(rows) == 467,
                   'exact locked 467-query population required')
    for actual, expected in zip(predictions, rows.to_dict('records')):
        driver.require(all(type(actual.get(key)) is type(expected[key]) and actual[key] == expected[key]
                           for key in ('image_id', 'design_id', 'attack_family', 'label')),
                       'locked C prediction metadata/order mismatch')
        for key in ('attack_logit', 'query_logit', 'correction_logit', 'attack_probability', 'bonafide_score'):
            driver.number(actual.get(key))
        driver.require(type(actual.get('context_available')) is bool and type(actual.get('referral_required')) is bool
                       and actual['referral_required'] is not actual['context_available'],
                       'availability/referral mismatch')
        logit = torch.tensor(actual['attack_logit'], dtype=torch.float64)
        driver.require(abs(actual['attack_probability'] - logit.sigmoid().item()) <= 1e-7
                       and abs(actual['bonafide_score'] - (-logit).sigmoid().item()) <= 1e-7,
                       'prediction score/logit mismatch')
        driver.require(math.isclose(actual['attack_logit'], actual['query_logit'] + actual['correction_logit'],
                                     abs_tol=driver.pilots.ATOL, rel_tol=driver.pilots.RTOL),
                       'prediction logit decomposition mismatch')


def summarize_predictions(predictions):
    driver.require(all(row['label'] != 1 or row['attack_family'] in {'face_swap', 'text_inpaint'}
                       for row in predictions), 'unknown locked C attack family')
    scores = [row['bonafide_score'] for row in predictions]
    labels = [row['label'] for row in predictions]
    genuine = [row for row in predictions if row['label'] == 0]
    summary = dict(n_images=len(predictions), n_genuine=labels.count(0), n_altered=labels.count(1),
        accuracy=driver.pilots.M.deepid_acc(scores, labels), auc=driver.pilots.M.auc(scores, labels),
        genuine_false_positives=[row['image_id'] for row in genuine if row['bonafide_score'] < .5],
        altered_misses=[row['image_id'] for row in predictions if row['label'] == 1 and row['bonafide_score'] >= .5],
        buffers_unchanged=True, n_context_available=sum(row['context_available'] for row in predictions),
        n_context_unavailable=sum(not row['context_available'] for row in predictions),
        referral_ids=[row['image_id'] for row in predictions if row['referral_required']], subgroups={})
    for family in ('face_swap', 'text_inpaint'):
        altered = [row for row in predictions if row['label'] == 1 and row['attack_family'] == family]
        population = genuine + altered
        family_scores = [row['bonafide_score'] for row in population]
        family_labels = [row['label'] for row in population]
        summary['subgroups'][family] = dict(n_genuine=len(genuine), n_altered=len(altered),
            auc=driver.pilots.M.auc(family_scores, family_labels),
            accuracy=driver.pilots.M.deepid_acc(family_scores, family_labels) if genuine and altered else None)
    return summary


def validate_summary(summary, predictions):
    computed = summarize_predictions(predictions)
    driver.require(all(type(summary.get(key)) is type(value) and summary[key] == value
                       for key, value in computed.items()), 'recomputed C summary/buffers mismatch')
    labels = torch.tensor([row['label'] for row in predictions], dtype=torch.float32)
    logits = torch.tensor([row['attack_logit'] for row in predictions], dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    driver.require(driver.number(summary.get('image_loss')) >= 0
                   and math.isclose(summary['image_loss'], loss, abs_tol=driver.pilots.ATOL, rel_tol=driver.pilots.RTOL),
                   'recomputed C image loss mismatch')


def evaluate(model, loader, device):
    core = driver.core
    core._strict_fp32(model, device)
    model.eval()
    before = driver.pilots.tensor_sha256(dict(model.named_buffers()))
    predictions, loss_sum = [], 0.
    with torch.inference_mode():
        for batch in loader:
            inputs, labels = core._batch_inputs(model, batch, device)
            outputs = core._forward(model, inputs)
            loss = core._image_loss(outputs, labels, track=core.RECIPE_TRACK, reduction='sum')
            driver.require(bool(torch.isfinite(loss)), 'non-finite evaluation loss')
            loss_sum += loss.item()
            # Float64 and sigmoid(-logit), mirroring core.evaluate_direct_reference: a float32 sigmoid ties large logits.
            attack_logits = outputs['attack_logit'].detach().double()
            probabilities = attack_logits.sigmoid().cpu().tolist()
            bonafide = (-attack_logits).sigmoid().cpu().tolist()
            logits = {key: outputs[key].cpu().tolist() for key in ('attack_logit', 'query_logit', 'correction_logit')}
            for index, probability in enumerate(probabilities):
                meta = {key: str(batch['meta'][key][index]) for key in ('image_id', 'design_id', 'attack_family')}
                available = bool(inputs['context_available'][index].item())
                row = dict(meta, label=int(labels[index].item()), attack_probability=probability,
                           bonafide_score=bonafide[index], context_available=available, referral_required=not available)
                row.update({key: logits[key][index] for key in logits})
                if 'availability_reason' in batch['meta']:
                    row['availability_reason'] = str(batch['meta']['availability_reason'][index])
                predictions.append(row)
    driver.require(before == driver.pilots.tensor_sha256(dict(model.named_buffers())), 'evaluation changed model buffers')
    driver.require(bool(predictions), 'evaluation loader has no images')
    driver.require(len({row['image_id'] for row in predictions}) == len(predictions), 'duplicate evaluation image IDs')
    summary = summarize_predictions(predictions)
    summary['image_loss'] = loss_sum / len(predictions)
    return summary, predictions


def score(binding):
    driver.require_gpu_allocation()
    request, config = load_request(binding)
    source = request['source_selection']
    rows = {role: driver.load_role_rows(Path(source['records']['path']), Path(source['split']['path']), role)
            for role in ('train', 'val')}
    driver.require(len(rows['train']) == 1778 and len(rows['val']) == 222, 'actual A source 1778/222 required')
    selected = {arm: driver.validate_arm(source, config, arm, rows['val'], rows['train']) for arm in ('off', 'on')}
    validate_pair(selected)
    driver.require(isinstance(request['target_split'], dict)
                   and request['target_split'].get('sha256') == contract_api.TARGET_SPLIT_SHA256,
                   'original C target split identity required')
    protected = list(driver.all_bindings(request))
    for arm in ('off', 'on'):
        protected.extend(driver.all_bindings(selected[arm]['result']))
        protected.extend(driver.all_bindings(selected[arm]['authority']))
        protected.append(selected[arm]['history'])
        protected.extend(driver.all_bindings(driver.document(source['arms'][arm]['completion'])))
        protected.extend(driver.all_bindings(driver.document(source['arms'][arm]['fresh_replay'])))
    for item in protected:
        driver.authenticate(item)
    contract = contract_api.load_protocol_c_contract(source['records']['path'], source['split']['path'],
        request['target_split']['path'], records_sha256=source['records']['sha256'])
    target = contract_api.load_target_enrollment(contract, request['target_enrollment']['path'],
                                                expected_sha256=request['target_enrollment']['sha256'])
    identifiers = sorted(target.query_ids)
    driver.require(len(identifiers) == len(set(identifiers)) == 467, 'locked 467 C queries required')
    output = driver.absolute(request['output_root'])
    relocation = driver.document(request['root'])
    manifest = driver.document(request['enrollment'])
    protected_paths = [ROOT, driver.absolute(binding['path']), driver.absolute(request['raw_root']),
        driver.absolute(manifest['output_root']), driver.absolute(relocation['original_root']),
        driver.absolute(relocation['relocated_root'])]
    protected_paths.extend(driver.absolute(item['path']) for item in protected)
    driver.require(not output.exists() and all(not output.is_relative_to(path) and not path.is_relative_to(output)
                                              for path in protected_paths),
                   'new isolated C output outside protected inputs required')
    driver.pilots.configure_precision('cuda', 'D_running_pilot')
    models = {arm: driver.fresh_model(source, selected[arm], arm) for arm in ('off', 'on')}
    evaluation = dict(schema_version='protocol_c_evaluation_v1', protocol='C', deployment=consumer.DEPLOYMENT,
        source_protocol='A', source_fold=0, source_validation_selected=True,
        records_sha256=contract.records_sha256, source_split_sha256=contract.source_split_sha256,
        target_split_sha256=contract.target_split_sha256, target_enrollment_sha256=request['target_enrollment']['sha256'],
        enrollment_manifest_sha256=request['enrollment']['sha256'], authorized_image_ids=identifiers,
        checkpoints={arm: selected[arm]['result']['checkpoint'] for arm in ('off', 'on')})
    output.mkdir(parents=True, exist_ok=False)
    evaluation_binding = driver.write(output / 'evaluation.json', evaluation)
    context = consumer.load_reference_context(source['records']['path'], source['split']['path'],
        request['target_split']['path'], records_sha256=source['records']['sha256'],
        target_binding=request['target_enrollment'], enrollment_binding=request['enrollment'],
        raw_root=Path(request['raw_root']), evaluation_binding=evaluation_binding, root_binding=request['root'])
    test_rows = context.rows()
    driver.require(test_rows.image_id.tolist() == identifiers, 'exact ordered C consumer query membership required')
    built = consumer.build_protocol_c_registration(context, output_path=output / 'test-registration.json')
    registration = driver.bind(built.path)
    table = consumer.load_protocol_c_registration(context, registration)
    reports = {}
    for arm in ('off', 'on'):
        data = consumer.ProtocolCReferenceDataset(context, table, reference_enabled=arm == 'on')
        loader = DataLoader(data, batch_size=1, shuffle=False, drop_last=False, num_workers=0,
                            generator=torch.Generator().manual_seed(3043))
        summary, predictions = evaluate(models[arm], loader, 'cuda')
        validate_predictions(predictions, test_rows)
        validate_summary(summary, predictions)
        directory = output / arm
        directory.mkdir()
        driver.pilots._write_csv(directory / 'test_predictions.csv', predictions)
        summary_binding = driver.write(directory / 'test_summary.json', dict(summary, protocol='C', role='test',
            aggregate=False, source_protocol='A', source_fold=0, selected_on='source_validation',
            selected_epoch=selected[arm]['result']['selected_epoch']))
        reports[arm] = dict(checkpoint=selected[arm]['result']['checkpoint'],
            train_result=source['arms'][arm]['train_result'], completion=source['arms'][arm]['completion'],
            fresh_replay=source['arms'][arm]['fresh_replay'], training_identities=selected[arm]['identities'],
            predictions=driver.bind(directory / 'test_predictions.csv'), summary=summary_binding)
    driver.validate_source(request)
    for item in (binding, *protected, evaluation_binding, registration):
        driver.authenticate(item)
    result = dict(schema_version='protocol_c_current_score_result_v1', passed=True, protocol='C', role='test',
        deployment=consumer.DEPLOYMENT, source_protocol='A', source_fold=0, n_source_train=1778, n_source_val=222,
        n_images=467, published_target_count=599, n_target_support=66, n_target_excluded=66,
        aggregate=False, seed=42, eval_batch_size=1, request=binding,
        evaluation_source_sha256=request['evaluation_source_sha256'], training_source_sha256=driver.TRAINING_SOURCE_SHA256,
        evaluation=evaluation_binding, registration=registration, image_ids=identifiers, arms=reports,
        learning_performed=False, threshold_tuning=False, bonafide_threshold=0.5,
        decision_rule='bonafide_score >= 0.5 is genuine; ties genuine')
    driver.write(output / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args(argv)
    return score(dict(path=args.request, sha256=args.request_sha256))


if __name__ == '__main__':
    main()
