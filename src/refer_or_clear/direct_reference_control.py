"""Externally authorized, fixed OFF100 control and selected fresh-process replay."""
from __future__ import annotations

import argparse
import copy
import csv
from io import StringIO
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
from torch.utils.data import DataLoader

from . import direct_reference_train as service
from .data import direct_reference as bridge
from .direct_reference_pilot import read_direct_reference_checkpoint, replay_direct_reference_checkpoint
from .pilots import _selection_record, compare_replay, json_sha256, summarize_predictions, tensor_sha256
from .train import selection_key


def authorize_control(context, control_binding):
    """Authenticate separate OFF authority and same-source diagnostics before pixels."""
    control = bridge._json(control_binding)
    fields = {'schema_version', 'source_sha256', 'config_sha256', 'diagnostic_request_sha256',
              'arm', 'epochs', 'fold', 'seed', 'diagnostic_gate'}
    service.require(isinstance(control, dict) and set(control) == fields, 'fixed control authority fields required')
    service.require(control['schema_version'] == 'direct_reference_control_request_v1' and control['arm'] == 'off',
                    'only OFF control authority accepted')
    for name, value in (('epochs', 100), ('fold', 0), ('seed', 42)):
        service.require(type(control[name]) is int and control[name] == value, f'fixed control setting: {name}')
    service.require(control['source_sha256'] == context['identities']['source_sha256'], 'control source mismatch')
    service.require(control['diagnostic_request_sha256'] == context['request']['sha256'], 'diagnostic request mismatch')
    config_binding = context['manifest']['authorities']['config']
    bridge._bytes(config_binding)
    service.require(control['config_sha256'] == config_binding['sha256'], 'original config bytes mismatch')
    gate = control['diagnostic_gate']
    service.require(isinstance(gate, dict) and set(gate) == {'path', 'sha256'}, 'diagnostic path/hash required')
    authenticated, receipt, rows = service._inputs(context, gate)
    return authenticated, receipt, rows, control


def _canonical(binding, expected):
    service.require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}, 'artifact path/hash required')
    service.require(binding['path'] == str(expected), f'canonical artifact path required: {expected}')
    return bridge._bytes(binding)


def _csv(content):
    return list(csv.DictReader(StringIO(content.decode('utf-8'))))


def _number(value):
    try:
        parsed = float(value)
    except (ValueError, TypeError):
        raise ValueError('finite numeric artifact field required') from None
    service.require(math.isfinite(parsed), 'nonfinite artifact field')
    return parsed


def _artifacts(context, receipt, rows, control, train_binding):
    directory = context['run_dir'] / 'training/off'
    content = _canonical(train_binding, directory / 'train.json')
    result = bridge._json(train_binding)
    fixed = dict(passed=True, phase='train', diagnostic_only=False, epochs_completed=100,
                 optimizer_steps=22200, role='val', arm='off')
    service.require(all(result.get(key) == value and type(result.get(key)) is type(value)
                        for key, value in fixed.items()), 'complete OFF100 training result required')
    service.require(result.get('scheduler') == dict(total_updates=22200, warmup_updates=1000), 'scheduler mismatch')
    service.require(result.get('diagnostic_gate') == control['diagnostic_gate'], 'result diagnostic authority mismatch')
    replay = result.get('replay', {})
    service.require(replay.get('passed') is True and replay.get('fresh_process') is False
                    and type(replay.get('n_images')) is int and replay['n_images'] == 222 and replay.get('phase') == 'train',
                    'separate full in-process replay required')
    phase_binding = service.bind(context['run_dir'] / 'training/shared/phase-authority.json')
    preprocess = service._preprocess(context, phase_binding)
    _canonical(result.get('registration'), context['run_dir'] / 'training/shared/registration.json')
    prepared = service._load_prepared(context, receipt, rows, control['diagnostic_gate'],
                                     dict(preprocess=preprocess, table_binding=result['registration']))
    config = copy.deepcopy(context['config'])
    config.update(arm='off', reference_enabled=False)
    predictions = _csv(_canonical(result.get('predictions'), directory / 'val_predictions.csv'))
    for prediction in predictions:
        prediction['label'] = int(prediction['label'])
        for field in ('attack_logit', 'query_logit', 'correction_logit', 'attack_probability', 'bonafide_score'):
            prediction[field] = _number(prediction[field])
        for field in ('context_available', 'referral_required'):
            service.require(prediction.get(field) in ('True', 'False'), 'prediction availability must be boolean')
            prediction[field] = prediction[field] == 'True'
        service.require(prediction['referral_required'] is not prediction['context_available'], 'referral mismatch')
        probability = torch.tensor(prediction['attack_logit'], dtype=torch.float32).sigmoid().item()
        service.require(abs(prediction['attack_probability'] - probability) <= 1e-7
                        and abs(prediction['bonafide_score'] - (1. - probability)) <= 1e-7,
                        'CSV score/logit mismatch')
    service._predictions(dict(n_images=len(predictions)), predictions, rows['val'])
    selected_summary = summarize_predictions(predictions)
    history_binding = service.bind(directory / 'log.csv')
    history = _csv(bridge._bytes(history_binding))
    service.require(len(history) == 100, 'exactly 100 history rows required')
    best_key, best_epoch = None, None
    for epoch, entry in enumerate(history, 1):
        service.require(entry.get('epoch') == str(epoch) and entry.get('n_images') == '1776'
                        and entry.get('optimizer_steps') == '222', 'full ordered epoch history required')
        accuracy, auc = _number(entry.get('val_accuracy')), _number(entry.get('val_auc'))
        service.require(0 <= accuracy <= 1 and 0 <= auc <= 1, 'invalid selection metrics')
        key = selection_key('accuracy', accuracy, auc, _number(entry.get('val_image_loss')), _number(entry.get('loss')))
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
    service.require(type(result.get('selected_epoch')) is int and result['selected_epoch'] == best_epoch
                    and result.get('selection_key') == _selection_record(best_key), 'selected epoch/earlier tie mismatch')
    service.require(selected_summary['accuracy'] == best_key[0] and selected_summary['auc'] == best_key[1],
                    'selected history metrics differ from CSV predictions')
    _canonical(result.get('checkpoint'), directory / 'best.pt')
    model = service._fresh_model(context, False, receipt)
    service.require(result.get('fresh_initial_state_sha256') == tensor_sha256(model.state_dict()), 'fresh state mismatch')
    identities = service._checkpoint_identities(context, config, preprocess, prepared['table_binding'],
                                                json_sha256(model.initialization_provenance))
    service.require(result.get('identities') == identities, 'training identities mismatch')
    checkpoint = read_direct_reference_checkpoint(directory / 'best.pt', config, identities)
    service.require(checkpoint['phase'] == 'train' and checkpoint['epoch'] == best_epoch
                    and checkpoint['selection_key'] == _selection_record(best_key), 'selected training checkpoint mismatch')
    service.require(bridge._bytes(train_binding) == content, 'training result changed during authentication')
    return dict(result=result, model=model, config=config, identities=identities, prepared=prepared,
                predictions=predictions, history=history_binding)


def replay_control(context, control_binding, train_result_binding):
    """Authenticate selected full-validation artifacts before any query iteration."""
    context, receipt, rows, control = authorize_control(context, control_binding)
    artifacts = _artifacts(context, receipt, rows, control, train_result_binding)
    with service.pixel_audit(context, rows['val']) as opened:
        dataset = service._dataset(context, artifacts['prepared'], rows['val'], False)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0,
                            generator=torch.Generator().manual_seed(context['config']['seed'] + 3001))
        summary, predictions, replay = replay_direct_reference_checkpoint(
            Path(artifacts['result']['checkpoint']['path']), artifacts['model'], loader, context['device'],
            config=artifacts['config'], identities=artifacts['identities'], expected_predictions=artifacts['predictions'])
    service._predictions(summary, predictions, rows['val'])
    service.require(replay.get('passed') is True and summary.get('buffers_unchanged') is True, 'selected replay failed')
    report = dict(schema_version='direct_reference_control_replay_v1', passed=True, fresh_process=True,
                  phase='train', arm='off', pid=os.getpid(), train_result=train_result_binding, control_request=control_binding,
                  checkpoint=artifacts['result']['checkpoint'], identities=artifacts['identities'],
                  history=artifacts['history'], image_ids=[row['image_id'] for row in predictions], n_images=222,
                  buffers_unchanged=True, max_abs_difference=replay['max_abs_difference'], replay=replay,
                  summary=summary, predictions=predictions, query_reads=opened, query_reads_scope='query_decode_helper_reads')
    authorize_control(context, control_binding)
    bridge._json(train_result_binding)
    bridge._bytes(artifacts['result']['checkpoint'])
    bridge._bytes(artifacts['result']['predictions'])
    bridge._bytes(artifacts['history'])
    bridge._bytes(artifacts['prepared']['table_binding'])
    bridge._bytes(artifacts['prepared']['preprocess']['phase_authority'])
    service.write_receipt(context['run_dir'] / 'training/off/fresh-replay.json', report)
    return report


def run_control(context, control_binding):
    """Run only the authorized OFF recipe, then require an externally bound child."""
    context, receipt, rows, control = authorize_control(context, control_binding)
    result = service.run_training(context, False, control['diagnostic_gate'])
    train_binding = service.bind(context['run_dir'] / 'training/off/train.json')
    service.require(bridge._json(train_binding) == result, 'returned training result differs from artifact')
    expected = _artifacts(context, receipt, rows, control, train_binding)
    del expected['model']
    command = [sys.executable, str(Path(__file__).resolve().parents[2] / 'scripts/train_direct_reference_control.py'),
               '--request', context['request']['path'], '--request-sha256', context['request']['sha256'],
               '--control-request', control_binding['path'], '--control-request-sha256', control_binding['sha256'],
               '--run-dir', str(context['run_dir']), '--device', str(context['device']), '--phase', 'replay',
               '--train-result', train_binding['path'], '--train-result-sha256', train_binding['sha256']]
    subprocess.run(command, check=True)
    replay_binding = service.bind(context['run_dir'] / 'training/off/fresh-replay.json')
    replay = bridge._json(replay_binding)
    fixed = dict(schema_version='direct_reference_control_replay_v1', passed=True, fresh_process=True,
                 phase='train', arm='off', n_images=222, buffers_unchanged=True, train_result=train_binding,
                 control_request=control_binding, checkpoint=result['checkpoint'], identities=expected['identities'],
                 history=expected['history'], image_ids=rows['val'].image_id.tolist())
    service.require(all(replay.get(key) == value and type(replay.get(key)) is type(value)
                        for key, value in fixed.items()), 'complete fresh child binding required')
    service.require(type(replay.get('pid')) is int and replay['pid'] > 0 and replay['pid'] != os.getpid(),
                    'distinct child process required')
    service.require(type(replay.get('max_abs_difference')) in (int, float)
                    and math.isfinite(replay['max_abs_difference']) and replay['max_abs_difference'] >= 0,
                    'finite child replay difference required')
    service._predictions(replay.get('summary', {}), replay.get('predictions', []), rows['val'])
    comparison = compare_replay(expected['predictions'], replay['predictions'])
    service.require(comparison['passed'] and comparison == replay.get('replay')
                    and replay['max_abs_difference'] == comparison['max_abs_difference']
                    and replay['summary'].get('buffers_unchanged') is True, 'child replay evidence mismatch')
    summary = summarize_predictions(replay['predictions'])
    service.require(all(replay['summary'].get(key) == value for key, value in summary.items()), 'child summary mismatch')
    for original, actual in zip(expected['predictions'], replay['predictions']):
        service.require(all(actual.get(key) is original[key] for key in ('context_available', 'referral_required')),
                        'child availability/referral mismatch')
    authorize_control(context, control_binding)
    bridge._json(train_binding)
    bridge._bytes(result['checkpoint'])
    bridge._bytes(result['predictions'])
    bridge._bytes(expected['history'])
    bridge._bytes(expected['prepared']['table_binding'])
    bridge._bytes(expected['prepared']['preprocess']['phase_authority'])
    report = dict(schema_version='direct_reference_control_result_v1', passed=True, arm='off', phase='train',
                  epochs_completed=100, train_result=train_binding, control_request=control_binding,
                  checkpoint=result['checkpoint'], identities=expected['identities'], fresh_replay=replay_binding)
    service.write_receipt(context['run_dir'] / 'training/off/control.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--control-request', type=Path, required=True)
    parser.add_argument('--control-request-sha256', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    parser.add_argument('--phase', choices=['train', 'replay'], default='train')
    parser.add_argument('--train-result', type=Path)
    parser.add_argument('--train-result-sha256')
    args = parser.parse_args(argv)
    if args.phase == 'replay' and (args.train_result is None or args.train_result_sha256 is None):
        parser.error('replay requires external train-result path and SHA256')
    if args.phase == 'train' and (args.train_result is not None or args.train_result_sha256 is not None):
        parser.error('training does not accept a resume/replay artifact')
    context = service.load_context(args.request, args.request_sha256, args.run_dir, args.device)
    control_binding = dict(path=str(args.control_request), sha256=args.control_request_sha256)
    if args.phase == 'replay':
        return replay_control(context, control_binding, dict(path=str(args.train_result), sha256=args.train_result_sha256))
    return run_control(context, control_binding)
