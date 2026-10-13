"""Allocated A0 OFF head-dropout probe, never an official A training receipt.

Authority direct_reference_regularization_probe_request_v1 binds base_request,
source_sha256, probability, auxiliary_query_weight, max_epochs=100, output_root.
Allowed (p,aux) pairs are (0,0), (.1,0), (0,.5). Only (0,0) may build the
train/val table; other probes require its registration binding. Replay requires gate SHA;
the smoke child uses --phase smoke. No TEST/evaluate entry exists.
"""
from __future__ import annotations

import argparse
import copy
import gc
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol_a_study as driver
from refer_or_clear.models.direct_reference_regularized import regularize_model

require = driver._require
RESULT = 'direct_reference_regularization_probe_result_v1'
CHECKPOINT = 'direct_reference_regularization_probe_checkpoint_v1'


def _load(binding):
    authority = driver._json(binding)
    fields = {'schema_version', 'base_request', 'source_sha256', 'probability', 'auxiliary_query_weight',
              'max_epochs', 'output_root'}
    require(set(authority) in (fields, fields | {'registration'})
            and authority.get('schema_version') == 'direct_reference_regularization_probe_request_v1',
            'strict probe authority required')
    probability = authority['probability']
    auxiliary = authority['auxiliary_query_weight']
    require(type(probability) in (int, float) and type(auxiliary) in (int, float)
            and (probability, auxiliary) in ((0., 0.), (.1, 0.), (0., .5)), 'restricted probability/auxiliary combination required')
    require(type(authority['max_epochs']) is int and authority['max_epochs'] == 100, 'fixed 100-epoch probe required')
    require((probability, auxiliary) == (0., 0.) or 'registration' in authority,
            'nonbaseline probe requires matched registration')
    source, context, base, original_output = driver.load_request(authority['base_request'])
    require(source['protocol'] == 'A' and type(source['fold']) is int and source['fold'] == 0
            and context.contract.fold == 0 and source['source_sha256'] == authority['source_sha256'],
            'same-source A0 base request required')
    config = driver.service.effective_config(base, context, False)
    require(config['epochs'] == 100 and config['seed'] == 42 and config['batch_size'] == 8
            and config['eval_batch_size'] == 1, 'fixed A0 recipe required')
    rows = {role: context.rows(role) for role in ('train', 'val')}
    for role, count in (('train', 1778), ('val', 222)):
        frame = rows[role]
        require(len(frame) == count and frame.role.eq(role).all() and frame.image_id.is_unique
                and frame.image_id.tolist() == sorted(frame.image_id.tolist()), 'exact sorted A0 source rows required')
    require(not set(rows['train'].image_id) & set(rows['val'].image_id), 'train/val ID overlap')
    output = driver._path(authority['output_root'])
    for protected in (driver.ROOT, original_output, context.raw_root,
                      driver._path(context.manifest['output_root']), driver._path(context.product_root)):
        require(not output.is_relative_to(protected) and not Path(protected).is_relative_to(output),
                'probe output overlaps original source/inputs/output')
    if 'registration' in authority:
        path = driver._path(authority['registration']['path'])
        require(not path.is_relative_to(output) and not output.is_relative_to(path), 'output overlaps shared registration')
    return authority, source, context, config, output, rows


def _table(ready, *, create=False, binding=None):
    authority, unused_source, context, unused_config, output, unused_rows = ready
    if create and 'registration' not in authority:
        table = driver.bridge.build_protocol_a_registration(context, roles=['train', 'val'],
                                                             output_path=output / 'registration.json')
        binding = driver.bind(output / 'registration.json')
    else:
        binding = authority.get('registration', binding)
        require(binding is not None, 'bound source registration required')
        table = driver.bridge.load_protocol_a_registration(context, binding, roles=['train', 'val'])
    return table, binding, driver.json_sha256(json.loads(table.serialized)['entries'])


def _model(ready):
    authority, source, unused_context, base, unused_output, unused_rows = ready
    model = driver.fresh_model(source, False)
    initial = driver.tensor_sha256(model.state_dict())
    model = regularize_model(model, authority['probability'])
    require(driver.tensor_sha256(model.state_dict()) == initial and model.reference_enabled is False,
            'regularization changed initialization or enabled references')
    require(model.initialization_provenance['regularization']['p'] == authority['probability']
            and model.architecture_id == 'anchor_direct_reference_ia18_head_dropout_v1', 'probe architecture/p mismatch')
    model.initialization_provenance['training_objective'] = dict(policy='fused_plus_auxiliary_query_bce_v1',
        auxiliary_query_weight=authority['auxiliary_query_weight'])
    config = copy.deepcopy(base)
    config.update(schema_version='direct_reference_regularization_probe_config_v1',
                  architecture_id=model.architecture_id, regularization_probability=authority['probability'],
                  auxiliary_query_weight=authority['auxiliary_query_weight'])
    return model, config, initial


def _loaders(ready, table, *, smoke=False, val_only=False):
    unused_authority, unused_source, context, config, unused_output, unused_rows = ready
    loaders = {}
    budgets = (('val', 1024**3),) if val_only else (('train', 3 * 1024**3), ('val', 1024**3))
    for role, budget in budgets:
        data = driver.bridge.ProtocolAReferenceDataset(context, table, role=role, reference_enabled=False,
            augment=dict(config['augment'], enabled=role == 'train', seed=42), cache_bytes=budget)
        if smoke:
            data = driver.Subset(data, range(32))
        loaders[role] = data
    return loaders


def _loader(data, role, *, order=None, epoch=0):
    return driver.DataLoader(data, batch_size=8 if role == 'train' else 1, sampler=order,
        shuffle=False, drop_last=role == 'train', num_workers=0,
        generator=driver.torch.Generator().manual_seed(42 + (2001 + epoch if role == 'train' else 3001)))


def _fresh(binding, parent, expected):
    replay = driver._json(binding)
    require(replay.get('schema_version') == 'direct_reference_regularization_probe_replay_v1'
            and replay.get('passed') is True and replay.get('fresh_process') is True
            and type(replay.get('pid')) is int and replay['pid'] > 0 and replay['pid'] != expected['pid']
            and replay.get('parent_result') == parent and replay.get('request') == expected['request']
            and replay.get('checkpoint') == expected['checkpoint'] and replay.get('identities') == expected['identities']
            and replay.get('comparison', {}).get('passed') is True, 'bound fresh-process replay required')
    return replay


def train_epoch(model, loader, optimizer, scheduler, device, auxiliary_query_weight):
    require(type(auxiliary_query_weight) in (int, float) and auxiliary_query_weight in (0., .5), 'fixed auxiliary weight required')
    if auxiliary_query_weight == 0:
        stats = driver.core.train_direct_reference_epoch(model, loader, optimizer, scheduler, device)
        return dict(stats, aux_query_loss=0.)
    driver.core._strict_fp32(model, device)
    model.train()
    totals = dict(loss=0., image_loss=0., aux_query_loss=0.)
    n_images, updates = 0, 0
    for batch in loader:
        inputs, labels = driver.core._batch_inputs(model, batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = driver.core._forward(model, inputs)
        fused = driver.core._image_loss(outputs, labels, track='D_running_pilot')
        query = driver.torch.nn.functional.binary_cross_entropy_with_logits(
            outputs['query_logit'].float(), labels.float().reshape(-1))
        auxiliary = auxiliary_query_weight * query
        loss = fused + auxiliary
        require(driver.torch.isfinite(loss).item(), 'nonfinite auxiliary training loss')
        loss.backward()
        require(all(parameter.grad is None or driver.torch.isfinite(parameter.grad).all().item()
                    for parameter in model.parameters()), 'nonfinite auxiliary gradient')
        driver.torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        count = len(labels)
        for key, value in (('loss', loss), ('image_loss', fused), ('aux_query_loss', auxiliary)):
            totals[key] += value.item() * count
        n_images += count
        updates += 1
    require(n_images > 0, 'empty auxiliary training loader')
    return dict({key: value / n_images for key, value in totals.items()}, n_images=n_images,
                optimizer_steps=updates, bce_map=None, dice_map=None)


def _gate(ready, binding, gate_sha256):
    output = ready[4]
    require(isinstance(gate_sha256, str) and len(gate_sha256) == 64, 'external smoke gate SHA256 required')
    gate = driver._json(dict(path=str(output / 'smoke/gate.json'), sha256=gate_sha256))
    require(gate.get('schema_version') == 'direct_reference_regularization_probe_gate_v1'
            and gate.get('request') == binding and gate.get('probability') == ready[0]['probability']
            and gate.get('auxiliary_query_weight') == ready[0]['auxiliary_query_weight']
            and gate.get('passed') is True and gate.get('no_test_open') is True, 'same-source/request/p smoke gate required')
    require(gate['smoke_result']['path'] == str(output / 'smoke/result.json'), 'canonical probe smoke required')
    smoke = driver._json(gate['smoke_result'])
    require(smoke.get('schema_version') == RESULT and smoke.get('phase') == 'smoke'
            and smoke.get('request') == binding and smoke['stats']['n_images'] == 32
            and smoke['stats']['optimizer_steps'] == 4 and smoke['summary']['n_images'] == 32
            and smoke['batch_invariance'].get('passed') is True
            and smoke['batch_invariance'].get('buffers_unchanged') is True
            and smoke['batch_invariance'].get('n_images') == 4
            and smoke.get('dropout_eval_disabled') is True, 'complete smoke and eval invariance required')
    _fresh(gate['fresh_replay'], gate['smoke_result'], smoke)
    require(gate['registration'] == smoke['registration']
            and gate['registration_entries_sha256'] == smoke['registration_entries_sha256']
            and gate['fresh_initial_state_sha256'] == smoke['fresh_initial_state_sha256'], 'smoke registration/init mismatch')
    return gate


def _run(ready, binding, table, registration, entry_sha, phase, *, gate=None):
    authority, source, context, unused_base, output, rows = ready
    directory = output / ('smoke' if phase == 'smoke' else 'training')
    require(not directory.exists(), 'new probe phase directory required')
    model, config, initial = _model(ready)
    if gate is not None:
        require(initial == gate['fresh_initial_state_sha256'], 'full probe must reset fresh pretrained tensors')
    identities = driver.checkpoint_ids(source, context, config, registration, binding, model)
    if gate is not None:
        require(identities == driver._json(gate['smoke_result'])['identities'], 'smoke/source/model identities changed')
    optimizer = driver.core.make_pilot_optimizer(model, {'track': 'D_running_pilot'}, 'train')
    scheduler = driver.core.pilot_scheduler(optimizer, 'D_running_pilot', 22200)
    data = _loaders(ready, table, smoke=phase == 'smoke')
    val_loader = _loader(data['val'], 'val')
    directory.mkdir(parents=True, exist_ok=False)
    sampler = driver.torch.Generator().manual_seed(1043)
    history, omissions, best_key = [], [], None
    best_summary, best_predictions, best_epoch = None, None, 0
    for epoch in range(1, 2 if phase == 'smoke' else 101):
        order = None if phase == 'smoke' else driver.torch.randperm(1778, generator=sampler).tolist()
        if order is not None:
            omissions.append(dict(epoch=epoch, image_ids=rows['train'].iloc[order[1776:]].image_id.tolist()))
        stats = train_epoch(model, _loader(data['train'], 'train', order=order, epoch=epoch),
                            optimizer, scheduler, 'cuda', authority['auxiliary_query_weight'])
        require(stats['n_images'] == (32 if phase == 'smoke' else 1776)
                and stats['optimizer_steps'] == (4 if phase == 'smoke' else 222), 'probe update accounting mismatch')
        summary, predictions = driver.core.evaluate_direct_reference(model, val_loader, 'cuda')
        driver.service._predictions(summary, predictions, rows['val'].head(32) if phase == 'smoke' else rows['val'])
        key = driver.selection_key('accuracy', summary['accuracy'], summary['auc'], summary['image_loss'], stats['loss'])
        history.append(dict(epoch=epoch, **stats, val_accuracy=summary['accuracy'], val_auc=summary['auc'],
                            val_image_loss=summary['image_loss'], lr_backbone=optimizer.param_groups[0]['lr'],
                            lr_new=optimizer.param_groups[-1]['lr']))
        driver._write_csv(directory / 'history.csv', history)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            best_summary, best_predictions = copy.deepcopy(summary), copy.deepcopy(predictions)
            driver.torch.save(dict(schema_version=CHECKPOINT, architecture_id=model.architecture_id, request=binding,
                phase=phase, config=config, identities=identities, epoch=epoch, selection_key=driver.core._selection_record(key),
                state_dict=model.state_dict(), initialization_provenance=model.initialization_provenance), directory / 'selected.pt')
    require(scheduler.last_epoch == (4 if phase == 'smoke' else 22200), 'fixed scheduler accounting mismatch')
    invariance = driver.invariant(model, val_loader) if phase == 'smoke' else None
    model.eval()
    disabled = all(not module.training for module in model.modules() if isinstance(module, driver.torch.nn.Dropout))
    require(disabled, 'evaluation dropout must be disabled')
    driver._write_csv(directory / 'val_predictions.csv', best_predictions)
    driver.write(directory / 'val_summary.json', dict(best_summary, role='val', selected_epoch=best_epoch))
    driver.write(directory / 'omissions.json', dict(epochs=omissions))
    result = dict(schema_version=RESULT, passed=True, diagnostic_only=True, no_test_open=True, arm='off', phase=phase,
        protocol='A', fold=0, probability=authority['probability'], auxiliary_query_weight=authority['auxiliary_query_weight'],
        request=binding, base_request=authority['base_request'],
        source_sha256=source['source_sha256'], config=config, identities=identities, pid=os.getpid(),
        epochs_completed=len(history), selected_epoch=best_epoch, optimizer_steps=scheduler.last_epoch,
        fresh_initial_state_sha256=initial, registration=registration, registration_entries_sha256=entry_sha,
        stats=stats, summary=best_summary, checkpoint=driver.bind(directory / 'selected.pt'),
        predictions=driver.bind(directory / 'val_predictions.csv'), batch_invariance=invariance, dropout_eval_disabled=disabled)
    driver.write(directory / 'result.json', result)
    return result


def _child(binding, phase, gate_sha256=None):
    command = [sys.executable, str(Path(__file__).resolve()), 'replay', '--request', binding['path'],
               '--request-sha256', binding['sha256'], '--phase', phase]
    if gate_sha256 is not None:
        command += ['--gate-sha256', gate_sha256]
    gc.collect()
    driver.torch.cuda.empty_cache()
    subprocess.run(command, check=True)


def smoke(binding):
    ready = _load(binding)
    require(not ready[4].exists(), 'new probe output required for smoke')
    table, registration, entry_sha = _table(ready, create=True)
    result = _run(ready, binding, table, registration, entry_sha, 'smoke')
    _child(binding, 'smoke')
    parent = driver.bind(ready[4] / 'smoke/result.json')
    fresh = driver.bind(ready[4] / 'smoke/fresh-replay.json')
    _fresh(fresh, parent, result)
    return driver.write(ready[4] / 'smoke/gate.json', dict(schema_version='direct_reference_regularization_probe_gate_v1',
        passed=True, no_test_open=True, request=binding, probability=ready[0]['probability'],
        auxiliary_query_weight=ready[0]['auxiliary_query_weight'], registration=registration,
        registration_entries_sha256=entry_sha, fresh_initial_state_sha256=result['fresh_initial_state_sha256'],
        smoke_result=parent, fresh_replay=fresh))


def train(binding, gate_sha256):
    ready = _load(binding)
    gate = _gate(ready, binding, gate_sha256)
    table, registration, entry_sha = _table(ready, binding=gate['registration'])
    require(registration == gate['registration'] and entry_sha == gate['registration_entries_sha256'], 'shared table changed')
    result = _run(ready, binding, table, registration, entry_sha, 'train', gate=gate)
    _child(binding, 'train', gate_sha256)
    parent = driver.bind(ready[4] / 'training/result.json')
    fresh = driver.bind(ready[4] / 'training/fresh-replay.json')
    replay_report = _fresh(fresh, parent, result)
    completed = dict(result, completed_after_fresh_replay=True, parent_result=parent, fresh_replay=fresh,
                     replay=replay_report['comparison'])
    driver.write(ready[4] / 'training/completion.json', completed)
    return completed


def replay(binding, *, phase='train', gate_sha256=None):
    require(phase in ('smoke', 'train'), 'source-only replay phase required')
    ready = _load(binding)
    if phase == 'train':
        _gate(ready, binding, gate_sha256)
    directory = ready[4] / ('smoke' if phase == 'smoke' else 'training')
    parent_binding = driver.bind(directory / 'result.json')
    parent = driver._json(parent_binding)
    require(parent.get('schema_version') == RESULT and parent.get('request') == binding
            and parent.get('phase') == phase and parent.get('probability') == ready[0]['probability']
            and parent.get('auxiliary_query_weight') == ready[0]['auxiliary_query_weight'], 'probe replay parent mismatch')
    table, registration, entry_sha = _table(ready, binding=parent['registration'])
    require(registration == parent['registration'] and entry_sha == parent['registration_entries_sha256'], 'replay table changed')
    payload = driver._bytes(parent['checkpoint'])
    checkpoint = driver.torch.load(BytesIO(payload), map_location='cpu', weights_only=True)
    model, config, initial = _model(ready)
    identities = driver.checkpoint_ids(ready[1], ready[2], config, registration, binding, model)
    require(checkpoint.get('schema_version') == CHECKPOINT and checkpoint.get('phase') == phase
            and checkpoint.get('request') == binding and checkpoint.get('config') == config
            and checkpoint.get('identities') == identities == parent['identities']
            and checkpoint.get('epoch') == parent['selected_epoch']
            and checkpoint.get('architecture_id') == model.architecture_id
            and checkpoint.get('initialization_provenance') == model.initialization_provenance
            and initial == parent['fresh_initial_state_sha256'], 'dedicated probe checkpoint mismatch')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    data = _loaders(ready, table, smoke=phase == 'smoke', val_only=True)['val']
    summary, actual = driver.core.evaluate_direct_reference(model, _loader(data, 'val'), 'cuda')
    expected = driver.predictions(parent['predictions'])
    driver.service._predictions(summary, actual, ready[5]['val'].head(32) if phase == 'smoke' else ready[5]['val'])
    comparison = driver.core.compare_replay(expected, actual)
    require(comparison['passed'] and all(summary.get(key) == parent['summary'].get(key)
            for key in ('n_images', 'accuracy', 'auc')) and summary.get('buffers_unchanged') is True,
            'fresh selected prediction/summary replay failed')
    for wanted, observed in zip(expected, actual):
        require(all(wanted[key] == observed[key] for key in ('label', 'design_id', 'attack_family',
                    'context_available', 'referral_required')), 'replay metadata mismatch')
    result = dict(schema_version='direct_reference_regularization_probe_replay_v1', passed=True, fresh_process=True,
        pid=os.getpid(), phase=phase, request=binding, parent_result=parent_binding, checkpoint=parent['checkpoint'],
        identities=identities, comparison=comparison, summary=summary, no_test_open=True)
    driver.write(directory / 'fresh-replay.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', choices=('smoke', 'train', 'replay'))
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--phase', choices=('smoke', 'train'), default='train')
    parser.add_argument('--gate-sha256')
    args = parser.parse_args(argv)
    require(args.command != 'smoke' or args.gate_sha256 is None, 'smoke may not inherit a training gate')
    driver.require_gpu_allocation()
    binding = dict(path=args.request, sha256=args.request_sha256)
    result = smoke(binding) if args.command == 'smoke' else train(binding, args.gate_sha256) if args.command == 'train' \
        else replay(binding, phase=args.phase, gate_sha256=args.gate_sha256)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
