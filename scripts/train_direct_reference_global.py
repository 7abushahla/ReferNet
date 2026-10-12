"""External global-readout ablation using the unchanged original OFF inputs."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys

HELPER_SHA = 'f2697cbf85067baeba354077dec2bc0427e39fcf71b0005396f0f265a054c012'
ARCHITECTURE = 'anchor_direct_global_ablation_v1'
SCHEMA = 'direct_reference_global_result_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_helper(path):
    path = Path(path)
    require(path.is_absolute() and not any(parent.is_symlink() for parent in (path, *path.parents)), 'absolute helper without symlinks required')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == HELPER_SHA, 'pinned helper SHA256 mismatch')
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('_global_on_helper', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def request_document(helper, binding):
    authority = helper.document(binding)
    fixed = dict(schema_version='direct_reference_global_request_v1', arm='global_only', epochs=100,
                 fold=0, seed=42, batch_size=8, source_sha256=helper.SCIENTIFIC_SHA256,
                 helper_sha256=HELPER_SHA, launcher_sha256=helper.digest(__file__))
    extra = {'original_run_dir', 'config_sha256', 'diagnostic_request_sha256', 'control_request',
             'registration', 'phase_authority', 'smoke_gate', 'replay_result'}
    require(set(authority) == set(fixed) | extra, 'fixed global authority fields required')
    require(all(type(authority[name]) is type(value) and authority[name] == value for name, value in fixed.items()), 'fixed global authority mismatch')
    helper.absolute(authority['original_run_dir'])
    for name in ('config_sha256', 'diagnostic_request_sha256'):
        require(isinstance(authority[name], str) and re.fullmatch('[0-9a-f]{64}', authority[name]), 'valid original hash required')
    for name in ('control_request', 'registration', 'phase_authority', 'smoke_gate', 'replay_result'):
        if authority[name] is None and name in ('smoke_gate', 'replay_result'):
            continue
        helper.document(authority[name])
    return authority


def global_model(original):
    require(original.reference_enabled is False, 'fresh OFF model required')

    class GlobalClassifier(type(original)):
        architecture_id = ARCHITECTURE

        def forward(self, query, *, overlap, reference, context_available):
            self._validate_inputs(query, overlap, reference, context_available)
            logit = self.base(query)['attack_logit']
            return dict(attack_logit=logit, query_logit=logit, correction_logit=logit.new_zeros(logit.shape))

    original.__class__ = GlobalClassifier
    for parameter in original.adapter.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return original


def authorize(helper, bundle, context, binding, output):
    authority = request_document(helper, binding)
    require(authority['original_run_dir'] == str(context['run_dir']), 'original run binding mismatch')
    require(authority['source_sha256'] == context['identities']['source_sha256']
            and authority['config_sha256'] == context['manifest']['authorities']['config']['sha256']
            and authority['diagnostic_request_sha256'] == context['request']['sha256'], 'original source/config/request required')
    output = helper.absolute(output)
    protected = list(context['manifest']['paths'].values()) + [str(context['run_dir']), binding['path'], __file__, helper.__file__]
    protected += [value['path'] for value in context['manifest']['authorities'].values()]
    protected += [authority[name]['path'] for name in ('control_request', 'registration', 'phase_authority')]
    protected += [authority[name]['path'] for name in ('smoke_gate', 'replay_result') if authority[name] is not None]
    require(not output.exists(), 'new isolated output required')
    require(all(not output.is_relative_to(helper.absolute(value)) and not helper.absolute(value).is_relative_to(output)
                for value in protected), 'output overlaps immutable inputs')
    context, receipt, rows, control = bundle.off.authorize_control(context, authority['control_request'])
    for name in ('registration', 'phase_authority'):
        filename = 'registration.json' if name == 'registration' else 'phase-authority.json'
        bundle.off._canonical(authority[name], context['run_dir'] / 'training/shared' / filename)
    prepared = bundle.service._load_prepared(context, receipt, rows, control['diagnostic_gate'],
        dict(preprocess=bundle.service._preprocess(context, authority['phase_authority']), table_binding=authority['registration']))
    return dict(context=context, receipt=receipt, rows=rows, prepared=prepared, control=control, authority=authority, binding=binding)


def fresh_model(bundle, ready):
    model = bundle.service._fresh_model(ready['context'], False, ready['receipt'])
    original_hash = bundle.pilots.tensor_sha256(model.state_dict())
    model = global_model(model)
    require(bundle.pilots.tensor_sha256(model.state_dict()) == original_hash, 'wrapping changed original initial tensors')
    bundle.pilots.optimizer_coverage(model)
    return model, original_hash


def scientific_binding(ready):
    authority = ready['authority']
    return {name: authority[name] for name in ('source_sha256', 'config_sha256', 'diagnostic_request_sha256',
            'helper_sha256', 'launcher_sha256', 'control_request', 'registration', 'phase_authority', 'original_run_dir')}


def phase_rows(ready, phase):
    require(phase in ('smoke', 'train'), 'known training phase required')
    return {role: frame if phase == 'train' else frame.sort_values('image_id').iloc[:32].reset_index(drop=True)
            for role, frame in ready['rows'].items()}


def validate_predictions(bundle, summary, predictions, rows):
    require(summary['n_images'] == len(rows) == len(predictions), 'complete validation population required')
    require(summary['buffers_unchanged'] is True, 'evaluation changed model buffers')
    for actual, expected in zip(predictions, rows.to_dict('records')):
        require(all(actual[name] == expected[name] for name in ('image_id', 'design_id', 'attack_family', 'label')), 'prediction membership/order mismatch')
        require(actual['correction_logit'] == 0. and actual['attack_logit'] == actual['query_logit'], 'global-only logits required')


def evaluate(bundle, ready, model, loader, rows):
    with bundle.service.pixel_audit(ready['context'], rows):
        summary, predictions = bundle.core.evaluate_direct_reference(model, loader, ready['context']['device'])
    validate_predictions(bundle, summary, predictions, rows)
    return summary, predictions


def assert_adapter(bundle, model, expected):
    require(bundle.pilots.tensor_sha256(model.adapter.state_dict()) == expected, 'inactive adapter changed')
    require(all(not parameter.requires_grad and parameter.grad is None for parameter in model.adapter.parameters()), 'inactive adapter received gradients')


def invariance(bundle, ready, model, dataset, rows):
    torch = bundle.torch
    selected = rows.iloc[:8]
    expected = None
    before = bundle.pilots.tensor_sha256(dict(model.named_buffers()))
    model.eval()
    with bundle.service.pixel_audit(ready['context'], selected), torch.inference_mode():
        samples = [dataset[index] for index in range(8)]
        for size in (1, 2, 4, 8):
            loader = bundle.DataLoader(samples, batch_size=size, shuffle=False,
                                      generator=torch.Generator().manual_seed(3043))
            values = []
            for batch in loader:
                inputs, _ = bundle.core._batch_inputs(model, batch, ready['context']['device'])
                actual = bundle.core._forward(model, inputs)['attack_logit']
                direct = model.base(inputs['query'])['attack_logit']
                require(torch.equal(actual, direct), 'global wrapper differs from base query output')
                values.append(actual)
            values = torch.cat(values)
            if expected is None:
                expected = values
            require(torch.allclose(expected, values, atol=1e-5, rtol=1e-4), 'companion batch invariance failed')
        reversed_batch = next(iter(bundle.DataLoader(list(reversed(samples)), batch_size=8,
                                  generator=torch.Generator().manual_seed(3043))))
        inputs, _ = bundle.core._batch_inputs(model, reversed_batch, ready['context']['device'])
        reversed_values = bundle.core._forward(model, inputs)['attack_logit'].flip(0)
        require(torch.allclose(expected, reversed_values, atol=1e-5, rtol=1e-4), 'batch-position invariance failed')
    require(bundle.pilots.tensor_sha256(dict(model.named_buffers())) == before, 'invariance check changed BN buffers')
    return dict(passed=True, n_images=8, batch_sizes=[1, 2, 4, 8], reversed_order=True, buffers_unchanged=True)


def validate_smoke(helper, bundle, ready):
    gate = ready['authority']['smoke_gate']
    require(gate is not None, 'completed same-tool smoke required')
    result = helper.document(gate)
    require(result.get('schema_version') == SCHEMA and result.get('architecture_id') == ARCHITECTURE
            and result.get('passed') is True and result.get('phase') == 'smoke'
            and result.get('epochs_completed') == 1 and result.get('optimizer_steps') == 4
            and result.get('scientific_binding') == scientific_binding(ready), 'same-source global smoke mismatch')
    require(result.get('invariance', {}).get('passed') is True and result.get('adapter_unchanged') is True
            and result.get('no_test_open') is True, 'smoke model gates missing')
    replay = helper.document(result['fresh_replay'])
    require(replay.get('schema_version') == 'direct_reference_global_replay_v1' and replay.get('passed') is True
            and replay.get('fresh_process') is True and replay.get('n_images') == 32
            and replay.get('phase_result') == result['phase_result']
            and replay.get('scientific_binding') == scientific_binding(ready), 'fresh smoke replay missing')
    phase = helper.document(result['phase_result'])
    phase_contract(helper, bundle, ready, phase)
    require(all(result.get(name) == value for name, value in phase.items()
                if name != 'fresh_process_replay'), 'outer smoke differs from bound training phase')
    require(result.get('fresh_process_replay') is True and phase['phase'] == 'smoke'
            and phase['checkpoint'] == replay['checkpoint'], 'smoke replay state mismatch')
    compare_summary(bundle, phase['summary'], replay['summary'])
    bundle.bridge._bytes(phase['checkpoint'])


def compare_summary(bundle, expected, actual):
    require(set(expected) == set(actual), 'replay summary fields mismatch')
    for name, value in expected.items():
        if name == 'image_loss':
            require(isinstance(value, (int, float)) and isinstance(actual[name], (int, float))
                    and math.isfinite(value) and math.isfinite(actual[name])
                    and abs(value - actual[name]) <= bundle.pilots.ATOL + bundle.pilots.RTOL * abs(value),
                    'replay loss tolerance violation')
        else:
            require(value == actual[name], 'replay summary mismatch: ' + name)


def phase_contract(helper, bundle, ready, result):
    phase = result.get('phase')
    require(phase in ('smoke', 'train'), 'known recorded phase required')
    epochs, updates, count = (1, 4, 32) if phase == 'smoke' else (100, 222, 1776)
    fixed = dict(schema_version=SCHEMA, architecture_id=ARCHITECTURE, passed=True, arm='global_only',
                 fold=0, seed=42, epochs_completed=epochs, optimizer_steps=epochs * updates,
                 adapter_unchanged=True, no_test_open=True, fresh_process_replay=False)
    require(all(type(result.get(name)) is type(value) and result[name] == value for name, value in fixed.items()),
            'recorded phase contract mismatch')
    require(result.get('scientific_binding') == scientific_binding(ready), 'recorded scientific binding mismatch')
    authority = request_document(helper, result['authority'])
    require(scientific_binding(dict(authority=authority)) == scientific_binding(ready)
            and authority['replay_result'] is None, 'recorded training authority mismatch')
    require((authority['smoke_gate'] is None) == (phase == 'smoke'), 'recorded phase authority mismatch')
    initialization = bundle.service._receipt_initialization(ready['receipt']['training_gates']['off'])
    require(result.get('fresh_initial_state_sha256') == initialization['state_sha256'], 'recorded original initialization mismatch')
    history = bundle.off._csv(bundle.bridge._bytes(result['history']))
    require(len(history) == epochs, 'complete recorded history required')
    best_key, best_epoch = None, None
    for epoch, entry in enumerate(history, 1):
        require(entry.get('epoch') == str(epoch) and entry.get('optimizer_steps') == str(updates)
                and entry.get('n_images') == str(count), 'recorded epoch accounting mismatch')
        values = [float(entry[name]) for name in ('val_accuracy', 'val_auc', 'val_image_loss', 'loss')]
        require(all(math.isfinite(value) for value in values), 'finite history metrics required')
        key = bundle.selection_key('accuracy', *values)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
    require(type(result.get('selected_epoch')) is int and result['selected_epoch'] == best_epoch
            and result.get('selection_key') == bundle.pilots._selection_record(best_key), 'earliest selected history mismatch')
    expected = bundle.off._csv(bundle.bridge._bytes(result['predictions']))
    for row in expected:
        require(row['label'] in ('0', '1'), 'binary stored label required')
        row['label'] = int(row['label'])
        for name in ('attack_probability', 'bonafide_score', 'attack_logit', 'query_logit', 'correction_logit'):
            row[name] = float(row[name])
            require(math.isfinite(row[name]), 'finite stored prediction required')
        for name in ('context_available', 'referral_required'):
            require(row[name] in ('True', 'False'), 'stored boolean required')
            row[name] = row[name] == 'True'
        require(row['context_available'] is not row['referral_required'], 'stored referral mismatch')
        probability = bundle.torch.tensor(row['attack_logit'], dtype=bundle.torch.float32).sigmoid().item()
        tolerance = bundle.pilots.ATOL + bundle.pilots.RTOL * abs(probability)
        require(abs(row['attack_probability'] - probability) <= tolerance
                and abs(row['bonafide_score'] - (1. - probability)) <= tolerance, 'stored score/logit mismatch')
    rows = phase_rows(ready, phase)['val']
    validate_predictions(bundle, result['summary'], expected, rows)
    summary = bundle.pilots.summarize_predictions(expected)
    require(all(result['summary'].get(name) == value for name, value in summary.items()), 'stored prediction summary mismatch')
    require([summary['accuracy'], summary['auc']] == result['selection_key'], 'stored selected metrics mismatch')
    return expected


def save_checkpoint(bundle, ready, path, model, optimizer, scheduler, epoch, key, phase, sampler, dataset):
    bundle.torch.save(dict(schema_version='direct_reference_global_checkpoint_v1', architecture_id=ARCHITECTURE,
        phase=phase, epoch=epoch, selection_key=bundle.pilots._selection_record(key), scientific_binding=scientific_binding(ready),
        authority=ready['binding'], state_dict=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
        sampler_rng=sampler.get_state(), torch_rng=bundle.torch.get_rng_state(), cuda_rng=bundle.torch.cuda.get_rng_state_all(),
        geometry_rng=dataset.geometry_rng.getstate(), photometry_rng=dataset.photometry_rng.getstate()), path)


def replay(helper, bundle, ready, output):
    binding = ready['authority']['replay_result']
    require(binding is not None, 'replay requires a bound prior phase result')
    result = helper.document(binding)
    require(result.get('schema_version') == SCHEMA and result.get('architecture_id') == ARCHITECTURE
            and result.get('passed') is True and result.get('scientific_binding') == scientific_binding(ready), 'global phase result mismatch')
    original_authority = request_document(helper, result['authority'])
    require({key: value for key, value in original_authority.items() if key != 'replay_result'}
            == {key: value for key, value in ready['authority'].items() if key != 'replay_result'}, 'replay authority differs from training authority')
    rows = phase_rows(ready, result['phase'])['val']
    expected = phase_contract(helper, bundle, ready, result)
    bundle.bridge._bytes(result['checkpoint'])
    checkpoint = bundle.torch.load(result['checkpoint']['path'], map_location='cpu', weights_only=True)
    require(checkpoint.get('schema_version') == 'direct_reference_global_checkpoint_v1'
            and checkpoint.get('architecture_id') == ARCHITECTURE and checkpoint.get('scientific_binding') == scientific_binding(ready)
            and checkpoint.get('authority') == result['authority'] and checkpoint.get('epoch') == result['selected_epoch']
            and checkpoint.get('selection_key') == result['selection_key'] and checkpoint.get('phase') == result['phase'], 'checkpoint identity mismatch')
    model, initial_hash = fresh_model(bundle, ready)
    require(initial_hash == result['fresh_initial_state_sha256'], 'replay fresh initialization mismatch')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    dataset = bundle.service._dataset(ready['context'], ready['prepared'], rows, False)
    loader = bundle.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                              generator=bundle.torch.Generator().manual_seed(3043))
    summary, predictions = evaluate(bundle, ready, model, loader, rows)
    comparison = bundle.pilots.compare_replay(expected, predictions)
    require(comparison['passed'], 'fresh global replay failed')
    compare_summary(bundle, result['summary'], summary)
    output.mkdir(parents=True, exist_ok=False)
    value = dict(schema_version='direct_reference_global_replay_v1', passed=True, fresh_process=True,
                 pid=os.getpid(), n_images=len(rows), phase_result=binding, scientific_binding=scientific_binding(ready),
                 checkpoint=result['checkpoint'], summary=summary, comparison=comparison)
    bundle.service.write_receipt(output / 'result.json', value)
    return value


def train(helper, bundle, ready, output, phase, source_dir):
    require(ready['authority']['replay_result'] is None, 'training cannot consume a replay result')
    if phase == 'train':
        validate_smoke(helper, bundle, ready)
    else:
        require(ready['authority']['smoke_gate'] is None, 'fresh smoke cannot consume another run')
    rows = phase_rows(ready, phase)
    epochs, updates_per_epoch = (100, 222) if phase == 'train' else (1, 4)
    output.mkdir(parents=True, exist_ok=False)
    model, initial_hash = fresh_model(bundle, ready)
    adapter_hash = bundle.pilots.tensor_sha256(model.adapter.state_dict())
    def reject_adapter(*args):
        raise RuntimeError('global-only path executed local adapter')
    model.adapter.register_forward_pre_hook(reject_adapter)
    optimizer = bundle.pilots.make_pilot_optimizer(model, {'track': 'D_running_pilot'}, 'train')
    require(not optimizer.state, 'fresh optimizer required')
    scheduler = bundle.pilots.pilot_scheduler(optimizer, 'D_running_pilot', epochs * updates_per_epoch)
    dataset = bundle.service._dataset(ready['context'], ready['prepared'], rows['train'], False, augmented=True)
    val_dataset = bundle.service._dataset(ready['context'], ready['prepared'], rows['val'], False)
    val_loader = bundle.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,
                                  generator=bundle.torch.Generator().manual_seed(3043))
    sampler = bundle.torch.Generator().manual_seed(1043)
    history, omissions, best_key, best_epoch = [], [], None, 0
    gate = None
    for epoch in range(1, epochs + 1):
        order = bundle.torch.randperm(len(rows['train']), generator=sampler).tolist()
        used = len(order) // 8 * 8
        omissions.append(dict(epoch=epoch, image_ids=rows['train'].iloc[order[used:]].image_id.tolist()))
        loader = bundle.DataLoader(dataset, batch_size=8, sampler=order, drop_last=True, num_workers=0,
                                  generator=bundle.torch.Generator().manual_seed(2043 + epoch))
        counters = {name: int(value) for name, value in model.base.named_buffers() if name.endswith('num_batches_tracked')}
        with bundle.service.pixel_audit(ready['context'], rows['train']):
            stats = bundle.core.train_direct_reference_epoch(model, loader, optimizer, scheduler, ready['context']['device'])
        require(stats['optimizer_steps'] == updates_per_epoch and stats['n_images'] == used, 'training update/population mismatch')
        require(counters and all(int(dict(model.base.named_buffers())[name]) == value + updates_per_epoch
                                 for name, value in counters.items()), 'backbone BN update mismatch')
        for module in (model.base.backbone, model.base.classifier):
            require(any(parameter.grad is not None and bundle.torch.count_nonzero(parameter.grad).item() > 0
                        for parameter in module.parameters()), 'backbone/classifier gradient missing')
        assert_adapter(bundle, model, adapter_hash)
        summary, predictions = evaluate(bundle, ready, model, val_loader, rows['val'])
        key = bundle.selection_key('accuracy', summary['accuracy'], summary['auc'], summary['image_loss'], stats['loss'])
        history.append(dict(epoch=epoch, **stats, val_accuracy=summary['accuracy'], val_auc=summary['auc'],
                            val_image_loss=summary['image_loss'], lr_pretrained=optimizer.param_groups[0]['lr'],
                            lr_new=optimizer.param_groups[1]['lr']))
        bundle.pilots._write_csv(output / 'log.csv', history)
        save_checkpoint(bundle, ready, output / 'last.pt', model, optimizer, scheduler, epoch, key, phase, sampler, dataset)
        if best_key is None or key > best_key:
            best_key, best_epoch, best_summary = key, epoch, copy.deepcopy(summary)
            save_checkpoint(bundle, ready, output / 'best.pt', model, optimizer, scheduler, epoch, key, phase, sampler, dataset)
            bundle.pilots._write_csv(output / 'val_predictions.csv', predictions)
        print(json.dumps(dict(epoch=epoch, val_accuracy=summary['accuracy'], val_auc=summary['auc'])), flush=True)
    if phase == 'smoke':
        gate = invariance(bundle, ready, model, val_dataset, rows['val'])
    require(scheduler.last_epoch == epochs * updates_per_epoch, 'incomplete scheduler')
    assert_adapter(bundle, model, adapter_hash)
    require(helper.document(ready['binding']) == ready['authority'] and helper.digest(__file__) == ready['authority']['launcher_sha256'], 'external authority/tooling drift')
    bundle.off.authorize_control(ready['context'], ready['authority']['control_request'])
    for name in ('registration', 'phase_authority'):
        bundle.bridge._bytes(ready['authority'][name])
    result = dict(schema_version=SCHEMA, architecture_id=ARCHITECTURE, passed=True, phase=phase, arm='global_only',
                  fold=0, seed=42, epochs_completed=epochs, optimizer_steps=epochs * updates_per_epoch,
                  selected_epoch=best_epoch, selection_key=bundle.pilots._selection_record(best_key), summary=best_summary,
                  scientific_binding=scientific_binding(ready), authority=ready['binding'], fresh_initial_state_sha256=initial_hash,
                  checkpoint=bundle.service.bind(output / 'best.pt'), predictions=bundle.service.bind(output / 'val_predictions.csv'),
                  history=bundle.service.bind(output / 'log.csv'), omitted_train_ids=omissions, adapter_unchanged=True,
                  invariance=gate, no_test_open=True, fresh_process_replay=False)
    phase_binding = bundle.service.write_receipt(output / 'phase-result.json', result)
    child_authority = dict(ready['authority'], replay_result=phase_binding)
    child_binding = bundle.service.write_receipt(output / 'replay-request.json', child_authority)
    command = [sys.executable, str(Path(__file__).resolve()), '--phase', 'replay', '--authority', child_binding['path'],
               '--authority-sha256', child_binding['sha256'], '--source-dir', str(source_dir), '--on-helper', helper.__file__,
               '--output-root', str(output / 'fresh-replay')]
    subprocess.run(command, check=True)
    replay_binding = bundle.service.bind(output / 'fresh-replay/result.json')
    replay_value = helper.document(replay_binding)
    require(replay_value['passed'] is True and replay_value['fresh_process'] is True
            and replay_value['pid'] != os.getpid() and replay_value['phase_result'] == phase_binding
            and replay_value['checkpoint'] == result['checkpoint'], 'fresh child replay evidence mismatch')
    result.update(phase_result=phase_binding, fresh_replay=replay_binding, fresh_process_replay=True)
    bundle.service.write_receipt(output / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('authority', 'authority-sha256', 'source-dir', 'on-helper', 'output-root'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--phase', choices=('smoke', 'train', 'replay'), required=True)
    args = parser.parse_args(argv)
    helper = load_helper(args.on_helper)
    helper.allocated_environment()
    binding = dict(path=args.authority, sha256=args.authority_sha256)
    authority = request_document(helper, binding)
    require(os.environ.get('OMP_NUM_THREADS') == os.environ.get('MKL_NUM_THREADS') == '4'
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'fixed runtime environment required')
    manifest = json.loads((helper.absolute(authority['original_run_dir']) / 'input_manifest.json').read_bytes())
    require(manifest['request']['sha256'] == authority['diagnostic_request_sha256'], 'original diagnostic request mismatch')
    source = helper.absolute(args.source_dir)
    bundle = helper.load_science(source, manifest['request'])
    bundle.torch.set_num_threads(4)
    bundle.torch.set_num_interop_threads(1)
    bundle.bridge.cv2.setNumThreads(4)
    context = bundle.service.load_context(Path(manifest['request']['path']), manifest['request']['sha256'],
                                          Path(authority['original_run_dir']), 'cuda')
    output = helper.absolute(args.output_root)
    ready = authorize(helper, bundle, context, binding, output)
    if args.phase == 'replay':
        return replay(helper, bundle, ready, output)
    return train(helper, bundle, ready, output, args.phase, source)


if __name__ == '__main__':
    raise SystemExit(0 if main()['passed'] else 1)
