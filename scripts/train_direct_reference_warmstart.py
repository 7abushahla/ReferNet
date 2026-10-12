"""External exploratory OFF-snapshot to frozen-base ON adapter training, fixed ten epochs."""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import os
import random
import sys

HELPER_SHA = 'f2697cbf85067baeba354077dec2bc0427e39fcf71b0005396f0f265a054c012'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_helper(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts and not any(parent.is_symlink() for parent in (path, *path.parents)), 'absolute helper without symlinks required')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == HELPER_SHA, 'pinned ON helper SHA256 mismatch')
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('_warmstart_on_helper', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def request_document(helper, binding):
    authority = helper.document(binding)
    fixed = dict(schema_version='direct_reference_warmstart_request_v1', exploratory_only=True,
                 arm='on', epochs=10, fold=0, seed=42, batch_size=8, helper_sha256=HELPER_SHA,
                 launcher_sha256=helper.digest(__file__), source_sha256=helper.SCIENTIFIC_SHA256)
    extra = {'original_run_dir', 'config_sha256', 'diagnostic_request_sha256', 'control_request', 'checkpoint', 'history', 'registration', 'phase_authority'}
    require(set(authority) == set(fixed) | extra, 'fixed exploratory authority fields required')
    require(all(authority.get(name) == value and type(authority.get(name)) is type(value) for name, value in fixed.items()), 'fixed exploratory authority mismatch')
    helper.absolute(authority['original_run_dir'])
    return authority


def authorize(helper, bundle, context, binding, output):
    authority = request_document(helper, binding)
    require(authority['original_run_dir'] == str(context['run_dir']), 'original run directory mismatch')
    require(authority['source_sha256'] == context['identities']['source_sha256']
            and authority['config_sha256'] == context['manifest']['authorities']['config']['sha256']
            and authority['diagnostic_request_sha256'] == context['request']['sha256'], 'original source/config/request mismatch')
    output = helper.absolute(output)
    protected = list(context['manifest']['paths'].values()) + [str(context['run_dir']), context['request']['path'], binding['path'], helper.__file__, __file__]
    protected += [value['path'] for value in context['manifest']['authorities'].values()]
    protected += [authority[name]['path'] for name in ('control_request', 'checkpoint', 'history', 'registration', 'phase_authority')]
    require(not output.exists(), 'new isolated exploratory output required')
    for value in protected:
        root = helper.absolute(value)
        require(not output.is_relative_to(root) and not root.is_relative_to(output), 'exploratory output overlaps immutable inputs')
    context, receipt, rows, control = bundle.off.authorize_control(context, authority['control_request'])
    for name in ('registration', 'phase_authority'):
        bundle.off._canonical(authority[name], context['run_dir'] / 'training/shared' / ('registration.json' if name == 'registration' else 'phase-authority.json'))
    prepared = bundle.service._load_prepared(context, receipt, rows, control['diagnostic_gate'],
        dict(preprocess=bundle.service._preprocess(context, authority['phase_authority']), table_binding=authority['registration']))
    history = bundle.off._csv(bundle.bridge._bytes(authority['history']))
    require(1 <= len(history) <= 100, 'bounded completed snapshot history required')
    best_key, best_epoch = None, None
    for epoch, entry in enumerate(history, 1):
        require(entry.get('epoch') == str(epoch) and entry.get('n_images') == '1776'
                and entry.get('optimizer_steps') == '222', 'ordered complete snapshot epochs required')
        key = tuple(bundle.off._number(entry[name]) for name in ('val_accuracy', 'val_auc'))
        require(all(0 <= value <= 1 for value in key), 'snapshot metrics outside probability range')
        if best_key is None or key > best_key: best_key, best_epoch = key, epoch
    bundle.bridge._bytes(authority['checkpoint'])
    model = bundle.service._fresh_model(context, False, receipt)
    config = dict(context['config'], arm='off', reference_enabled=False)
    identities = bundle.service._checkpoint_identities(context, config, prepared['preprocess'], prepared['table_binding'],
                                                      bundle.pilots.json_sha256(model.initialization_provenance))
    checkpoint = bundle.core.read_direct_reference_checkpoint(authority['checkpoint']['path'], config, identities)
    require(checkpoint['phase'] == 'train' and checkpoint['epoch'] == best_epoch
            and checkpoint['selection_key'] == bundle.pilots._selection_record(best_key), 'checkpoint is not captured history earliest best')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    bundle.bridge._bytes(authority['checkpoint'])
    return dict(context=context, receipt=receipt, rows=rows, prepared=prepared, control=control, authority=authority,
                checkpoint=checkpoint, baseline=model, snapshot_epochs=len(history), identities=identities)


def frozen_model(original, state):
    class FrozenBaseClassifier(type(original)):
        def train(self, mode=True):
            super().train(mode)
            self.base.eval()
            return self
    model = FrozenBaseClassifier(original.base, reference_enabled=True, adapter_seed=42)
    model.load_state_dict(state, strict=True)
    for parameter in model.base.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.train()
    return model


def adapter_optimizer(model, torch):
    require(all(not parameter.requires_grad for parameter in model.base.parameters()), 'base must be frozen')
    require(all(parameter.requires_grad for parameter in model.adapter.parameters()), 'complete adapter must train')
    return torch.optim.AdamW(model.adapter.parameters(), lr=1e-3, weight_decay=.05, betas=(.9, .999), eps=1e-8)


def train_epoch(model, loader, optimizer, scheduler, device, bundle):
    bundle.core._strict_fp32(model, device)
    model.train()
    loss_sum, images, updates, skipped = 0.0, 0, 0, 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        inputs, labels = bundle.core._batch_inputs(model, batch, device)
        outputs = bundle.core._forward(model, inputs)
        loss = bundle.core._image_loss(outputs, labels, track='D_running_pilot')
        require(bundle.torch.isfinite(loss).item(), 'nonfinite exploratory loss')
        if loss.requires_grad:
            loss.backward()
            require(all(parameter.grad is None or bundle.torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()), 'nonfinite exploratory gradient')
            require(all(parameter.grad is None for parameter in model.base.parameters()), 'base gradient detected')
            bundle.torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            updates += 1
        else:
            _, available = model._validate_inputs(**inputs)
            require(not available.any().item(), 'detached loss despite available context')
            skipped += 1
        loss_sum += loss.item() * len(labels)
        images += len(labels)
    require(images > 0, 'empty exploratory epoch')
    return dict(loss=loss_sum / images, image_loss=loss_sum / images, n_images=images, optimizer_steps=updates, skipped_batches=skipped)


def save_state(path, model, optimizer, scheduler, epoch, key, dataset, generators, bindings, torch):
    import numpy as np
    state = np.random.get_state()
    rng = dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]], torch=torch.get_rng_state(),
               cuda=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
               dataset={name: getattr(dataset, name).getstate() for name in ('geometry_rng', 'photometry_rng')},
               generators={name: generator.get_state() for name, generator in generators.items()})
    torch.save(dict(schema_version='direct_reference_warmstart_checkpoint_v1', exploratory_only=True, epoch=epoch,
                    selection_key=list(key), bindings=bindings, state_dict={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng), path)


def summary_record(bundle, summary, predictions):
    result = copy.deepcopy(summary)
    result['genuine_fpr'] = len(summary['genuine_false_positives']) / summary['n_genuine'] if summary['n_genuine'] else None
    result['availability_reasons'] = dict(Counter(row.get('availability_reason', 'unspecified') for row in predictions))
    result['by_availability'] = {}
    for name, enabled in (('available', True), ('unavailable', False)):
        subset = [row for row in predictions if bool(row['context_available']) == enabled]
        result['by_availability'][name] = bundle.pilots.summarize_predictions(subset) if subset else None
    return result


def explore(helper, bundle, ready, output, binding):
    context, rows, prepared = ready['context'], ready['rows'], ready['prepared']
    output = helper.absolute(output)
    output.mkdir(parents=True, exist_ok=False)
    model = frozen_model(ready['baseline'], ready['checkpoint']['state_dict']).to(context['device'])
    baseline = ready['baseline']
    optimizer = adapter_optimizer(model, bundle.torch)
    scheduler = bundle.pilots.pilot_scheduler(optimizer, 'D_running_pilot', 2220)
    require(not optimizer.state, 'fresh adapter optimizer required')
    train_dataset = bundle.service._dataset(context, prepared, rows['train'], True, augmented=True)
    val_dataset = bundle.service._dataset(context, prepared, rows['val'], True)
    off_dataset = bundle.service._dataset(context, prepared, rows['val'], False)
    generators = {name: bundle.torch.Generator().manual_seed(42 + offset) for name, offset in (('sampler', 1001), ('val', 3001), ('baseline', 3001))}
    val_loader = bundle.DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0, generator=generators['val'])
    off_loader = bundle.DataLoader(off_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0, generator=generators['baseline'])
    base_hash = bundle.pilots.tensor_sha256(model.base.state_dict())
    buffer_hash = bundle.pilots.tensor_sha256(dict(model.base.named_buffers()))
    bindings = dict(authority=binding, scientific_identities=ready['identities'], snapshot=ready['authority']['checkpoint'],
                    history=ready['authority']['history'], registration=ready['authority']['registration'], phase_authority=ready['authority']['phase_authority'])
    def assert_base():
        require(bundle.pilots.tensor_sha256(model.base.state_dict()) == base_hash
                and bundle.pilots.tensor_sha256(dict(model.base.named_buffers())) == buffer_hash, 'frozen base tensors/buffers changed')
        require(all(not parameter.requires_grad and parameter.grad is None for parameter in model.base.parameters()), 'frozen base gradients changed')
        require(not any(module.training for module in model.base.modules()), 'base/BN must retain stored evaluation state')
    def evaluate(value, loader):
        with bundle.service.pixel_audit(context, rows['val']):
            summary, predictions = bundle.core.evaluate_direct_reference(value, loader, context['device'])
        bundle.service._predictions(summary, predictions, rows['val'])
        require(summary['buffers_unchanged'] is True, 'validation changed stored buffers')
        assert_base()
        return summary, predictions
    off_summary, off_predictions = evaluate(baseline, off_loader)
    bundle.pilots._write_csv(output / 'off_baseline_predictions.csv', off_predictions)
    bundle.service.write_receipt(output / 'off_baseline.json', summary_record(bundle, off_summary, off_predictions))
    require(all(off_summary[name] is not None and math.isclose(off_summary[name], value, rel_tol=0.0, abs_tol=1e-6)
                for name, value in zip(('accuracy', 'auc'), ready['checkpoint']['selection_key'])),
            'evaluated OFF baseline differs from captured checkpoint selection metrics')
    on_summary, on_predictions = evaluate(model, val_loader)
    bundle.pilots._write_csv(output / 'epoch0_predictions.csv', on_predictions)
    bundle.service.write_receipt(output / 'epoch0.json', summary_record(bundle, on_summary, on_predictions))
    best_key, best_epoch = (on_summary['accuracy'], on_summary['auc']), 0
    require(all(value is not None and math.isfinite(value) for value in best_key), 'finite cold ON selection metrics required')
    history = [dict(epoch=0, loss=None, image_loss=None, n_images=0, optimizer_steps=0, skipped_batches=0,
                    val_accuracy=best_key[0], val_auc=best_key[1], lr=optimizer.param_groups[0]['lr'],
                    omitted_train_ids=[], base_unchanged=True, stored_bn_unchanged=True)]
    save_state(output / 'best.pt', model, optimizer, scheduler, 0, best_key, train_dataset, generators, bindings, bundle.torch)
    for epoch in range(1, 11):
        order = bundle.torch.randperm(1778, generator=generators['sampler']).tolist()
        generators['train'] = bundle.torch.Generator().manual_seed(42 + 2001 + epoch)
        loader = bundle.DataLoader(train_dataset, batch_size=8, sampler=order, drop_last=True, num_workers=0, generator=generators['train'])
        with bundle.service.pixel_audit(context, rows['train']):
            stats = train_epoch(model, loader, optimizer, scheduler, context['device'], bundle)
        require(stats['n_images'] == 1776 and stats['optimizer_steps'] + stats['skipped_batches'] == 222, 'full TRAIN/drop-last accounting mismatch')
        assert_base()
        summary, predictions = evaluate(model, val_loader)
        key = (summary['accuracy'], summary['auc'])
        require(all(value is not None and math.isfinite(value) for value in key), 'finite validation selection metrics required')
        history.append(dict(epoch=epoch, **stats, val_accuracy=key[0], val_auc=key[1], lr=optimizer.param_groups[0]['lr'],
                            omitted_train_ids=rows['train'].iloc[order[1776:]].image_id.tolist(), base_unchanged=True, stored_bn_unchanged=True))
        bundle.service.write_receipt(output / f'epoch-{epoch:02d}.json', summary_record(bundle, summary, predictions))
        save_state(output / 'last.pt', model, optimizer, scheduler, epoch, key, train_dataset, generators, bindings, bundle.torch)
        if key > best_key:
            best_key, best_epoch = key, epoch
            save_state(output / 'best.pt', model, optimizer, scheduler, epoch, key, train_dataset, generators, bindings, bundle.torch)
        bundle.pilots._write_csv(output / 'log.csv', [{name: value for name, value in entry.items() if name != 'omitted_train_ids'} for entry in history])
        print(json.dumps(dict(epoch=epoch, val_accuracy=key[0], val_auc=key[1], optimizer_steps=stats['optimizer_steps'], skipped_batches=stats['skipped_batches'])), flush=True)
    updates = sum(entry['optimizer_steps'] for entry in history)
    require(scheduler.last_epoch == updates and updates <= 2220, 'actual scheduler update mismatch')
    chosen_binding = bundle.service.bind(output / 'best.pt')
    bundle.bridge._bytes(chosen_binding)
    chosen = bundle.torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    require(chosen['schema_version'] == 'direct_reference_warmstart_checkpoint_v1' and chosen['bindings'] == bindings
            and chosen['epoch'] == best_epoch and chosen['selection_key'] == list(best_key), 'exploratory selected state mismatch')
    model.load_state_dict(chosen['state_dict'], strict=True)
    selected_summary, selected_predictions = evaluate(model, val_loader)
    require((selected_summary['accuracy'], selected_summary['auc']) == best_key, 'selected validation metrics changed')
    model.reference_enabled = False
    disabled_summary, disabled_predictions = evaluate(model, off_loader)
    model.reference_enabled = True
    final_off_summary, final_off_predictions = evaluate(baseline, off_loader)
    require(bundle.pilots.compare_replay(off_predictions, final_off_predictions)['passed'], 'original OFF baseline changed')
    for name, predictions in (('selected_on', selected_predictions), ('selected_reference_disabled', disabled_predictions), ('original_off_final', final_off_predictions)):
        bundle.pilots._write_csv(output / (name + '_predictions.csv'), predictions)
    bundle.off.authorize_control(context, ready['authority']['control_request'])
    require(helper.document(binding) == ready['authority'], 'warmstart authority drift')
    for name in ('checkpoint', 'history', 'registration', 'phase_authority'): bundle.bridge._bytes(ready['authority'][name])
    require(helper.digest(helper.__file__) == HELPER_SHA and helper.digest(__file__) == ready['authority']['launcher_sha256'], 'external tooling drift')
    result = dict(schema_version='direct_reference_warmstart_result_v1', passed=True, exploratory_only=True,
                  independently_trained_matched_comparison=False, epochs_completed=10, arm='on', fold=0, seed=42, batch_size=8,
                  scientific_identities=ready['identities'], authority=binding, original_run_dir=str(context['run_dir']),
                  snapshot_checkpoint=ready['authority']['checkpoint'], snapshot_history=ready['authority']['history'],
                  snapshot_selected_epoch=ready['checkpoint']['epoch'], snapshot_epochs=ready['snapshot_epochs'],
                  registration=ready['authority']['registration'], phase_authority=ready['authority']['phase_authority'],
                  selected_epoch=best_epoch, selection_key=list(best_key), optimizer_steps=updates,
                  skipped_batches=sum(entry['skipped_batches'] for entry in history),
                  skip_policy='all-effective-context-unavailable batches retain loss/counts, skip optimizer and scheduler; no synthetic gradient anchor',
                  scheduler=dict(planned_updates=2220, warmup_updates=222, actual_updates=updates),
                  optimizer=dict(name='AdamW', parameters='adapter_only', lr=1e-3, weight_decay=.05, betas=[.9, .999], eps=1e-8),
                  frozen_base_sha256=base_hash, frozen_base_buffer_sha256=buffer_hash, base_unchanged=True, stored_bn_unchanged=True,
                  checkpoint=chosen_binding, continuation_checkpoint=bundle.service.bind(output / 'last.pt'), history=history,
                  off_baseline=summary_record(bundle, off_summary, off_predictions), cold_on=summary_record(bundle, on_summary, on_predictions),
                  selected_on=summary_record(bundle, selected_summary, selected_predictions),
                  selected_reference_disabled=summary_record(bundle, disabled_summary, disabled_predictions),
                  original_off_final=summary_record(bundle, final_off_summary, final_off_predictions), no_test_open=True,
                  decode_audit_scope='original TRAIN-only update / VAL-only evaluation helper guards; not a syscall audit')
    bundle.service.write_receipt(output / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--authority', '--warmstart-request', required=True)
    parser.add_argument('--authority-sha256', '--warmstart-request-sha256', required=True)
    parser.add_argument('--on-helper', '--helper', required=True)
    for name in ('source-dir', 'output-root'): parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    args = parser.parse_args(argv)
    helper = load_helper(Path(args.on_helper))
    helper.allocated_environment()
    binding = dict(path=args.authority, sha256=args.authority_sha256)
    authority = request_document(helper, binding)
    require(os.environ.get('OMP_NUM_THREADS') == os.environ.get('MKL_NUM_THREADS') == '4'
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'strict project thread/determinism environment required')
    run = helper.absolute(authority['original_run_dir'])
    manifest = json.loads((run / 'input_manifest.json').read_bytes())
    require(manifest['request']['sha256'] == authority['diagnostic_request_sha256'], 'original manifest request mismatch')
    bundle = helper.load_science(helper.absolute(args.source_dir), manifest['request'])
    bundle.torch.set_num_threads(4)
    bundle.torch.set_num_interop_threads(1)
    bundle.bridge.cv2.setNumThreads(4)
    context = bundle.service.load_context(Path(manifest['request']['path']), manifest['request']['sha256'], run, args.device)
    ready = authorize(helper, bundle, context, binding, Path(args.output_root))
    return explore(helper, bundle, ready, Path(args.output_root), binding)


if __name__ == '__main__':
    raise SystemExit(0 if main()['passed'] else 1)
