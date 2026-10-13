"""GPU source-TRAIN normalization diagnostic, never optimization or recalibration.

The authenticated original OFF checkpoint retains its saved config/identities.
The current A loader/registration is separately reported diagnostic input, not
claimed to reproduce historical OFF preprocessing. No VAL or TEST pass exists.
"""
from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol_a_study as driver

require = driver._require
COMPONENTS = {'sum': 'attack_logit', 'query': 'query_logit', 'correction': 'correction_logit'}


def select_train_indices(rows):
    require({'image_id', 'role', 'label'} <= set(rows) and rows.role.eq('train').all(),
            'explicit source TRAIN rows required')
    require(rows.image_id.is_unique and rows.image_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all(),
            'unique nonempty training IDs required')
    require(rows.label.isin([0, 1]).all(), 'binary training labels required')
    source = rows.reset_index(drop=True)
    groups = [source.loc[source.label == label].sort_values('image_id').index.tolist()[:64] for label in (0, 1)]
    require(all(len(group) == 64 for group in groups), '64 genuine and64 altered training rows required')
    return [index for pair in zip(*groups) for index in pair]


def _rng_snapshot(loader):
    private, generators, seen = [], [], set()
    dataset = getattr(loader, 'dataset', None)
    while dataset is not None and id(dataset) not in seen:
        seen.add(id(dataset))
        for name in ('geometry_rng', 'photometry_rng'):
            value = getattr(dataset, name, None)
            if isinstance(value, random.Random):
                private.append((value, value.getstate()))
        dataset = getattr(dataset, 'dataset', None)
    for owner in (loader, getattr(loader, 'sampler', None)):
        generator = getattr(owner, 'generator', None)
        if isinstance(generator, torch.Generator) and all(generator is not entry[0] for entry in generators):
            generators.append((generator, generator.get_state().clone()))
    return dict(python=random.getstate(), numpy=copy.deepcopy(np.random.get_state()),
        torch=torch.get_rng_state().clone(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        private=private, generators=generators)


def _restore(model, buffers, flags, rng):
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    for module, flag in flags:
        module.training = flag
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    if rng['cuda'] is not None:
        torch.cuda.set_rng_state_all(rng['cuda'])
    for value, state in rng['private']:
        value.setstate(state)
    for generator, state in rng['generators']:
        generator.set_state(state)


def _component_metrics(rows, key):
    def summarize(population):
        summary = driver.core.summarize_predictions(population)
        logits = torch.tensor([row[key] for row in population], dtype=torch.float32)
        labels = torch.tensor([row['label'] for row in population])
        summary['image_loss'] = float(driver.core._image_loss({'attack_logit': logits}, labels,
            track='D_running_pilot'))
        return summary
    predictions = []
    probabilities = torch.tensor([row[key] for row in rows], dtype=torch.float32).sigmoid().tolist()
    for row, probability in zip(rows, probabilities):
        predictions.append(dict(row, bonafide_score=1. - probability, attack_probability=probability))
    result = summarize(predictions)
    families = sorted({'genuine' if row['label'] == 0 else row['attack_family'] for row in rows})
    result['families'] = {family: summarize([row for row in predictions
        if ('genuine' if row['label'] == 0 else row['attack_family']) == family]) for family in families}
    return result


def diagnose_normalization(model, loader, device):
    driver.core._strict_fp32(model, device)
    require(model.reference_enabled is False, 'OFF model required')
    require(getattr(loader, 'batch_size', None) == 8 and getattr(loader, 'num_workers', None) == 0,
            'fixed batch8 single-process loader required')
    norms = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    require(norms and all(module.track_running_stats and module.running_mean is not None
                         and module.running_var is not None for module in norms), 'stored BN statistics required')
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    flags = [(module, module.training) for module in model.modules()]
    rng = _rng_snapshot(loader)
    parameters = driver.tensor_sha256(dict(model.named_parameters()))
    original_buffers = driver.tensor_sha256(dict(model.named_buffers()))
    reports = {}
    try:
        batches = list(loader)
        require(len(batches) == 16, 'exact128 TRAIN examples required')
        identifiers = []
        for batch in batches:
            require(len(batch['label']) == 8 and sorted(batch['label'].tolist()) == [0] * 4 + [1] * 4,
                    'each batch must contain four genuine and four altered examples')
            require(batch['meta']['role'] == ['train'] * 8, 'TRAIN-only diagnostic required')
            identifiers.extend(batch['meta']['image_id'])
        require(len(set(identifiers)) == 128, 'duplicate training IDs')
        for condition in ('stored_bn', 'batch_bn'):
            _restore(model, buffers, flags, rng)
            model.eval()
            if condition == 'batch_bn':
                for module in norms:
                    module.train()
            predictions = []
            with torch.no_grad():
                for batch in batches:
                    inputs, labels = driver.core._batch_inputs(model, batch, device)
                    outputs = driver.core._forward(model, inputs)
                    for index in range(8):
                        row = {name: str(batch['meta'][name][index]) for name in ('image_id', 'design_id', 'attack_family')}
                        row.update(label=int(labels[index]), context_available=bool(inputs['context_available'][index]),
                            **{name: float(outputs[name][index]) for name in COMPONENTS.values()})
                        predictions.append(row)
            changed = original_buffers != driver.tensor_sha256(dict(model.named_buffers()))
            require(condition != 'stored_bn' or not changed, 'stored-BN forward changed buffers')
            require(parameters == driver.tensor_sha256(dict(model.named_parameters())), 'diagnostic changed parameters')
            reports[condition] = dict(components={name: _component_metrics(predictions, key)
                for name, key in COMPONENTS.items()}, buffers_changed_temporarily=changed,
                n_context_available=sum(row['context_available'] for row in predictions), predictions=predictions)
    finally:
        _restore(model, buffers, flags, rng)
    require(original_buffers == driver.tensor_sha256(dict(model.named_buffers())), 'buffers not restored')
    return dict(n_images=128, image_ids=identifiers, batch_size=8, conditions=reports, state_restored=True,
        weights_updated=False, dropout_enabled=False, optimization_performed=False,
        metric_method='existing metrics and float32 CPU sigmoid/BCE on saved component logits; ties genuine',
        interpretation='correction alone is a residual diagnostic, not an independently trained classifier')


def load_selected_checkpoint(train_binding, context):
    train = driver._json(train_binding)
    require(all(type(train.get(key)) is type(value) and train[key] == value for key, value in
        dict(passed=True, phase='train', arm='off', diagnostic_only=False, epochs_completed=100).items()),
        'authenticated completed original OFF training receipt required')
    require(type(train.get('selected_epoch')) is int and 1 <= train['selected_epoch'] <= 100,
            'selected training epoch required')
    metadata = driver.json.loads(context.identities)
    require(context.contract.fold == 0 and metadata.get('protocol') == 'A' and metadata.get('fold') == 0,
            'original A0 diagnostic required')
    identities = train['identities']
    for name, expected in (('records_sha256', metadata['records_sha256']),
        ('split_sha256', metadata['split_sha256']), ('enrollment_sha256', metadata['enrollment_manifest_sha256'])):
        require(identities.get(name) == expected, 'checkpoint and diagnostic metadata differ')
    driver._bytes(train['checkpoint'])
    checkpoint = torch.load(train['checkpoint']['path'], map_location='cpu', weights_only=True)
    require(isinstance(checkpoint, dict), 'checkpoint envelope required')
    config = checkpoint['config']
    require(config.get('arm') == 'off' and config.get('reference_enabled') is False and config.get('fold') == 0,
            'original OFF A0 checkpoint required')
    checkpoint = driver.core.read_direct_reference_checkpoint(train['checkpoint']['path'], config, identities)
    driver._bytes(train['checkpoint'])
    require(checkpoint['phase'] == 'train' and checkpoint['epoch'] == train['selected_epoch'],
            'checkpoint is not the selected trained epoch')
    return train, checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'train-result', 'registration'):
        parser.add_argument('--' + name, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output', required=True)
    arguments = parser.parse_args(argv)
    try:
        driver.require_gpu_allocation()
        binding = dict(path=arguments.request, sha256=arguments.request_sha256)
        request, context, config, unused_output = driver.load_request(binding)
        output = driver._path(arguments.output)
        require(not output.exists(), 'new diagnostic JSON output required')
        train_binding = dict(path=arguments.train_result, sha256=arguments.train_result_sha256)
        train, checkpoint = load_selected_checkpoint(train_binding, context)
        registration = dict(path=arguments.registration, sha256=arguments.registration_sha256)
        table = driver.bridge.load_protocol_a_registration(context, registration, roles=['train', 'val'])
        for protected in (context.raw_root, driver._path(context.product_root),
                          driver._path(context.manifest['output_root']), Path(train_binding['path']).parent,
                          driver.ROOT, unused_output):
            require(not output.is_relative_to(protected) and not protected.is_relative_to(output),
                    'diagnostic output overlaps protected input/run')
        indices = select_train_indices(context.rows('train'))
        model = driver.fresh_model(request, False)
        model.initialization_provenance = copy.deepcopy(checkpoint['initialization_provenance'])
        driver.core._validate_model(model, checkpoint['config'], checkpoint['identities'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        dataset = driver.bridge.ProtocolAReferenceDataset(context, table, role='train', reference_enabled=False,
            augment=dict(config['augment'], enabled=False, seed=42), cache_bytes=3 * 1024**3)
        loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False, num_workers=0,
            generator=torch.Generator().manual_seed(42), drop_last=False)
        report = diagnose_normalization(model, loader, 'cuda')
        driver._json(train_binding)
        driver._bytes(train['checkpoint'])
        driver._bytes(registration)
        report.update(schema_version='direct_reference_normalization_diagnostic_v1', role='train',
            request=binding, train_result=train_binding, checkpoint=train['checkpoint'],
            trained_identities=checkpoint['identities'], trained_config=checkpoint['config'],
            diagnostic_source_sha256=request['source_sha256'], input_identities=driver.json.loads(context.identities),
            registration=registration, checkpoint_epoch=checkpoint['epoch'],
            limitations='Current A diagnostic inputs, not asserted historical OFF preprocessing parity; no VAL/TEST, optimization, persistent recalibration, or family decision.')
        driver.write(output, report)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'normalization diagnostic failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
