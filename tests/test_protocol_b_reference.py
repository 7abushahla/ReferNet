import copy
from dataclasses import FrozenInstanceError
import hashlib
import importlib
import json
from pathlib import Path
import random
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from test_protocol_b_enrollment import inventory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from refer_or_clear import protocol_b_contract as contract_api
from refer_or_clear.data import direct_reference as original
from refer_or_clear.data import enrollment
from refer_or_clear.data import protocol_b_enrollment as enrollment_api
from refer_or_clear.data.references import fold_of


def binding(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return binding(path)


@pytest.fixture
def api():
    assert (ROOT / 'src/refer_or_clear/data/protocol_b_reference.py').is_file(), 'Protocol B reference module missing'
    return importlib.import_module('refer_or_clear.data.protocol_b_reference')


def prepare_reference_inputs(api, inventory, tmp_path, monkeypatch):
    records = pd.DataFrame(inventory['records'])
    records['attack_family'] = ['text_copy' if int(label) else '' for label in records.label]
    records.to_csv(inventory['records_path'], index=False)
    split_rows = []
    for role, members in (('train', inventory['contract'].source_train), ('val', inventory['contract'].source_val),
                          ('test', inventory['contract'].target_test)):
        split_rows.extend(dict(image_id=row.image_id, design_id=row.design_id, component_id=row.component_id,
                               label=row.label, protocol='B', role=role) for row in members)
    split = tmp_path / 'split.csv'
    pd.DataFrame(split_rows).to_csv(split, index=False)
    reviewed = {0: binding(split)['sha256']}
    monkeypatch.setattr(contract_api, 'REVIEWED_SPLIT_SHA256', reviewed)
    monkeypatch.setattr(enrollment_api, 'REVIEWED_SPLIT_SHA256', reviewed)
    contract = contract_api.load_protocol_b_contract(inventory['records_path'], split, group=0,
                                                    records_sha256=binding(inventory['records_path'])['sha256'])
    target_binding = write_json(tmp_path / 'target.json', contract_api.reserve_target_enrollment(contract))
    target = contract_api.load_target_enrollment(contract, Path(target_binding['path']), expected_sha256=target_binding['sha256'])
    manifest = enrollment_api.make_protocol_b_enrollment_manifest(inventory['records_path'], contract, target, inventory['raw'])
    monkeypatch.setattr(enrollment, '_features', lambda image: ([], None))
    monkeypatch.setattr(enrollment, '_fit', lambda *args: (np.eye(3), 50))
    monkeypatch.setattr(original, '_features', lambda image: ([], None))
    monkeypatch.setattr(original, '_fit', lambda *args: (np.eye(3), 50))
    monkeypatch.setattr(api, '_features', lambda image: ([], None), raising=False)
    monkeypatch.setattr(api, '_fit', lambda *args: (np.eye(3), 50), raising=False)
    texture = np.full((64, 96, 3), (11, 66, 199), dtype=np.uint8)
    for row in records.to_dict('records'):
        assert cv2.imwrite(str(inventory['raw'] / row['path']), texture)
    built = enrollment_api.build_protocol_b_enrollment_products(manifest, inventory['raw'], tmp_path / 'products')
    return dict(records=inventory['records_path'], split=split, raw=inventory['raw'], contract=contract, target=target,
                target_binding=target_binding, enrollment_binding=binding(tmp_path / 'products/manifest.json'),
                manifest=built, tmp=tmp_path)


def clone_reference_inputs(template, destination):
    source = template['tmp']
    shutil.copytree(source, destination, dirs_exist_ok=True)

    def relocate(value):
        if isinstance(value, dict):
            return {relocate(key): relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str) and (value == str(source) or value.startswith(str(source) + '/')):
            return str(destination) + value[len(str(source)):]
        return value

    for path in destination.rglob('*.json'):
        document = json.loads(path.read_text())
        relocated = relocate(document)
        if relocated != document:
            write_json(path, relocated)
    manifest_path = destination / Path(template['enrollment_binding']['path']).relative_to(source)
    manifest = json.loads(manifest_path.read_text())
    for banks in manifest['banks'].values():
        for bank in banks.values():
            products = bank['products']
            for name, item in products.items():
                if name != 'manifest':
                    products[name] = binding(Path(item['path']))
            dependency_path = Path(products['manifest']['path'])
            dependency = json.loads(dependency_path.read_text())
            dependency['output_sha256'] = {name: item['sha256'] for name, item in products.items() if name != 'manifest'}
            products['manifest'] = write_json(dependency_path, dependency)
    target_binding = binding(destination / Path(template['target_binding']['path']).relative_to(source))
    assert target_binding['sha256'] == template['target_binding']['sha256']
    records = destination / template['records'].relative_to(source)
    split = destination / template['split'].relative_to(source)
    assert binding(records)['sha256'] == binding(template['records'])['sha256']
    assert binding(split)['sha256'] == binding(template['split'])['sha256']
    return dict(template, tmp=destination, raw=destination / template['raw'].relative_to(source),
                records=records, split=split, target_binding=target_binding,
                manifest=manifest, enrollment_binding=write_json(manifest_path, manifest))


@pytest.fixture(scope='session')
def reference_template(tmp_path_factory):
    root = tmp_path_factory.mktemp('protocol-b-reference-template')
    specimen = inventory.__wrapped__(root)
    module = importlib.import_module('refer_or_clear.data.protocol_b_reference')
    with pytest.MonkeyPatch.context() as patch:
        return prepare_reference_inputs(module, specimen, root, patch)


@pytest.fixture
def reference_inputs(api, reference_template, tmp_path, monkeypatch):
    reviewed = {0: reference_template['contract'].split_sha256}
    monkeypatch.setattr(contract_api, 'REVIEWED_SPLIT_SHA256', reviewed)
    monkeypatch.setattr(enrollment_api, 'REVIEWED_SPLIT_SHA256', reviewed)
    for module in (enrollment, original, api):
        monkeypatch.setattr(module, '_features', lambda image: ([], None), raising=False)
        monkeypatch.setattr(module, '_fit', lambda *args: (np.eye(3), 50), raising=False)
    return clone_reference_inputs(reference_template, tmp_path)


def context(api, fixture, **changes):
    arguments = dict(group=0, records_sha256=binding(fixture['records'])['sha256'],
                     target_binding=fixture['target_binding'], enrollment_binding=fixture['enrollment_binding'],
                     raw_root=fixture['raw'])
    arguments.update(changes)
    return api.load_reference_context(fixture['records'], fixture['split'], **arguments)


@pytest.fixture
def registered(api, reference_inputs):
    fixture = reference_inputs
    loaded = context(api, fixture)
    path = fixture['tmp'] / 'registration.json'
    table = api.build_protocol_b_registration(loaded, roles=('train', 'val'), output_path=path)
    return dict(fixture, context=loaded, table=table, table_binding=binding(path))


def dataset(api, fixture, *, role='train', enabled=False, cache_bytes=0, augment=None):
    return api.ProtocolBReferenceDataset(fixture['context'], fixture['table'], role=role,
                                        reference_enabled=enabled, augment=augment or {'enabled': False}, cache_bytes=cache_bytes)


def evaluation(fixture, **changes):
    checkpoints = {}
    for arm in ('off', 'on'):
        path = fixture['tmp'] / f'{arm}-synthetic.pt'
        path.write_bytes(f'synthetic {arm} checkpoint bytes'.encode())
        checkpoints[arm] = binding(path)
    value = dict(schema_version='protocol_b_evaluation_v1', protocol='B', group=0,
                 records_sha256=binding(fixture['records'])['sha256'], split_sha256=binding(fixture['split'])['sha256'],
                 target_enrollment_sha256=fixture['target_binding']['sha256'],
                 enrollment_manifest_sha256=fixture['enrollment_binding']['sha256'],
                 authorized_image_ids=sorted(fixture['target'].query_ids), source_validation_selected=True,
                 checkpoints=checkpoints)
    value.update(changes)
    return write_json(fixture['tmp'] / 'evaluation.json', value)


def test_protocol_b_reference_module_exists():
    path = Path(__file__).resolve().parents[1] / 'src/refer_or_clear/data/protocol_b_reference.py'
    assert path.is_file(), 'Protocol B reference module missing'


def test_context_authenticates_metadata_without_decoding_pixels(api, reference_inputs, monkeypatch):
    fixture = reference_inputs
    monkeypatch.setattr(cv2, 'imread', lambda *args, **kwargs: pytest.fail('context decoded pixels'))
    loaded = context(api, fixture)
    assert loaded.contract == fixture['contract'] and loaded.target == fixture['target']
    assert loaded.raw_root == fixture['raw'].resolve()
    assert loaded.manifest['schema_version'] == 'protocol_b_enrollment_v1'
    assert loaded.rows('train').image_id.tolist() == sorted(row.image_id for row in fixture['contract'].source_train)
    assert loaded.rows('val').image_id.tolist() == sorted(row.image_id for row in fixture['contract'].source_val)
    with pytest.raises(ValueError):
        loaded.rows('test')
    assert not set(fixture['target'].query_ids) & set(fixture['target'].support_ids + fixture['target'].excluded_ids)


def test_context_is_frozen_and_detaches_mutable_manifest_and_rows(api, reference_inputs):
    loaded = context(api, reference_inputs)
    original_manifest = loaded.manifest
    changed = loaded.manifest
    changed['banks'].clear()
    assert loaded.manifest == original_manifest
    rows = loaded.rows('train')
    identifier = rows.iloc[0].image_id
    rows.loc[0, 'image_id'] = 'caller-alias'
    assert loaded.rows('train').iloc[0].image_id == identifier
    with pytest.raises((FrozenInstanceError, AttributeError)):
        loaded.raw_root = reference_inputs['tmp']


@pytest.mark.parametrize('field', ['records_sha256', 'target_binding', 'enrollment_binding'])
def test_context_hash_failure_is_closed_before_pixel_decode(api, reference_inputs, monkeypatch, field):
    fixture = reference_inputs
    monkeypatch.setattr(cv2, 'imread', lambda *args, **kwargs: pytest.fail('unauthenticated context decoded pixels'))
    supplied = '0' * 64 if field == 'records_sha256' else dict(fixture[field], sha256='0' * 64)
    with pytest.raises(ValueError):
        context(api, fixture, **{field: supplied})


@pytest.mark.parametrize('group', [True, -1, 10, 0.0, '0'])
def test_context_rejects_invalid_group(api, reference_inputs, group):
    with pytest.raises(ValueError):
        context(api, reference_inputs, group=group)


def test_split_drift_and_unknown_role_are_rejected(api, reference_inputs):
    fixture = reference_inputs
    loaded = context(api, fixture)
    with pytest.raises(ValueError):
        loaded.rows('unknown')
    fixture['split'].write_bytes(fixture['split'].read_bytes() + b'\n')
    with pytest.raises(ValueError):
        context(api, fixture)


@pytest.mark.parametrize('roles', [('test',), ('train', 'test'), ('train', 'val', 'test'), ('unknown',), ()])
def test_training_context_cannot_prepare_test_or_combined_registration(api, reference_inputs, monkeypatch, roles):
    loaded = context(api, reference_inputs)
    monkeypatch.setattr(cv2, 'imread', lambda *args, **kwargs: pytest.fail('forbidden preparation decoded pixels'))
    with pytest.raises(ValueError):
        api.build_protocol_b_registration(loaded, roles=roles, output_path=reference_inputs['tmp'] / 'forbidden.json')


def test_registration_contains_exact_source_members_and_opposite_training_banks(api, registered):
    fixture = registered
    entries = fixture['table']['entries']
    expected = set(fixture['context'].rows('train').image_id) | set(fixture['context'].rows('val').image_id)
    assert set(entries) == expected
    assert not set(entries) & set(fixture['target'].query_ids + fixture['target'].support_ids + fixture['target'].excluded_ids)
    for row in fixture['context'].rows('train').to_dict('records'):
        entry = entries[row['image_id']]
        assert entry['bank'] == f"train_half{1 - fold_of(row['component_id'])}"
        bank = fixture['context'].manifest['banks'][row['design_id']][entry['bank']]
        assert row['component_id'] not in bank['component_ids']
    for identifier in fixture['context'].rows('val').image_id:
        assert entries[identifier]['bank'] == 'enrollment'
    loaded = api.load_protocol_b_registration(fixture['context'], fixture['table_binding'], roles=('train', 'val'))
    assert loaded['entries'] == entries


@pytest.mark.parametrize('failure', ['hash', 'unknown-id', 'missing-id', 'wrong-role', 'invalid-matrix'])
def test_registration_loader_rejects_hash_identity_role_and_matrix_changes(api, registered, monkeypatch, failure):
    fixture = registered
    supplied = dict(fixture['table_binding'])
    if failure == 'hash':
        supplied['sha256'] = '0' * 64
    else:
        table = copy.deepcopy(dict(fixture['table']))
        identifier = next(iter(table['entries']))
        if failure == 'unknown-id':
            table['entries']['unknown'] = dict(table['entries'][identifier], image_id='unknown')
        elif failure == 'missing-id':
            table['entries'].pop(identifier)
        elif failure == 'wrong-role':
            table['entries'][identifier]['role'] = 'test'
        else:
            table['entries'][identifier]['query_canvas_to_bank'] = [[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]]
        supplied = write_json(Path(supplied['path']), table)
    monkeypatch.setattr(cv2, 'imread', lambda *args, **kwargs: pytest.fail('invalid registration decoded pixels'))
    with pytest.raises(ValueError):
        api.load_protocol_b_registration(fixture['context'], supplied, roles=('train', 'val'))


def test_registration_never_overwrites_existing_output(api, reference_inputs):
    loaded = context(api, reference_inputs)
    path = reference_inputs['tmp'] / 'existing.json'
    path.write_text('preserve existing')
    with pytest.raises((ValueError, FileExistsError)):
        api.build_protocol_b_registration(loaded, roles=('train', 'val'), output_path=path)
    assert path.read_text() == 'preserve existing'


def test_dataset_role_is_required_and_test_is_blocked_without_authority(api, registered, monkeypatch):
    fixture = registered
    monkeypatch.setattr(cv2, 'imread', lambda *args, **kwargs: pytest.fail('unauthorized dataset decoded pixels'))
    with pytest.raises(TypeError):
        api.ProtocolBReferenceDataset(fixture['context'], fixture['table'], reference_enabled=False, augment={'enabled': False})
    for role in ('unknown', 'test'):
        with pytest.raises(ValueError):
            dataset(api, fixture, role=role)


def test_off_never_reads_median_and_on_preserves_query_geometry_zone_overlap(api, registered, monkeypatch):
    fixture = registered
    off = dataset(api, fixture)
    on = dataset(api, fixture, enabled=True)
    median_paths = {product['products']['median']['path'] for banks in fixture['manifest']['banks'].values() for product in banks.values()}
    original_read, original_bytes = cv2.imread, Path.read_bytes

    def forbid_median(path, flags=cv2.IMREAD_COLOR):
        assert str(path) not in median_paths, 'OFF decoded reference median'
        return original_read(path, flags)

    def forbid_median_bytes(path):
        assert str(path) not in median_paths, 'OFF read reference median bytes'
        return original_bytes(path)

    with monkeypatch.context() as guard:
        guard.setattr(cv2, 'imread', forbid_median)
        guard.setattr(Path, 'read_bytes', forbid_median_bytes)
        sample_off = off[0]
    sample_on = on[0]
    assert sample_off['query'].shape == (4, 512, 800)
    assert sample_on['reference'].shape == (3, 512, 800)
    assert sample_off.get('reference') is None
    assert sample_off['query'].dtype == torch.float32 and sample_off['overlap'].dtype == torch.float32
    assert torch.equal(sample_off['query'], sample_on['query'])
    assert torch.equal(sample_off['overlap'], sample_on['overlap'])
    assert sample_off['context_available'].item() == sample_on['context_available'].item()
    assert set(torch.unique(sample_on['overlap']).tolist()) <= {0., 1.}
    assert torch.all(sample_on['query'][3][sample_on['overlap'][0] == 0] == 0)


def test_tensor_geometry_matches_original_helpers_with_nonidentity_saved_registration(api, registered):
    fixture = registered
    row = fixture['context'].rows('train').iloc[0].to_dict()
    document = copy.deepcopy(dict(fixture['table']))
    angle = .02
    matrix = np.array([[np.cos(angle), -np.sin(angle), 7.],
                       [np.sin(angle), np.cos(angle), -5.], [0., 0., 1.]])
    document['entries'][row['image_id']]['query_canvas_to_bank'] = matrix.tolist()
    supplied = write_json(fixture['tmp'] / 'transformed-registration.json', document)
    table = api.load_protocol_b_registration(fixture['context'], supplied, roles=('train', 'val'))
    consumer = api.ProtocolBReferenceDataset(fixture['context'], table, role='train', reference_enabled=True,
                                             augment={'enabled': False}, cache_bytes=0)
    sample = consumer[0]
    entry = document['entries'][row['image_id']]
    raw = cv2.imread(str(fixture['raw'] / row['path']), cv2.IMREAD_COLOR)
    canvas, valid, unused_transform = original.to_canvas(raw)
    unused_target, masks = original._common(entry['products'], registration=False)
    inverse, overlap = original._overlap(valid, masks, matrix)
    zone = (original._warp(masks['zone'], inverse, True) > 0).astype(np.uint8) * overlap
    median = cv2.imread(entry['products']['median']['path'], cv2.IMREAD_COLOR)
    expected_reference = original._warp(median, inverse)
    expected_query = torch.from_numpy(np.ascontiguousarray(np.concatenate([
        original._normalize(canvas), zone[None].astype(np.float32)])))
    assert torch.equal(sample['query'], expected_query)
    assert torch.equal(sample['overlap'], torch.from_numpy(overlap[None].astype(np.float32)))
    assert torch.equal(sample['reference'], torch.from_numpy(np.ascontiguousarray(original._normalize(expected_reference))))
    assert sample['context_available'].item() is original._available(overlap)
    assert torch.equal(sample['query'][:3], torch.from_numpy(original._normalize(canvas)))


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('role', ['train', 'val'])
def test_cached_and_uncached_samples_are_exact_and_outputs_do_not_alias(api, registered, enabled, role):
    plain = dataset(api, registered, role=role, enabled=enabled)
    cached = dataset(api, registered, role=role, enabled=enabled, cache_bytes=64 * 1024 * 1024)
    for index in (0, 1, 0):
        expected, actual = plain[index], cached[index]
        assert expected.keys() == actual.keys()
        for name in expected:
            if isinstance(expected[name], torch.Tensor):
                assert torch.equal(expected[name], actual[name])
            else:
                assert expected[name] == actual[name]
    corrupted = cached[0]
    corrupted['query'].fill_(123.)
    corrupted['overlap'].zero_()
    if enabled:
        corrupted['reference'].zero_()
    assert torch.equal(plain[0]['query'], cached[0]['query'])
    assert torch.equal(plain[0]['overlap'], cached[0]['overlap'])


def test_augmented_cache_parity_preserves_external_and_internal_rng_streams(api, registered):
    augment = dict(preset='direct_reference_canvas_v1', enabled=True, seed=42)
    plain = dataset(api, registered, enabled=True, augment=augment)
    cached = dataset(api, registered, enabled=True, augment=augment, cache_bytes=64 * 1024 * 1024)
    python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    for index in (0, 1, 0):
        expected, actual = plain[index], cached[index]
        assert torch.equal(expected['query'], actual['query'])
        assert torch.equal(expected['reference'], actual['reference'])
        assert torch.equal(expected['overlap'], actual['overlap'])
        assert expected['meta'] == actual['meta']
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


@pytest.mark.parametrize('authority', ['records', 'table_binding'])
def test_changed_records_and_registration_still_fail_closed_after_cache_warmup(api, registered, authority):
    cached = dataset(api, registered, cache_bytes=64 * 1024 * 1024)
    cached[0]
    path = registered['records'] if authority == 'records' else Path(registered['table_binding']['path'])
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError):
        cached[0]


def test_explicit_evaluation_binds_exact_remaining_queries_and_keeps_tables_separate(api, reference_inputs):
    fixture = reference_inputs
    loaded = context(api, fixture, evaluation_binding=evaluation(fixture))
    assert loaded.rows('test').image_id.tolist() == sorted(fixture['target'].query_ids)
    table = api.build_protocol_b_registration(loaded, roles=('test',), output_path=fixture['tmp'] / 'test-registration.json')
    assert set(table['entries']) == set(fixture['target'].query_ids)
    assert all(entry['role'] == 'test' and entry['bank'] == 'enrollment' for entry in table['entries'].values())
    test_dataset = api.ProtocolBReferenceDataset(loaded, table, role='test', reference_enabled=False, augment={'enabled': False})
    assert len(test_dataset) == len(fixture['target'].query_ids)
    for roles in (('train', 'test'), ('train', 'val', 'test')):
        with pytest.raises(ValueError):
            api.build_protocol_b_registration(loaded, roles=roles, output_path=fixture['tmp'] / 'mixed.json')


@pytest.mark.parametrize('failure', ['schema', 'selected', 'extra-id', 'missing-id', 'group', 'checkpoint-hash', 'unpaired'])
def test_evaluation_rejects_unselected_unpaired_or_wrong_membership_before_pixels(api, reference_inputs, monkeypatch, failure):
    fixture = reference_inputs
    supplied = evaluation(fixture)
    value = json.loads(Path(supplied['path']).read_text())
    if failure == 'schema': value['schema_version'] = 'other'
    if failure == 'selected': value['source_validation_selected'] = False
    if failure == 'extra-id': value['authorized_image_ids'].append(fixture['target'].support_ids[0])
    if failure == 'missing-id': value['authorized_image_ids'].pop()
    if failure == 'group': value['group'] = False
    if failure == 'checkpoint-hash': value['checkpoints']['off']['sha256'] = '0' * 64
    if failure == 'unpaired': value['checkpoints'].pop('on')
    supplied = write_json(Path(supplied['path']), value)
    monkeypatch.setattr(cv2, 'imread', lambda *args, **kwargs: pytest.fail('bad evaluation decoded pixels'))
    with pytest.raises(ValueError):
        context(api, fixture, evaluation_binding=supplied)


def test_template_clone_is_independent_and_rehashes_relocated_dependencies(tmp_path, monkeypatch):
    root = tmp_path / 'template'
    root.mkdir()
    raw = root / 'raw'
    raw.mkdir()
    (raw / 'query.png').write_bytes(b'synthetic query bytes')
    (raw / 'layout.json').write_text('{"z":1,"a":2}')
    records, split = root / 'records.csv', root / 'split.csv'
    records.write_text('unchanged records bytes')
    split.write_text('unchanged split bytes')
    target = write_json(root / 'target.json', dict(support_ids=['support'], query_ids=['query']))
    products = root / 'products'
    bank_root = products / 'group0/sidtd/example/enrollment'
    bank_root.mkdir(parents=True)
    (bank_root / 'target.png').write_bytes(b'synthetic product bytes')
    geometry = dict(raw_root=str(raw), source_records=str(records))
    bindings = dict(target=binding(bank_root / 'target.png'), geometry=write_json(bank_root / 'geometry.json', geometry))
    dependency = dict(geometry, output_sha256={name: item['sha256'] for name, item in bindings.items()})
    bindings['manifest'] = write_json(bank_root / 'manifest.json', dependency)
    manifest = dict(raw_root=str(raw), output_root=str(products), records_authority=binding(records),
                    target_enrollment_sha256=target['sha256'], banks={'sidtd/example': {'enrollment': dict(products=bindings)}})
    manifest_binding = write_json(products / 'manifest.json', manifest)
    template = dict(tmp=root, records=records, split=split, raw=raw, target_binding=target,
                    enrollment_binding=manifest_binding, manifest=manifest, contract='unchanged contract', target='unchanged reservation')
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob('*') if path.is_file()}
    monkeypatch.setattr(enrollment_api, 'build_protocol_b_enrollment_products', lambda *args, **kwargs: pytest.fail('clone rebuilt banks'))
    cloned = clone_reference_inputs(template, tmp_path / 'clone')
    relocated = cloned['manifest']['banks']['sidtd/example']['enrollment']['products']
    for item in relocated.values():
        assert Path(item['path']).is_relative_to(cloned['tmp'])
        assert binding(Path(item['path'])) == item
    cloned_geometry = json.loads(Path(relocated['geometry']['path']).read_text())
    assert cloned_geometry['raw_root'] == str(cloned['raw'])
    assert cloned_geometry['source_records'] == str(cloned['records'])
    assert relocated['geometry']['sha256'] != bindings['geometry']['sha256']
    cloned_dependency = json.loads(Path(relocated['manifest']['path']).read_text())
    assert cloned_dependency['output_sha256'] == {name: item['sha256'] for name, item in relocated.items() if name != 'manifest'}
    assert cloned['enrollment_binding'] == binding(Path(cloned['enrollment_binding']['path']))
    assert cloned['target_binding']['sha256'] == target['sha256']
    assert cloned['records'].read_bytes() == records.read_bytes()
    assert (cloned['raw'] / 'query.png').read_bytes() == (raw / 'query.png').read_bytes()
    assert (cloned['raw'] / 'layout.json').read_bytes() == (raw / 'layout.json').read_bytes()
    assert (cloned['raw'] / 'query.png').stat().st_ino != (raw / 'query.png').stat().st_ino
    cloned['records'].write_text('mutation isolated to clone')
    assert all((root / relative).read_bytes() == content for relative, content in before.items())
