import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader

from test_direct_reference_train import service, setup
from refer_or_clear import direct_reference_pilot as core
from refer_or_clear.data import direct_reference as bridge
from refer_or_clear.pilots import _selection_record, _write_csv, json_sha256, tensor_sha256
from refer_or_clear.train import selection_key


def module():
    assert importlib.util.find_spec('refer_or_clear.direct_reference_control') is not None, 'OFF control CLI missing'
    return importlib.import_module('refer_or_clear.direct_reference_control')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o644)
    path.write_text(json.dumps(value, allow_nan=False))
    return bridge_binding(path)


def bridge_binding(path):
    from refer_or_clear.direct_reference_diagnostic import bind
    return bind(path)


@pytest.fixture
def authority(setup):
    context = setup['context']
    config_path = setup['tmp'] / 'config.yaml'
    config_path.write_bytes((Path(__file__).resolve().parents[1] / 'configs/direct_reference_pilot.yaml').read_bytes())
    context['manifest']['authorities']['config'] = bridge_binding(config_path)
    value = dict(schema_version='direct_reference_control_request_v1',
                 source_sha256=context['identities']['source_sha256'],
                 config_sha256=bridge_binding(config_path)['sha256'],
                 diagnostic_request_sha256=context['request']['sha256'], arm='off', epochs=100, fold=0, seed=42,
                 diagnostic_gate=setup['gate_binding'])
    return dict(setup, control=value, control_binding=write(setup['tmp'] / 'control-request.json', value))


@pytest.fixture
def artifacts(service, authority, monkeypatch):
    context = authority['context']
    prepared = service.prepare_training(context, authority['gate_binding'])
    directory = context['run_dir'] / 'training/off'
    directory.mkdir()
    config = dict(context['config'], arm='off', reference_enabled=False)
    model = service._fresh_model(context, False, authority['gate'])
    identities = service._checkpoint_identities(context, config, prepared['preprocess'], prepared['table_binding'],
                                                json_sha256(model.initialization_provenance))
    samples = []
    for row in authority['rows']['val'].to_dict('records'):
        query = torch.zeros(4, 32, 32)
        query[0].fill_(2. if row['label'] else -2.)
        samples.append(dict(query=query, overlap=torch.ones(1, 32, 32), context_available=torch.tensor(False),
                            label=row['label'], meta={key: row[key] for key in ('image_id', 'design_id', 'attack_family')}))
    summary, predictions = core.evaluate_direct_reference(model, DataLoader(samples, batch_size=1), 'cpu')
    key = selection_key('accuracy', summary['accuracy'], summary['auc'], summary['image_loss'], .2)
    core.save_direct_reference_checkpoint(directory / 'best.pt', model, config, identities, 1, key, 'train')
    _write_csv(directory / 'val_predictions.csv', predictions)
    history = [dict(epoch=epoch, loss=.2, image_loss=.2, n_images=1776, optimizer_steps=222,
                    val_accuracy=summary['accuracy'], val_auc=summary['auc'], val_image_loss=summary['image_loss'])
               for epoch in range(1, 101)]
    _write_csv(directory / 'log.csv', history)
    result = dict(passed=True, phase='train', diagnostic_only=False, epochs_completed=100, selected_epoch=1,
                  optimizer_steps=22200, role='val', arm='off', identities=identities, selection_key=_selection_record(key),
                  scheduler=dict(total_updates=22200, warmup_updates=1000),
                  fresh_initial_state_sha256=tensor_sha256(model.state_dict()),
                  checkpoint=bridge_binding(directory / 'best.pt'), predictions=bridge_binding(directory / 'val_predictions.csv'),
                  registration=prepared['table_binding'], diagnostic_gate=authority['gate_binding'],
                  replay=dict(passed=True, fresh_process=False, n_images=222, phase='train'))
    binding = write(directory / 'train.json', result)
    write(directory / 'replay.json', result['replay'])
    def dataset(context, prepared, selected, enabled, augmented=False):
        authority['events'].append('replay-dataset')
        assert not enabled and not augmented
        assert selected.image_id.tolist() == authority['rows']['val'].image_id.tolist()
        return samples
    monkeypatch.setattr(service, '_dataset', dataset)
    authority['events'].clear()
    return dict(authority, result=result, result_binding=binding, config=config, predictions=predictions,
                history=history, directory=directory)


@pytest.mark.parametrize('field,value', [
    ('schema_version', 'direct_reference_diagnostic_request_v1'), ('arm', 'on'), ('epochs', 1),
    ('epochs', 100.0), ('fold', False), ('seed', 41), ('source_sha256', '0'*64),
    ('config_sha256', '0'*64), ('diagnostic_request_sha256', '0'*64), ('resume', 'best.pt')])
def test_control_authority_rejects_before_pixels(authority, field, value):
    control = module()
    document = dict(authority['control'], **{field: value})
    binding = write(Path(authority['control_binding']['path']), document)
    with pytest.raises(ValueError):
        control.authorize_control(authority['context'], binding)
    assert authority['events'] == []


@pytest.mark.parametrize('failure', ['control-hash', 'gate-hash', 'old-source', 'config-json-hash'])
def test_authenticated_authorities_not_self_asserted(authority, failure):
    control = module()
    document = copy.deepcopy(authority['control'])
    if failure == 'gate-hash':
        document['diagnostic_gate']['sha256'] = '0'*64
    elif failure == 'old-source':
        gate = copy.deepcopy(authority['gate'])
        gate['identities']['source_sha256'] = '0'*64
        document['diagnostic_gate'] = write(Path(authority['gate_binding']['path']), gate)
    elif failure == 'config-json-hash':
        document['config_sha256'] = json_sha256(authority['context']['config'])
    binding = write(Path(authority['control_binding']['path']), document)
    if failure == 'control-hash':
        binding['sha256'] = '0'*64
    with pytest.raises(ValueError):
        control.authorize_control(authority['context'], binding)
    assert authority['events'] == []


def test_replay_selected_complete_population_and_new_checkpoint(artifacts):
    control = module()
    result = control.replay_control(artifacts['context'], artifacts['control_binding'], artifacts['result_binding'])
    assert result['passed'] is True and result['fresh_process'] is True
    assert result['checkpoint'] == artifacts['result']['checkpoint']
    assert result['identities'] == artifacts['result']['identities']
    assert result['image_ids'] == artifacts['rows']['val'].image_id.tolist()
    assert result['n_images'] == 222 and result['buffers_unchanged'] is True
    assert result['max_abs_difference'] == 0.
    assert result['train_result'] == artifacts['result_binding']
    assert result['control_request'] == artifacts['control_binding']
    assert json.loads((artifacts['directory'] / 'replay.json').read_text())['fresh_process'] is False


@pytest.mark.parametrize('failure', ['hash', 'arm', 'identity', 'phase', 'history-short', 'late-tie',
                                    'csv-short', 'csv-order', 'csv-metadata', 'checkpoint-smoke', 'table-subset', 'wrong-path',
                                    'csv-score', 'history-metrics'])
def test_replay_rejects_before_dataset_iteration(artifacts, failure):
    control = module()
    result = copy.deepcopy(artifacts['result'])
    if failure == 'arm': result['arm'] = 'on'
    if failure == 'identity': result['identities']['config_sha256'] = '0'*64
    if failure == 'phase': result['phase'] = 'smoke'
    if failure == 'late-tie': result['selected_epoch'] = 2
    if failure == 'history-short': _write_csv(artifacts['directory'] / 'log.csv', artifacts['history'][:-1])
    if failure.startswith('csv-'):
        predictions = copy.deepcopy(artifacts['predictions'])
        if failure == 'csv-short': predictions.pop()
        if failure == 'csv-order': predictions.reverse()
        if failure == 'csv-metadata': predictions[0]['label'] = 1
        if failure == 'csv-score': predictions[0]['bonafide_score'] = .1
        _write_csv(artifacts['directory'] / 'val_predictions.csv', predictions)
        result['predictions'] = bridge_binding(artifacts['directory'] / 'val_predictions.csv')
    if failure == 'checkpoint-smoke':
        checkpoint = torch.load(result['checkpoint']['path'], weights_only=True)
        checkpoint['phase'] = 'smoke'
        torch.save(checkpoint, result['checkpoint']['path'])
        result['checkpoint'] = bridge_binding(Path(result['checkpoint']['path']))
    if failure == 'table-subset':
        table = bridge._json(result['registration'])
        table['entries'].pop(next(iter(table['entries'])))
        result['registration'] = write(Path(result['registration']['path']), table)
    if failure == 'wrong-path':
        other = artifacts['directory'] / 'other.pt'
        other.write_bytes(Path(result['checkpoint']['path']).read_bytes())
        result['checkpoint'] = bridge_binding(other)
    if failure == 'history-metrics':
        history = copy.deepcopy(artifacts['history'])
        for entry in history: entry['val_accuracy'] = .5
        _write_csv(artifacts['directory'] / 'log.csv', history)
        result['selection_key'] = [.5, 1.]
        checkpoint = torch.load(result['checkpoint']['path'], weights_only=True)
        checkpoint['selection_key'] = [.5, 1.]
        torch.save(checkpoint, result['checkpoint']['path'])
        result['checkpoint'] = bridge_binding(Path(result['checkpoint']['path']))
    binding = write(Path(artifacts['result_binding']['path']), result)
    if failure == 'hash': binding['sha256'] = '0'*64
    with pytest.raises(ValueError):
        control.replay_control(artifacts['context'], artifacts['control_binding'], binding)
    assert 'replay-dataset' not in artifacts['events']


@pytest.mark.parametrize('failure', [None, 'flag-only', 'identity', 'checkpoint', 'ids', 'buffers', 'difference', 'pid', 'summary'])
def test_parent_requires_full_fresh_child_receipt(artifacts, service, monkeypatch, failure):
    control = module()
    monkeypatch.setattr(service, 'run_training', lambda context, enabled, gate: artifacts['result'] if enabled is False else pytest.fail('ON exposed'))
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        report = control.replay_control(artifacts['context'], artifacts['control_binding'], artifacts['result_binding'])
        report['pid'] += 1
        if failure == 'flag-only': report = dict(passed=True)
        if failure == 'identity': report['identities'] = {}
        if failure == 'checkpoint': report['checkpoint'] = dict(report['checkpoint'], sha256='0'*64)
        if failure == 'ids': report['image_ids'].reverse()
        if failure == 'buffers': report['buffers_unchanged'] = False
        if failure == 'difference': report['max_abs_difference'] = float('inf')
        if failure == 'pid': report['pid'] -= 1
        if failure == 'summary': report['summary']['accuracy'] = .5
        path = artifacts['directory'] / 'fresh-replay.json'
        path.chmod(0o644)
        path.write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(control.subprocess, 'run', child)
    if failure is None:
        result = control.run_control(artifacts['context'], artifacts['control_binding'])
        assert result['passed'] is True and result['arm'] == 'off'
    else:
        with pytest.raises(ValueError): control.run_control(artifacts['context'], artifacts['control_binding'])
    assert calls[0][0] == sys.executable
    assert Path(calls[0][1]).name == 'train_direct_reference_control.py'
    assert calls[0][calls[0].index('--phase')+1] == 'replay'
    assert calls[0][calls[0].index('--train-result-sha256')+1] == artifacts['result_binding']['sha256']


@pytest.mark.parametrize('extra', [[], ['--epochs','1'], ['--arm','on'], ['--resume','best.pt'], ['--phase','replay']])
def test_cli_requires_control_and_external_replay_binding(extra):
    control = module()
    with pytest.raises(SystemExit):
        control.main(['--request','/request','--request-sha256','a'*64,'--run-dir','/run',*extra])


def test_script_entrypoint_exists():
    assert (Path(__file__).resolve().parents[1] / 'scripts/train_direct_reference_control.py').is_file()


@pytest.mark.parametrize('phase', ['train', 'replay'])
def test_cli_dispatches_only_off_service_and_authenticated_binding(authority, service, monkeypatch, phase):
    control = module()
    calls = []
    monkeypatch.setattr(control, 'run_control', lambda context, binding: calls.append((context, binding)) or dict(passed=True))
    monkeypatch.setattr(control, 'replay_control', lambda context, binding, train: calls.append((context, binding, train)) or dict(passed=True))
    arguments = ['--request', authority['context']['request']['path'], '--request-sha256', authority['context']['request']['sha256'],
                 '--control-request', authority['control_binding']['path'], '--control-request-sha256', authority['control_binding']['sha256'],
                 '--run-dir', str(authority['tmp']), '--phase', phase]
    if phase == 'replay': arguments += ['--train-result', '/external/train.json', '--train-result-sha256', 'f'*64]
    assert control.main(arguments)['passed'] is True
    assert calls[0][:2] == (authority['context'], authority['control_binding'])
    if phase == 'replay': assert calls[0][2] == dict(path='/external/train.json', sha256='f'*64)


def test_script_help_rejects_unapproved_options():
    script = Path(__file__).resolve().parents[1] / 'scripts/train_direct_reference_control.py'
    assert subprocess.run([sys.executable, str(script), '--help'], capture_output=True).returncode == 0
    assert subprocess.run([sys.executable, str(script), '--arm', 'on'], capture_output=True).returncode != 0
