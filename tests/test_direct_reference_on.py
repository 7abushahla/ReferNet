import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from test_direct_reference_control import service, setup, authority, artifacts, write, bridge_binding
from refer_or_clear import direct_reference_control as off
from refer_or_clear import direct_reference_pilot as core
from refer_or_clear import pilots
from refer_or_clear.data import direct_reference as bridge
from refer_or_clear.train import selection_key


def launcher():
    path = Path(__file__).resolve().parents[1] / 'scripts/train_direct_reference_on.py'
    assert path.is_file(), 'external ON launcher missing'
    spec = importlib.util.spec_from_file_location('_on_launcher_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def revise_scores(ready, failure):
    predictions = copy.deepcopy(ready['predictions'])
    changed = 0
    for prediction in predictions:
        if failure == 'accuracy' and prediction['label'] == 0 or failure == 'auc' and prediction['label'] == 1 and changed < 2:
            logit = .4 if failure == 'accuracy' else -4.
            probability = torch.tensor(logit).sigmoid().item()
            prediction.update(attack_logit=logit, query_logit=logit, attack_probability=probability, bonafide_score=1.-probability)
            changed += 1
    summary = pilots.summarize_predictions(predictions)
    key = selection_key('accuracy', summary['accuracy'], summary['auc'], .2, .2)
    history = [dict(entry,val_accuracy=summary['accuracy'],val_auc=summary['auc']) for entry in ready['history']]
    pilots._write_csv(ready['directory'] / 'log.csv',history)
    pilots._write_csv(ready['directory'] / 'val_predictions.csv',predictions)
    checkpoint = torch.load(ready['result']['checkpoint']['path'],weights_only=True)
    checkpoint['selection_key'] = pilots._selection_record(key)
    torch.save(checkpoint,ready['result']['checkpoint']['path'])
    result = dict(ready['result'],selection_key=pilots._selection_record(key),
                  predictions=bridge_binding(ready['directory'] / 'val_predictions.csv'),
                  checkpoint=bridge_binding(ready['directory'] / 'best.pt'))
    train_binding = write(ready['directory'] / 'train.json',result)
    fresh = dict(ready['fresh'],train_result=train_binding,checkpoint=result['checkpoint'],
                 history=bridge_binding(ready['directory'] / 'log.csv'),predictions=predictions,
                 summary=dict(summary,buffers_unchanged=True),replay=pilots.compare_replay(predictions,predictions))
    completion = dict(ready['completion'],train_result=train_binding,checkpoint=result['checkpoint'],
                      fresh_replay=write(ready['directory'] / 'fresh-replay.json',fresh))
    return write(ready['directory'] / 'control.json',completion)


@pytest.fixture
def completed(artifacts):
    predictions = copy.deepcopy(artifacts['predictions'])
    report = dict(schema_version='direct_reference_control_replay_v1', passed=True, fresh_process=True,
                  phase='train', arm='off', pid=987654, n_images=222, buffers_unchanged=True,
                  train_result=artifacts['result_binding'], control_request=artifacts['control_binding'],
                  checkpoint=artifacts['result']['checkpoint'], identities=artifacts['result']['identities'],
                  history=bridge_binding(artifacts['directory'] / 'log.csv'),
                  image_ids=artifacts['rows']['val'].image_id.tolist(), predictions=predictions,
                  replay=pilots.compare_replay(predictions, predictions), max_abs_difference=0.,
                  summary=dict(pilots.summarize_predictions(predictions), buffers_unchanged=True))
    replay_binding = write(artifacts['directory'] / 'fresh-replay.json', report)
    completion = dict(schema_version='direct_reference_control_result_v1', passed=True, arm='off', phase='train',
                      epochs_completed=100, train_result=artifacts['result_binding'], control_request=artifacts['control_binding'],
                      checkpoint=artifacts['result']['checkpoint'], identities=artifacts['result']['identities'], fresh_replay=replay_binding)
    binding = write(artifacts['directory'] / 'control.json', completion)
    return dict(artifacts, completion=completion, completion_binding=binding, fresh=report)


@pytest.fixture
def ready(completed, service, monkeypatch):
    module = launcher()
    monkeypatch.setattr(module, 'SCIENTIFIC_SHA256', completed['context']['identities']['source_sha256'])
    bundle = SimpleNamespace(service=service, bridge=bridge, core=core, off=off, pilots=pilots,
                             selection_key=selection_key, torch=torch, DataLoader=DataLoader)
    document = dict(schema_version='direct_reference_on_request_v1', arm='on', epochs=100, fold=0, seed=42,
                    source_sha256=completed['control']['source_sha256'], config_sha256=completed['control']['config_sha256'],
                    diagnostic_request_sha256=completed['control']['diagnostic_request_sha256'],
                    launcher_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                    off_completion=completed['completion_binding'], registration=completed['result']['registration'],
                    phase_authority=bridge_binding(completed['context']['run_dir'] / 'training/shared/phase-authority.json'))
    binding = write(completed['tmp'] / 'on-request.json', document)
    return dict(completed, module=module, bundle=bundle, on=document, on_binding=binding)


@pytest.mark.parametrize('field,value', [('schema_version','other'), ('arm','off'), ('epochs',99), ('epochs',100.0),
                                       ('fold',False), ('seed',41), ('source_sha256','0'*64),
                                       ('config_sha256','0'*64), ('diagnostic_request_sha256','0'*64),
                                       ('launcher_sha256','0'*64), ('resume','selected.pt')])
def test_on_authority_is_explicit_fixed_and_separate(ready, field, value):
    document = dict(ready['on'], **{field:value})
    binding = write(Path(ready['on_binding']['path']), document)
    with pytest.raises(ValueError):
        ready['module'].authorize(ready['bundle'], ready['context'], binding)
    assert ready['events'] == []


@pytest.mark.parametrize('failure', ['off-hash','off-incomplete','flag-only','replay-failed','replay-ids','replay-identity',
                                    'table-hash','phase-hash','phase-path','existing-on','accuracy','auc','checkpoint-legacy'])
def test_off_gate_and_exact_shared_products_fail_before_model_or_dataset(ready, failure):
    document = copy.deepcopy(ready['on'])
    if failure == 'off-hash': document['off_completion']['sha256'] = '0'*64
    if failure in ('off-incomplete','flag-only'):
        completion = dict(ready['completion'], epochs_completed=99) if failure == 'off-incomplete' else dict(passed=True)
        document['off_completion'] = write(Path(document['off_completion']['path']), completion)
    if failure.startswith('replay-'):
        report = copy.deepcopy(ready['fresh'])
        if failure == 'replay-failed': report['passed'] = False
        if failure == 'replay-ids': report['image_ids'].reverse()
        if failure == 'replay-identity': report['identities'] = {}
        completion = dict(ready['completion'], fresh_replay=write(ready['directory'] / 'fresh-replay.json', report))
        document['off_completion'] = write(Path(document['off_completion']['path']), completion)
    if failure == 'table-hash': document['registration']['sha256'] = '0'*64
    if failure == 'phase-hash': document['phase_authority']['sha256'] = '0'*64
    if failure == 'phase-path':
        document['phase_authority'] = write(ready['tmp'] / 'other-phase.json', bridge._json(document['phase_authority']))
    if failure == 'existing-on': (ready['context']['run_dir'] / 'training/on').mkdir()
    if failure in ('accuracy','auc'):
        document['off_completion'] = revise_scores(ready,failure)
    if failure == 'checkpoint-legacy':
        checkpoint = torch.load(ready['result']['checkpoint']['path'], weights_only=True)
        checkpoint['architecture_id'] = 'anchor_direct_running_ia18_pilot_v1'
        torch.save(checkpoint, ready['result']['checkpoint']['path'])
        train = dict(ready['result'], checkpoint=bridge_binding(Path(ready['result']['checkpoint']['path'])))
        train_binding = write(Path(ready['result_binding']['path']), train)
        completion = dict(ready['completion'], train_result=train_binding, checkpoint=train['checkpoint'])
        document['off_completion'] = write(Path(document['off_completion']['path']), completion)
    binding = write(Path(ready['on_binding']['path']), document)
    with pytest.raises(ValueError): ready['module'].authorize(ready['bundle'], ready['context'], binding)
    assert not any(event == 'replay-dataset' or isinstance(event,tuple) and event[0] == 'fresh-model' for event in ready['events'])


def test_metadata_authority_reuses_exact_table_without_fresh_model(ready):
    result = ready['module'].authorize(ready['bundle'], ready['context'], ready['on_binding'])
    assert result['prepared']['table_binding'] == ready['result']['registration']
    assert result['prepared']['preprocess']['phase_authority'] == ready['on']['phase_authority']
    assert ready['events'] == []


@pytest.mark.parametrize('failure',['schema','phase','summary','pid','referral'])
def test_fresh_replay_is_complete_evidence_not_flags(ready,failure):
    module = ready['module']
    authorized = module.authorize(ready['bundle'],ready['context'],ready['on_binding'])
    chosen = module.selected(ready['bundle'],ready['context'],authorized,ready['result_binding'],'off')
    report = copy.deepcopy(ready['fresh'])
    if failure == 'schema': report['schema_version'] = 'other'
    if failure == 'phase': report['phase'] = 'smoke'
    if failure == 'summary': report['summary']['accuracy'] = .1
    if failure == 'pid': report['pid'] = False
    if failure == 'referral': report['predictions'][0]['referral_required'] = False
    with pytest.raises(ValueError): module.check_replay(ready['bundle'],report,chosen,ready['result_binding'],ready['control_binding'],'off')


@pytest.mark.parametrize('failure',['scheduler','fresh-state','score'])
def test_selected_metadata_binds_recipe_initial_state_and_scores(ready,failure):
    module = ready['module']
    authorized = module.authorize(ready['bundle'],ready['context'],ready['on_binding'])
    result = copy.deepcopy(ready['result'])
    if failure == 'scheduler': result['scheduler'] = dict(total_updates=100,warmup_updates=1)
    if failure == 'fresh-state': result['fresh_initial_state_sha256'] = '0'*64
    if failure == 'score':
        predictions = copy.deepcopy(ready['predictions'])
        predictions[0]['bonafide_score'] = .999
        pilots._write_csv(ready['directory'] / 'val_predictions.csv',predictions)
        result['predictions'] = bridge_binding(ready['directory'] / 'val_predictions.csv')
    binding = write(ready['directory'] / 'train.json',result)
    with pytest.raises(ValueError): module.selected(ready['bundle'],ready['context'],authorized,binding,'off')


def test_on_child_must_record_the_authorized_external_launcher_digest(ready):
    module = ready['module']
    authorized = module.authorize(ready['bundle'],ready['context'],ready['on_binding'])
    chosen = module.selected(ready['bundle'],ready['context'],authorized,ready['result_binding'],'off')
    report = dict(ready['fresh'],schema_version='direct_reference_on_replay_v1',arm='on',
                  on_request=ready['on_binding'],launcher_sha256='0'*64)
    with pytest.raises(ValueError): module.check_replay(ready['bundle'],report,chosen,ready['result_binding'],ready['on_binding'],'on')


def on_products(ready, prepared, monkeypatch):
    bundle, context = ready['bundle'], ready['context']
    model = bundle.service._fresh_model(context,True,ready['gate'])
    config = dict(context['config'],arm='on',reference_enabled=True)
    identities = bundle.service._checkpoint_identities(context,config,prepared['preprocess'],prepared['table_binding'],
                                                       pilots.json_sha256(model.initialization_provenance))
    samples = []
    for row in ready['rows']['val'].to_dict('records'):
        query = torch.zeros(4,32,32)
        query[0].fill_(2. if row['label'] else -2.)
        samples.append(dict(query=query,overlap=torch.ones(1,32,32),context_available=torch.tensor(False),
                            reference=torch.zeros(3,32,32),label=row['label'],
                            meta={key:row[key] for key in ('image_id','design_id','attack_family')}))
    def dataset(context, supplied, selected, enabled, augmented=False):
        ready['events'].append('on-dataset')
        assert enabled is True and augmented is False and len(selected) == 222
        assert supplied['table_binding'] == prepared['table_binding']
        return samples
    monkeypatch.setattr(bundle.service,'_dataset',dataset)
    summary,predictions = core.evaluate_direct_reference(model,DataLoader(samples,batch_size=1),'cpu')
    directory = context['run_dir'] / 'training/on'
    directory.mkdir()
    key = selection_key('accuracy',summary['accuracy'],summary['auc'],summary['image_loss'],.2)
    core.save_direct_reference_checkpoint(directory / 'best.pt',model,config,identities,1,key,'train')
    pilots._write_csv(directory / 'val_predictions.csv',predictions)
    pilots._write_csv(directory / 'log.csv',ready['history'])
    result = dict(ready['result'],arm='on',identities=identities,selection_key=pilots._selection_record(key),
                  checkpoint=bridge_binding(directory / 'best.pt'),predictions=bridge_binding(directory / 'val_predictions.csv'))
    write(directory / 'train.json',result)
    write(directory / 'replay.json',result['replay'])
    return result


def test_train_passes_prepared_to_original_on_service_only(ready, monkeypatch):
    module = ready['module']
    before = {path:path.read_bytes() for path in (ready['context']['run_dir'] / 'training').rglob('*') if path.is_file()}
    invoked = []
    def training(context, enabled, gate, *, prepared):
        invoked.append((enabled, gate, prepared['table_binding']))
        assert enabled is True and prepared['table_binding'] == ready['on']['registration']
        return on_products(ready,prepared,monkeypatch)
    monkeypatch.setattr(ready['bundle'].service, 'run_training', training)
    def child(command, **kwargs):
        assert command[0] == sys.executable and Path(command[1]) == Path(module.__file__)
        assert command[command.index('--phase')+1] == 'replay'
        result_binding = bridge_binding(ready['context']['run_dir'] / 'training/on/train.json')
        report = module.replay(ready['bundle'],ready['context'],ready['on_binding'],result_binding)
        report['pid'] = os.getpid()+1
        write(ready['context']['run_dir'] / 'training/on/fresh-replay.json',report)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(module.subprocess, 'run', child)
    args = SimpleNamespace(source_dir=Path('/restored/code'), request=Path('/restored/request.json'), request_sha256='e'*64,
                           on_request=Path(ready['on_binding']['path']), on_request_sha256=ready['on_binding']['sha256'],
                           run_dir=ready['context']['run_dir'], device='cuda')
    assert module.train(ready['bundle'], ready['context'], ready['on_binding'], args)['passed'] is True
    assert invoked == [(True, ready['gate_binding'], ready['on']['registration'])]
    assert all(path.read_bytes() == content for path,content in before.items())
    result = json.loads((ready['context']['run_dir'] / 'training/on/on.json').read_text())
    report = bridge._json(result['fresh_replay'])
    assert report['n_images'] == 222 and report['buffers_unchanged'] is True and report['max_abs_difference'] == 0.
    assert report['identities']['registration_sha256'] == ready['result']['identities']['registration_sha256']
    assert report['identities']['preprocess_sha256'] == ready['result']['identities']['preprocess_sha256']
    assert bridge._json(bridge_binding(ready['context']['run_dir'] / 'training/on/replay.json'))['fresh_process'] is False


@pytest.fixture
def scientific_source(tmp_path):
    module = launcher()
    source = tmp_path / 'science'
    package = source / 'src/refer_or_clear'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('marker = "frozen-package"\n')
    for name in ('direct_reference_train','direct_reference_control','direct_reference_pilot','pilots','train'):
        (package / f'{name}.py').write_text('marker = "frozen"\nselection_key = None\n')
    (package / 'data').mkdir()
    (package / 'data/__init__.py').write_text('')
    (package / 'data/direct_reference.py').write_text('marker = "frozen-data"\n')
    (package / 'models').mkdir()
    (package / 'models/__init__.py').write_text('')
    (package / 'anchor_contract.py').write_text('def contract_for(track):\n    return "frozen-contract"\n')
    (package / 'models/anchor_changer.py').write_text('from refer_or_clear.anchor_contract import contract_for\nmarker = contract_for("D_running_pilot")\n')
    (package / 'models/anchor_rgb.py').write_text('from refer_or_clear.anchor_contract import contract_for\nmarker = contract_for("R")\n')
    (package / 'direct_reference_train.py').write_text('from .models import anchor_changer, anchor_rgb\nmarker = (anchor_changer.marker, anchor_rgb.marker)\n')
    inventory = [dict(path=path.relative_to(source).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      size=path.stat().st_size) for path in sorted(source.rglob('*.py'))]
    digest = hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    request_binding = write(tmp_path / 'request.json', dict(source_sha256=digest))
    foreign = tmp_path / 'editable/src/refer_or_clear'
    foreign.mkdir(parents=True)
    (foreign / '__init__.py').write_text('marker = "editable-package"\n')
    (foreign / 'anchor_contract.py').write_text('def contract_for(track):\n    return "editable-contract"\n')
    return dict(module=module,source=source,package=package,request=request_binding,digest=digest,foreign=foreign)


def bootstrap(fixture, extra=''):
    code = '''
import importlib.util
from pathlib import Path
import sys
spec = importlib.util.spec_from_file_location('external_on', sys.argv[1])
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
launcher.SCIENTIFIC_SHA256 = sys.argv[2]
sys.path.insert(0, sys.argv[5])
''' + extra + '''
bundle = launcher.load_science(Path(sys.argv[3]), dict(path=sys.argv[4],sha256=launcher.digest(sys.argv[4])))
assert bundle.service.marker == ('frozen-contract','frozen-contract')
assert bundle.service.__name__ == 'refer_or_clear.direct_reference_train'
assert bundle.bridge.marker == 'frozen-data'
assert all(Path(module.__file__).is_relative_to(Path(sys.argv[3])) for name,module in sys.modules.items()
           if name == 'refer_or_clear' or name.startswith('refer_or_clear.'))
'''
    return subprocess.run([sys.executable,'-c',code,fixture['module'].__file__,fixture['digest'],str(fixture['source']),
                           fixture['request']['path'],str(fixture['foreign'].parent)],capture_output=True,text=True,timeout=30)


def test_absolute_imports_resolve_verified_canonical_source(scientific_source):
    result = bootstrap(scientific_source)
    assert result.returncode == 0, result.stderr


def test_bootstrap_rejects_preloaded_foreign_canonical_package(scientific_source):
    result = bootstrap(scientific_source,'import refer_or_clear.anchor_contract\n')
    assert result.returncode != 0
    assert 'foreign' in result.stderr


def test_bootstrap_rejects_changed_source_before_import(scientific_source):
    (scientific_source['package'] / 'data/direct_reference.py').write_text('raise RuntimeError("unverified code executed")\n')
    result = bootstrap(scientific_source)
    assert result.returncode != 0 and 'inventory mismatch before imports' in result.stderr
    assert 'unverified code executed' not in result.stderr


def test_inventory_matches_original_frozen_algorithm(scientific_source):
    helper_path = Path(__file__).resolve().parents[1] / 'scripts/anchor_job_manifest.py'
    spec = importlib.util.spec_from_file_location('_original_inventory_helper',helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    assert helper.object_sha256(helper.source_entries(scientific_source['source'])) == scientific_source['digest']


def test_original_eec_inventory_matches_archived_actual_metadata():
    module = launcher()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'results/diagnostics/direct-reference-v1/direct-reference-diagnostic-38040/input_manifest.json').read_text())
    control = json.loads((root / 'results/runs/direct-reference-v1/requests/fold0-off100-eec5973.json').read_text())
    inventory_sha = hashlib.sha256(json.dumps(manifest['source_files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert inventory_sha == manifest['source_sha256'] == control['source_sha256'] == module.SCIENTIFIC_SHA256
    files = {entry['path']:entry for entry in manifest['source_files']}
    for name in ('anchor_changer.py','anchor_rgb.py'):
        path = 'src/refer_or_clear/models/'+name
        original = subprocess.run(['git','show','eec5973:'+path],cwd=root,capture_output=True,check=True).stdout
        assert hashlib.sha256(original).hexdigest() == files[path]['sha256']
        assert b'from refer_or_clear.anchor_contract import contract_for' in original


@pytest.fixture
def allocated_environment(monkeypatch):
    prefix = '/shared/b00090279/conda/envs/refer-or-clear'
    monkeypatch.setattr(sys,'prefix',prefix)
    monkeypatch.setattr(sys,'executable',prefix+'/bin/python')
    for name,value in dict(CONDA_PREFIX=prefix,SLURM_JOB_PARTITION='gpu',SLURM_JOB_ID='1234',
                           SLURMD_NODENAME='gpu-dy-g5-0-16',SLURM_CPUS_PER_TASK='4',
                           SLURM_GPUS_ON_NODE='1',CUDA_VISIBLE_DEVICES='0').items():
        monkeypatch.setenv(name,value)
    for name in ('SLURM_GPUS','SLURM_JOB_GPUS','SLURM_STEP_GPUS'):
        monkeypatch.delenv(name,raising=False)


def cli_arguments():
    return ['--source-dir','/restored/code','--request','/restored/request.json','--request-sha256','a'*64,
            '--on-request','/authority/on.json','--on-request-sha256','b'*64,'--run-dir','/original/run']


@pytest.mark.parametrize('field,value', [('sys.prefix','/wrong'),('sys.executable','/wrong/bin/python'),
    ('CONDA_PREFIX','/wrong'),('SLURM_JOB_PARTITION','cpu'),('SLURM_JOB_ID','0'),('SLURM_JOB_ID','invalid'),
    ('SLURMD_NODENAME','login01'),('SLURMD_NODENAME','head01'),('SLURMD_NODENAME','frontend01'),
    ('SLURMD_NODENAME','lgn01'),('SLURMD_NODENAME',''),('SLURM_CPUS_PER_TASK','3'),
    ('CUDA_VISIBLE_DEVICES',''),('CUDA_VISIBLE_DEVICES','-1'),('SLURM_GPUS_ON_NODE','0')])
def test_bad_environment_blocks_scientific_loading(allocated_environment,monkeypatch,field,value):
    module = launcher()
    if field.startswith('sys.'): monkeypatch.setattr(sys,field.split('.')[1],value)
    else: monkeypatch.setenv(field,value)
    def unexpected(*args):
        raise RuntimeError('scientific loading reached before environment rejection')
    monkeypatch.setattr(module,'load_science',unexpected)
    with pytest.raises(ValueError): module.main(cli_arguments())


def test_exact_allocated_environment_allows_cli_handoff(allocated_environment,monkeypatch):
    module = launcher()
    observed = []
    bundle = SimpleNamespace(service=SimpleNamespace(load_context=lambda *args: observed.append('context') or {}))
    monkeypatch.setattr(module,'load_science',lambda *args: observed.append('science') or bundle)
    monkeypatch.setattr(module,'train',lambda *args: observed.append('train') or dict(passed=True))
    assert module.main(cli_arguments())['passed'] is True
    assert observed == ['science','context','train']


@pytest.mark.parametrize('arguments', [[], ['--arm','off'], ['--epochs','1'], ['--resume','best.pt']])
def test_cli_has_no_flexible_recipe(arguments):
    module = launcher()
    with pytest.raises(SystemExit): module.main(arguments)
