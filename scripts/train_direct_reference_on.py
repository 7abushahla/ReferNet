"""External ON100 launcher. Deploy outside the unchanged eec5973 source tree.

ON authority fields: schema_version, arm, epochs, fold, seed, source_sha256,
config_sha256, diagnostic_request_sha256, launcher_sha256, off_completion,
registration, phase_authority. The last three are external path/SHA256 pairs.
Restore original absolute input paths first. This script does not stage inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

SCIENTIFIC_SHA256 = 'a5ff28ae2a332b977551241f2d07641d51df78d54a4cc1b58169c7338ce11292'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def allocated_environment():
    prefix = '/shared/b00090279/conda/envs/refer-or-clear'
    require(sys.prefix == prefix and str(Path(sys.executable).parent) == prefix+'/bin'
            and os.environ.get('CONDA_PREFIX') == prefix, 'exact project interpreter/Conda prefix required')
    require(os.environ.get('SLURM_JOB_PARTITION') == 'gpu', 'allocated GPU partition required')
    job = os.environ.get('SLURM_JOB_ID','')
    require(job.isdecimal() and int(job) > 0, 'allocated Slurm job required')
    node = os.environ.get('SLURMD_NODENAME','').lower()
    require(node and not any(name in node for name in ('login','head','frontend','lgn')), 'allocated compute node required')
    cpus = os.environ.get('SLURM_CPUS_PER_TASK','')
    require(cpus.isdecimal() and int(cpus) >= 4, 'four allocated CPUs required')
    counts = [os.environ.get(name,'0') for name in ('SLURM_GPUS','SLURM_GPUS_ON_NODE')]
    devices = [os.environ.get(name,'') for name in ('SLURM_JOB_GPUS','SLURM_STEP_GPUS')]
    require(any(count.isdecimal() and int(count) > 0 for count in counts)
            or any(device not in ('','-1','NoDevFiles') for device in devices), 'allocated GPU resource required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES','') not in ('','-1','NoDevFiles'), 'allocated GPU visibility required')


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def absolute(value):
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts and not any(parent.is_symlink() for parent in (path,*path.parents)),
            'absolute non-symlink path required')
    return path


def document(binding):
    require(isinstance(binding,dict) and set(binding) == {'path','sha256'}
            and isinstance(binding['sha256'],str) and re.fullmatch('[0-9a-f]{64}',binding['sha256']), 'external path/hash required')
    content = absolute(binding['path']).read_bytes()
    require(hashlib.sha256(content).hexdigest() == binding['sha256'], 'authority hash mismatch')
    def pairs(entries):
        result = {}
        for key,value in entries:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    value = json.loads(content,object_pairs_hook=pairs)
    json.dumps(value,allow_nan=False)
    require(isinstance(value,dict), 'JSON object required')
    return value


def load_science(source, request_binding):
    source = absolute(source)
    request = document(request_binding)
    require(request.get('source_sha256') == SCIENTIFIC_SHA256, 'original eec5973 scientific source required')
    entries = []
    suffixes = {'.py','.sh','.yaml','.yml','.toml','.txt','.md','.json','.cfg','.ini','.lock'}
    for directory,folders,files in os.walk(source,followlinks=False):
        folders[:] = sorted(name for name in folders if name not in {'.git','__pycache__','.pytest_cache'} and not name.startswith('._'))
        if Path(directory) == source:
            folders[:] = [name for name in folders if name in {'src','scripts','configs','tests','docs'}]
        for name in folders: absolute(Path(directory) / name)
        for name in sorted(files):
            if name.startswith('._') or Path(name).suffix.lower() not in suffixes: continue
            path = absolute(Path(directory) / name)
            require(path.is_file(), 'invalid scientific inventory file')
            entries.append(dict(path=path.relative_to(source).as_posix(),sha256=digest(path),size=path.stat().st_size))
    observed = hashlib.sha256(json.dumps(sorted(entries,key=lambda entry:entry['path']),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(observed == SCIENTIFIC_SHA256, 'scientific inventory mismatch before imports')
    require(not Path(__file__).resolve().is_relative_to(source), 'launcher must remain outside scientific source')
    def canonical_modules():
        root = source / 'src/refer_or_clear'
        for name,module in list(sys.modules.items()):
            if name == 'refer_or_clear' or name.startswith('refer_or_clear.'):
                locations = [getattr(module,'__file__',None),*getattr(module,'__path__',[])]
                require(all(location is not None and absolute(location).is_relative_to(root) for location in locations),
                        f'foreign preloaded/imported scientific module: {name}')
    canonical_modules()
    sys.path.insert(0,str(source / 'src'))
    sys.dont_write_bytecode = True
    modules = {name:importlib.import_module(f'refer_or_clear.{filename}') for name,filename in
               dict(service='direct_reference_train',off='direct_reference_control',core='direct_reference_pilot',
                    bridge='data.direct_reference',pilots='pilots',selection='train').items()}
    canonical_modules()
    import torch
    from torch.utils.data import DataLoader
    return SimpleNamespace(**modules,selection_key=modules['selection'].selection_key,torch=torch,DataLoader=DataLoader)


def selected(bundle, context, authorized, binding, arm):
    service, control = bundle.service, bundle.off
    directory = context['run_dir'] / 'training' / arm
    control._canonical(binding,directory / 'train.json')
    result = bundle.bridge._json(binding)
    fixed = dict(passed=True,phase='train',diagnostic_only=False,epochs_completed=100,optimizer_steps=22200,role='val',arm=arm)
    require(all(result.get(key) == value and type(result.get(key)) is type(value) for key,value in fixed.items()), 'complete selected ON/OFF100 required')
    require(result.get('scheduler') == dict(total_updates=22200,warmup_updates=1000), 'fixed scheduler receipt required')
    require(result.get('diagnostic_gate') == authorized['gate'] and result.get('registration') == authorized['prepared']['table_binding'], 'training gate/table mismatch')
    config = dict(context['config'],arm=arm,reference_enabled=arm == 'on')
    initialization = service._receipt_initialization(authorized['receipt']['training_gates'][arm])
    require(result.get('fresh_initial_state_sha256') == initialization['state_sha256'], 'original fresh tensor state required')
    provenance = copy.deepcopy(initialization['provenance'])
    provenance['base']['checkpoint_path'] = str(bundle.bridge._path(context['manifest']['authorities']['weights']['path']))
    identities = service._checkpoint_identities(context,config,authorized['prepared']['preprocess'],result['registration'],bundle.pilots.json_sha256(provenance))
    require(result.get('identities') == identities, 'selected source/config/preprocess/initialization mismatch')
    predictions = control._csv(control._canonical(result.get('predictions'),directory / 'val_predictions.csv'))
    for prediction in predictions:
        prediction['label'] = int(prediction['label'])
        for key in ('attack_logit','query_logit','correction_logit','attack_probability','bonafide_score'):
            prediction[key] = control._number(prediction[key])
        for key in ('context_available','referral_required'):
            require(prediction.get(key) in ('True','False'), 'availability must be boolean')
            prediction[key] = prediction[key] == 'True'
        probability = bundle.torch.tensor(prediction['attack_logit'],dtype=bundle.torch.float32).sigmoid().item()
        tolerance = bundle.pilots.ATOL + bundle.pilots.RTOL * abs(probability)
        require(abs(prediction['attack_probability']-probability) <= tolerance
                and abs(prediction['bonafide_score']-(1.-probability)) <= tolerance
                and prediction['context_available'] is not prediction['referral_required'], 'score/referral inconsistency')
    service._predictions(dict(n_images=len(predictions)),predictions,authorized['rows']['val'])
    summary = bundle.pilots.summarize_predictions(predictions)
    history = service.bind(directory / 'log.csv')
    entries = control._csv(bundle.bridge._bytes(history))
    require(len(entries) == 100, '100-epoch history required')
    best_key, best_epoch = None, None
    for epoch,entry in enumerate(entries,1):
        require(entry.get('epoch') == str(epoch) and entry.get('n_images') == '1776' and entry.get('optimizer_steps') == '222', 'ordered complete history required')
        key = bundle.selection_key('accuracy',control._number(entry['val_accuracy']),control._number(entry['val_auc']),control._number(entry['val_image_loss']),control._number(entry['loss']))
        if best_key is None or key > best_key: best_key,best_epoch = key,epoch
    require(type(result.get('selected_epoch')) is int and result['selected_epoch'] == best_epoch
            and result.get('selection_key') == bundle.pilots._selection_record(best_key)
            and (summary['accuracy'],summary['auc']) == best_key, 'selected metrics/earlier tie mismatch')
    control._canonical(result.get('checkpoint'),directory / 'best.pt')
    checkpoint = bundle.core.read_direct_reference_checkpoint(directory / 'best.pt',config,identities)
    require(checkpoint['phase'] == 'train' and checkpoint['epoch'] == best_epoch and checkpoint['selection_key'] == result['selection_key'], 'selected training checkpoint mismatch')
    require(result.get('replay',{}).get('passed') is True and result['replay'].get('fresh_process') is False
            and result['replay'].get('n_images') == 222, 'full in-process replay required')
    return dict(result=result,identities=identities,config=config,predictions=predictions,summary=summary,history=history)


def check_replay(bundle, report, selected_result, train_binding, request_binding, arm):
    fixed = dict(schema_version='direct_reference_control_replay_v1' if arm == 'off' else 'direct_reference_on_replay_v1',
                 phase='train',passed=True,fresh_process=True,arm=arm,n_images=222,buffers_unchanged=True,
                 train_result=train_binding,checkpoint=selected_result['result']['checkpoint'],identities=selected_result['identities'],
                 history=selected_result['history'],image_ids=[row['image_id'] for row in selected_result['predictions']])
    fixed['control_request' if arm == 'off' else 'on_request'] = request_binding
    if arm == 'on': fixed['launcher_sha256'] = digest(__file__)
    require(all(report.get(key) == value and type(report.get(key)) is type(value) for key,value in fixed.items()), 'complete fresh replay binding required')
    require(type(report.get('pid')) is int and report['pid'] > 0, 'replay process identity required')
    actual = report.get('predictions',[])
    comparison = bundle.pilots.compare_replay(selected_result['predictions'],actual)
    require(comparison['passed'] and comparison == report.get('replay') and comparison['max_abs_difference'] == report.get('max_abs_difference'), 'fresh replay evidence mismatch')
    require(all(actual_row.get(key) == row[key] for row,actual_row in zip(selected_result['predictions'],actual)
                for key in ('label','design_id','attack_family','context_available','referral_required')), 'replay metadata/referrals mismatch')
    summary = bundle.pilots.summarize_predictions(actual)
    require(all(report.get('summary',{}).get(key) == value for key,value in summary.items())
            and report['summary'].get('buffers_unchanged') is True, 'replay summary/buffers mismatch')


def authorize(bundle, context, binding, *, replay=False):
    authority = document(binding)
    fixed = dict(schema_version='direct_reference_on_request_v1',arm='on',epochs=100,fold=0,seed=42,
                 source_sha256=SCIENTIFIC_SHA256,launcher_sha256=digest(__file__))
    require(set(authority) == set(fixed) | {'config_sha256','diagnostic_request_sha256','off_completion','registration','phase_authority'}, 'fixed ON authority fields required')
    require(all(authority.get(key) == value and type(authority.get(key)) is type(value) for key,value in fixed.items()), 'fixed ON authority mismatch')
    require(not (context['run_dir'] / 'training/on').exists() or replay, 'new ON output directory required')
    bundle.off._canonical(authority['off_completion'],context['run_dir'] / 'training/off/control.json')
    completion = document(authority['off_completion'])
    require(completion.get('schema_version') == 'direct_reference_control_result_v1' and completion.get('passed') is True
            and completion.get('arm') == 'off' and completion.get('phase') == 'train' and type(completion.get('epochs_completed')) is int
            and completion['epochs_completed'] == 100, 'completed OFF control required')
    context,receipt,rows,control = bundle.off.authorize_control(context,completion['control_request'])
    require(all(authority[key] == control[key] for key in ('source_sha256','config_sha256','diagnostic_request_sha256')), 'original OFF source/recipe/request required')
    bundle.off._canonical(authority['phase_authority'],context['run_dir'] / 'training/shared/phase-authority.json')
    bundle.off._canonical(authority['registration'],context['run_dir'] / 'training/shared/registration.json')
    prepared = bundle.service._load_prepared(context,receipt,rows,control['diagnostic_gate'],dict(
        preprocess=bundle.service._preprocess(context,authority['phase_authority']),table_binding=authority['registration']))
    authorized = dict(prepared=prepared,receipt=receipt,rows=rows,gate=control['diagnostic_gate'])
    baseline = selected(bundle,context,authorized,completion['train_result'],'off')
    require(completion.get('identities') == baseline['identities'] and completion.get('checkpoint') == baseline['result']['checkpoint'], 'OFF root identity mismatch')
    require(baseline['summary']['accuracy'] >= .97 and baseline['summary']['auc'] >= .99, 'OFF competence gate not met')
    bundle.off._canonical(completion['fresh_replay'],context['run_dir'] / 'training/off/fresh-replay.json')
    check_replay(bundle,document(completion['fresh_replay']),baseline,completion['train_result'],completion['control_request'],'off')
    return authorized


def replay(bundle, context, binding, train_binding):
    authorized = authorize(bundle,context,binding,replay=True)
    chosen = selected(bundle,context,authorized,train_binding,'on')
    model = bundle.service._fresh_model(context,True,authorized['receipt'])
    with bundle.service.pixel_audit(context,authorized['rows']['val']) as opened:
        dataset = bundle.service._dataset(context,authorized['prepared'],authorized['rows']['val'],True)
        loader = bundle.DataLoader(dataset,batch_size=1,shuffle=False,drop_last=False,num_workers=0,
                                  generator=bundle.torch.Generator().manual_seed(context['config']['seed']+3001))
        summary,predictions,result = bundle.core.replay_direct_reference_checkpoint(Path(chosen['result']['checkpoint']['path']),model,loader,context['device'],
            config=chosen['config'],identities=chosen['identities'],expected_predictions=chosen['predictions'])
    bundle.service._predictions(summary,predictions,authorized['rows']['val'])
    require(summary.get('buffers_unchanged') is True, 'replay changed buffers')
    report = dict(schema_version='direct_reference_on_replay_v1',passed=True,fresh_process=True,arm='on',phase='train',pid=os.getpid(),
                  launcher_sha256=digest(__file__),train_result=train_binding,on_request=binding,checkpoint=chosen['result']['checkpoint'],
                  identities=chosen['identities'],history=chosen['history'],n_images=222,image_ids=[row['image_id'] for row in predictions],
                  buffers_unchanged=True,predictions=predictions,summary=summary,replay=result,max_abs_difference=result['max_abs_difference'],query_reads=opened)
    authorize(bundle,context,binding,replay=True)
    selected(bundle,context,authorized,train_binding,'on')
    bundle.service.write_receipt(context['run_dir'] / 'training/on/fresh-replay.json',report)
    return report


def train(bundle, context, binding, args):
    authorized = authorize(bundle,context,binding)
    result = bundle.service.run_training(context,True,authorized['gate'],prepared=authorized['prepared'])
    train_binding = bundle.service.bind(context['run_dir'] / 'training/on/train.json')
    require(document(train_binding) == result, 'returned ON result mismatch')
    chosen = selected(bundle,context,authorized,train_binding,'on')
    command = [sys.executable,str(Path(__file__).resolve())]
    for name in ('source_dir','request','request_sha256','on_request','on_request_sha256','run_dir','device'):
        command += ['--'+name.replace('_','-'),str(getattr(args,name))]
    command += ['--phase','replay','--train-result',train_binding['path'],'--train-result-sha256',train_binding['sha256']]
    subprocess.run(command,check=True)
    replay_binding = bundle.service.bind(context['run_dir'] / 'training/on/fresh-replay.json')
    report = document(replay_binding)
    require(type(report.get('pid')) is int and report['pid'] > 0 and report['pid'] != os.getpid(), 'distinct child process required')
    check_replay(bundle,report,chosen,train_binding,binding,'on')
    authorize(bundle,context,binding,replay=True)
    selected(bundle,context,authorized,train_binding,'on')
    completion = dict(schema_version='direct_reference_on_result_v1',passed=True,arm='on',epochs_completed=100,
                      scientific_source_sha256=SCIENTIFIC_SHA256,launcher_sha256=digest(__file__),on_request=binding,
                      train_result=train_binding,fresh_replay=replay_binding,checkpoint=result['checkpoint'],identities=chosen['identities'])
    bundle.service.write_receipt(context['run_dir'] / 'training/on/on.json',completion)
    return completion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir','request','request-sha256','on-request','on-request-sha256','run-dir'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',choices=['cuda'],default='cuda')
    parser.add_argument('--phase',choices=['train','replay'],default='train')
    parser.add_argument('--train-result')
    parser.add_argument('--train-result-sha256')
    args = parser.parse_args(argv)
    if (args.phase == 'replay' and not (args.train_result and args.train_result_sha256)) or (args.phase == 'train' and (args.train_result or args.train_result_sha256)):
        parser.error('external train-result path/hash permitted and required only for replay')
    allocated_environment()
    bundle = load_science(Path(args.source_dir),dict(path=args.request,sha256=args.request_sha256))
    context = bundle.service.load_context(Path(args.request),args.request_sha256,Path(args.run_dir),args.device)
    binding = dict(path=args.on_request,sha256=args.on_request_sha256)
    return replay(bundle,context,binding,dict(path=args.train_result,sha256=args.train_result_sha256)) if args.phase == 'replay' else train(bundle,context,binding,args)


if __name__ == '__main__':
    raise SystemExit(0 if main()['passed'] else 1)
