"""Prepare reporting metadata, not evaluation or execution authorization.

current_report_preparation_v1 binds protocol, index, records, split, enrollment,
scorer_result and arms {off,on}. Each arm explicitly binds train_result,
val_summary and fresh_replay. B additionally binds target_enrollment; C binds
target_split and target_enrollment, while split/enrollment retain A provenance.
All bindings are absolute path/SHA256 pairs. Native summaries are never rewritten.
The scorer's bound request supplies producer provenance, not directory discovery.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import report_current_matched as reporter
from refer_or_clear.protocol_a_contract import load_protocol_a_contract
from refer_or_clear.protocol_b_contract import load_protocol_b_contract, load_target_enrollment as load_b_target
from refer_or_clear.protocol_c_contract import load_protocol_c_contract, load_target_enrollment as load_c_target

require = reporter.require


class Metadata:
    def __init__(self):
        self.consumed = {}

    def read(self, binding, suffix='.json'):
        reporter.validate_binding(binding)
        path = Path(binding['path'])
        require(path.suffix.lower() == suffix and '..' not in path.parts
                and path.is_file() and not any(parent.is_symlink() for parent in (path, *path.parents)),
                'regular explicit JSON/CSV metadata required')
        content = path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        require(checksum == binding['sha256'], 'metadata SHA256 mismatch')
        require(str(path) not in self.consumed or self.consumed[str(path)] == checksum,
                'conflicting metadata bindings')
        self.consumed[str(path)] = checksum
        return reporter.json_object(content) if suffix == '.json' else reporter.csv_rows(content)

    def recheck(self):
        for path, checksum in self.consumed.items():
            self.read(dict(path=path, sha256=checksum), Path(path).suffix.lower())


def _write(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(payload)
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest())


def _populations(metadata, request):
    protocol, index = request['protocol'], request['index']
    records, split = request['records'], request['split']
    metadata.read(split, '.csv')
    if protocol == 'C':
        require(index == 0, 'C retains original A source fold zero')
        metadata.read(request['target_split'], '.csv')
        contract = load_protocol_c_contract(Path(records['path']), Path(split['path']),
            Path(request['target_split']['path']), records_sha256=records['sha256'])
    else:
        loader = load_protocol_a_contract if protocol == 'A' else load_protocol_b_contract
        contract = loader(Path(records['path']), Path(split['path']), records_sha256=records['sha256'],
                          **{('fold' if protocol == 'A' else 'group'): index})
    if protocol == 'A':
        queries = tuple(row.image_id for row in contract.target_test)
    else:
        target = request['target_enrollment']
        metadata.read(target)
        loader = load_c_target if protocol == 'C' else load_b_target
        queries = loader(contract, Path(target['path']), expected_sha256=target['sha256']).query_ids
    return dict(val=sorted(row.image_id for row in contract.source_val), test=sorted(queries))


def _fresh(metadata, binding, train_binding, train, source_binding, legacy, val_summary):
    fresh = metadata.read(binding)
    require(fresh.get('passed') is True and fresh.get('fresh_process') is True
            and type(fresh.get('pid')) is int and fresh['pid'] > 0,
            'completed fresh-process replay required')
    require(fresh.get('train_result' if legacy else 'parent_result') == train_binding
            and fresh.get('checkpoint') == train['checkpoint']
            and fresh.get('identities') == train['identities'], 'fresh replay parent/checkpoint identities mismatch')
    require(fresh.get('replay' if legacy else 'comparison', {}).get('passed') is True,
            'fresh replay comparison required')
    if not legacy:
        require(fresh.get('request') == source_binding, 'fresh replay request mismatch')
    if 'pid' in train:
        require(type(train['pid']) is int and train['pid'] > 0 and train['pid'] != fresh['pid'],
                'distinct replay process required')
    summary = fresh.get('summary')
    require(isinstance(summary, dict), 'fresh replay native summary required')
    for key in ('n_images', 'accuracy', 'auc'):
        if key in val_summary:
            require(key in summary and type(summary[key]) is type(val_summary[key])
                    and summary[key] == val_summary[key], 'fresh replay/selected validation summary mismatch')


def _assemble(metadata, request):
    protocol, index = request['protocol'], request['index']
    records = reporter.unique_rows(metadata.read(request['records'], '.csv'), 'records')
    metadata.read(request['enrollment'])
    populations = _populations(metadata, request)
    require(all(populations.values()) and not set(populations['val']) & set(populations['test']),
            'nonempty disjoint scored populations required')
    require(all(len(ids) == len(set(ids)) and set(ids) <= set(records) for ids in populations.values()),
            'official population records missing or duplicated')
    if protocol == 'C':
        taxonomy = Counter('genuine' if reporter.binary_label(records[identifier]['label']) == 0
                           else records[identifier]['attack_family'] for identifier in populations['test'])
        require(taxonomy == dict(genuine=234, face_swap=113, text_inpaint=120),
                'locked C 467-query taxonomy required')
    scorer = metadata.read(request['scorer_result'])
    source_binding = scorer['request']
    source = metadata.read(source_binding)
    legacy = scorer.get('schema_version') == f'protocol_{protocol.lower()}_current_score_result_v1'
    if legacy:
        require(protocol in ('A', 'C') and index == 0, 'legacy A/C source fold zero required')
        require(source.get('schema_version') == f'protocol_{protocol.lower()}_current_score_v1'
                and source.get('protocol') == protocol, 'bound scorer request schema mismatch')
        selection = source['source_selection'] if protocol == 'C' else source
        source_sha = source.get('training_source_sha256')
    else:
        require(protocol in ('A', 'B'), 'unsupported current producer schema')
        key = 'fold' if protocol == 'A' else 'group'
        require(scorer.get('schema_version') == f'protocol_{protocol.lower()}_evaluation_result_v1'
                and scorer.get('protocol') == protocol and type(scorer.get(key)) is int and scorer[key] == index
                and scorer.get('passed') is True and scorer.get('same_query_population') is True,
                'completed matched evaluation result required')
        require(source.get('schema_version') == f'protocol_{protocol.lower()}_request_v1'
                and source.get('protocol') == protocol and type(source.get(key)) is int and source[key] == index,
                'bound evaluation request schema/index mismatch')
        selection = source
        source_sha = source.get('source_sha256')
    for name in ('records', 'split', 'enrollment'):
        require(selection.get(name) == request[name], 'source metadata authority mismatch')
    for name in ('target_split', 'target_enrollment'):
        if name in request:
            require(source.get(name) == request[name], 'target metadata authority mismatch')
    result = dict(schema_version='current_matched_report_request_v1', protocol=protocol, index=index,
                  records=request['records'], arms={})
    if legacy:
        result['scorer_result'] = request['scorer_result']
    trains = {}
    for arm in ('off', 'on'):
        inputs = request['arms'][arm]
        require(isinstance(inputs, dict) and set(inputs) == {'train_result', 'val_summary', 'fresh_replay'},
                'explicit selected train/validation summary/fresh replay bindings required')
        train = metadata.read(inputs['train_result'])
        trains[arm] = train
        require(train.get('passed') is True and train.get('phase') == 'train'
                and train.get('diagnostic_only') is False and train.get('arm') == arm
                and type(train.get('epochs_completed')) is int and train['epochs_completed'] == 100,
                'completed selected 100-epoch arm required')
        for name, checksum in (('source_sha256', source_sha), ('records_sha256', request['records']['sha256']),
            ('split_sha256', request['split']['sha256']), ('enrollment_sha256', request['enrollment']['sha256'])):
            require(train.get('identities', {}).get(name) == checksum, 'training source/metadata identity mismatch')
        val_summary = metadata.read(inputs['val_summary'])
        if legacy:
            parent = scorer['arms'][arm]
            require(all(parent.get(name) == inputs[name] and selection['arms'][arm].get(name) == inputs[name]
                        for name in ('train_result', 'fresh_replay')), 'scorer source-selection parent mismatch')
            require(parent.get('completion') == selection['arms'][arm].get('completion'), 'completion authority mismatch')
            completion = metadata.read(parent['completion'])
            require(completion.get('passed') is True and completion.get('arm') == arm
                    and type(completion.get('epochs_completed')) is int and completion['epochs_completed'] == 100
                    and completion.get('train_result') == inputs['train_result']
                    and completion.get('fresh_replay') == inputs['fresh_replay']
                    and completion.get('checkpoint') == train['checkpoint'], 'completed producer training chain required')
            test = dict(predictions=parent['predictions'], summary=parent['summary'])
        else:
            require(train.get('request') == source_binding and train.get('summary') == inputs['val_summary'],
                    'selected training request/summary binding mismatch')
            summary_binding = scorer['summaries'][arm]
            summary = metadata.read(summary_binding)
            require(summary.get('authorization') == scorer['authorization'], 'test summary authorization mismatch')
            test = dict(predictions=summary['predictions'], summary=summary_binding)
        _fresh(metadata, inputs['fresh_replay'], inputs['train_result'], train, source_binding, legacy, val_summary)
        result['arms'][arm] = dict(completion=inputs['train_result'],
            val=dict(predictions=train['predictions'], summary=inputs['val_summary']), test=test)
    if not legacy:
        authorization = metadata.read(scorer['authorization'])
        key = 'fold' if protocol == 'A' else 'group'
        require(authorization.get('schema_version') == f'protocol_{protocol.lower()}_evaluation_v1'
                and authorization.get('protocol') == protocol and type(authorization.get(key)) is int
                and authorization[key] == index and authorization.get('source_validation_selected') is True
                and authorization.get('authorized_image_ids') == populations['test']
                and authorization.get('checkpoints') == {arm: train['checkpoint'] for arm, train in trains.items()},
                'evaluation authorization population/checkpoint mismatch')
        for name, checksum in (('records_sha256', request['records']['sha256']),
            ('split_sha256', request['split']['sha256']), ('enrollment_manifest_sha256', request['enrollment']['sha256'])):
            require(authorization.get(name) == checksum, 'evaluation authorization metadata identity mismatch')
    memberships = {role: [dict(image_id=identifier, role=role) for identifier in identifiers]
                   for role, identifiers in populations.items()}
    return result, memberships, source


def prepare_report(request_binding, output_dir):
    metadata = Metadata()
    request = metadata.read(request_binding)
    protocol = request.get('protocol')
    fields = {'schema_version', 'protocol', 'index', 'records', 'split', 'enrollment', 'scorer_result', 'arms'}
    fields |= {'target_enrollment'} if protocol == 'B' else {'target_split', 'target_enrollment'} if protocol == 'C' else set()
    require(set(request) == fields and request.get('schema_version') == 'current_report_preparation_v1'
            and protocol in ('A', 'B', 'C') and type(request.get('index')) is int and 0 <= request['index'] <= 9,
            'strict report preparation fields/protocol/index required')
    require(isinstance(request['arms'], dict) and set(request['arms']) == {'off', 'on'}, 'matched arms required')
    output = Path(output_dir)
    require(output.is_absolute() and '..' not in output.parts
            and not any(parent.is_symlink() for parent in (output, *output.parents)), 'canonical absolute output required')
    require(not output.exists(), 'new exclusive output directory required')
    result, memberships, source = _assemble(metadata, request)
    for path in metadata.consumed:
        require(not Path(path).is_relative_to(output) and not output.is_relative_to(Path(path)),
                'output overlaps input metadata')
    if 'output_root' in source:
        root = Path(source['output_root'])
        require(root.is_absolute() and not output.is_relative_to(root) and not root.is_relative_to(output),
                'output overlaps original producer output')
    metadata.recheck()
    output.mkdir(parents=True, exist_ok=False)
    try:
        result['memberships'] = {}
        for role, rows in memberships.items():
            path = output / f'{role}-memberships.csv'
            with path.open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result['memberships'][role] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report_binding = _write(output / 'report-request.json', result)
        report = reporter.build_report(report_binding, bootstrap=False)
        for path, checksum in report['metadata_sha256'].items():
            metadata.read(dict(path=path, sha256=checksum), Path(path).suffix.lower())
        metadata.recheck()
        receipt = dict(schema_version='current_report_preparation_result_v1', protocol=protocol, index=request['index'],
            metadata_ready=True, authorization_granted=False, request=request_binding, report_request=report_binding,
            scorer_result=request['scorer_result'], n_val=len(memberships['val']), n_test=len(memberships['test']),
            metadata_sha256=metadata.consumed,
            limitations='Metadata reporting only. No checkpoint/source/product byte verification or execution authority.')
        _write(output / 'preparation-result.json', receipt)
        with Path(report_binding['path'] + '.sha256').open('x') as stream:
            stream.write(report_binding['sha256'] + '\n')
        for path in output.iterdir():
            path.chmod(0o444)
        return receipt
    except Exception:
        shutil.rmtree(output)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare_report(dict(path=args.request, sha256=args.request_sha256), Path(args.output_dir))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'report preparation failed: {error}', file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
