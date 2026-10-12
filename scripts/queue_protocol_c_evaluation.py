"""Metadata-only C transition after original A OFF/ON training, never A test.

protocol_c_evaluation_queue_v1 has the same fields as the reviewed A queue
authority. Its score_template binds protocol_c_current_score_v1 with
source_selection={records,split,enrollment,weights,config}, WITHOUT arms.
The controller adds actual completed A metadata bindings only at that location.
The critical map must attest all 34 original src files in the frozen C inventory.
C enrollment/root/reservation JSON must already exist; products are not opened.
The reviewed wrapper receives SOURCE_DIR REQUEST REQUEST_SHA INPUTS_DIR
INPUTS_MANIFEST INPUTS_SHA. No A test result, A evaluation document, performance
threshold, checkpoint loading, CSV, weights, or query access is performed here.
--watch uses 300-second metadata polls and a maximum 20-hour deadline. Exclusive
intent/submission JSONL records forbid retry after a failed or ambiguous submit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import queue_protocol_a_evaluation as common


def bound_bytes(binding):
    payload = common.binding_path(binding).read_bytes()
    common.require(common.digest(payload) == binding['sha256'], 'reviewed metadata SHA256 mismatch')
    return payload


def validate_authority(binding):
    authority = common.document(binding)
    fields = {'schema_version', 'parent_job_id', 'evaluation_source_dir', 'score_template', 'wrapper',
              'inputs_dir', 'inputs_manifest', 'run_dir', 'request_path', 'ledger_path', 'submit_cwd'}
    common.require(set(authority) == fields and authority['schema_version'] == 'protocol_c_evaluation_queue_v1'
                   and type(authority['parent_job_id']) is int and authority['parent_job_id'] == 38152,
                   'strict C ON38152 queue authority required')
    template = common.document(authority['score_template'])
    fixed = dict(schema_version='protocol_c_current_score_v1', protocol='C', seed=42, eval_batch_size=1,
                 training_source_sha256=common.TRAINING_SOURCE_SHA256)
    extra = {'evaluation_source_sha256', 'critical_source_hashes', 'source_selection', 'target_split',
             'target_enrollment', 'enrollment', 'root', 'raw_root', 'output_root'}
    common.require(set(template) == set(fixed) | extra, 'strict C scoring template required')
    common.fixed_fields(template, fixed, 'fixed C transfer contract required')
    selected = template['source_selection']
    common.require(isinstance(selected, dict) and set(selected) == {'records', 'split', 'enrollment', 'weights', 'config'},
                   'original A source-selection metadata without arms required, never A test authority')
    source = common.absolute(authority['evaluation_source_dir'])
    inventory = common.source_entries(source)
    common.require(common.object_sha256(inventory) == template['evaluation_source_sha256'], 'frozen C source mismatch')
    hashes = {entry['path']: entry['sha256'] for entry in inventory}
    critical = template['critical_source_hashes']
    common.require(isinstance(critical, dict) and len(critical) == 34
                   and all(isinstance(path, str) and path.startswith('src/') and path.endswith('.py')
                           and common.checksum(value) and hashes.get(path) == value for path, value in critical.items()),
                   'all 34 original scientific source hashes required')
    authorities = list(selected.values()) + [template[name] for name in ('target_split', 'target_enrollment', 'enrollment', 'root')]
    protected = [common.binding_path(item) for item in authorities]
    target, enrollment, relocation = (common.document(template[name]) for name in ('target_enrollment', 'enrollment', 'root'))
    common.fixed_fields(target, dict(schema_version='protocol_c_target_enrollment_v1', protocol='C',
                                    records_sha256=selected['records']['sha256'],
                                    source_split_sha256=selected['split']['sha256'],
                                    target_split_sha256=template['target_split']['sha256']), 'C support reservation metadata required')
    common.fixed_fields(enrollment, dict(schema_version='protocol_c_enrollment_v1', protocol='C',
                                        records_sha256=selected['records']['sha256'],
                                        source_split_sha256=selected['split']['sha256'],
                                        target_split_sha256=template['target_split']['sha256'],
                                        target_enrollment_authority=template['target_enrollment']), 'prebuilt C support metadata required')
    banks = enrollment.get('banks')
    common.require(isinstance(banks, dict) and banks and all(isinstance(bank.get('enrollment', {}).get('products'), dict)
                   and bank['enrollment']['products'] for bank in banks.values()), 'completed C support products metadata required')
    common.fixed_fields(relocation, dict(schema_version='protocol_c_product_root_v1', protocol='C',
                                        enrollment_manifest_sha256=template['enrollment']['sha256']), 'C root relocation authority required')
    common.require(relocation.get('original_root') == enrollment.get('output_root'), 'original C product root mismatch')
    bound_bytes(authority['wrapper'])
    payload = bound_bytes(authority['inputs_manifest'])
    rows = [re.fullmatch(r'([0-9a-f]{64}) [ *](.+)', line) for line in payload.decode().splitlines() if line.strip()]
    common.require(len(rows) == 7 and all(rows), 'reviewed seven-file input inventory required')
    names = [row.group(2) for row in rows]
    common.require(len(set(names)) == 7 and all(not Path(name).is_absolute() and '..' not in Path(name).parts
                   and '\\' not in name for name in names), 'safe unique input inventory paths required')
    protected += [source, Path(__file__).resolve(), Path(common.__file__).resolve(), common.absolute(binding['path']),
                  common.binding_path(authority['score_template']), common.binding_path(authority['wrapper']),
                  common.binding_path(authority['inputs_manifest']), common.absolute(authority['inputs_dir']),
                  common.absolute(authority['run_dir']), common.absolute(template['raw_root']),
                  common.absolute(relocation['original_root']), common.absolute(relocation['relocated_root'])]
    outputs = [common.absolute(authority[name]) for name in ('request_path', 'ledger_path')] + [common.absolute(template['output_root'])]
    common.require(all(not output.is_relative_to(path) and not path.is_relative_to(output) for output in outputs for path in protected)
                   and all(not first.is_relative_to(second) and not second.is_relative_to(first)
                           for index, first in enumerate(outputs) for second in outputs[index + 1:]), 'C output overlaps immutable input')
    common.require(common.absolute(authority['inputs_dir']).is_dir()
                   and common.absolute(authority['submit_cwd']).is_dir(), 'existing input/submission directories required')
    return authority, template


def existing_submission(authority, binding, template):
    path = common.absolute(authority['ledger_path'])
    if not path.exists():
        return None
    records = [common.parse(line) for line in path.read_text().splitlines() if line.strip()]
    common.require(len(records) == 2 and [record.get('status') for record in records] == ['intent', 'submitted'],
                   'incomplete or failed C ledger; manual review required, no resubmission')
    request_binding = records[1].get('request')
    common.require(records[0].get('request') == request_binding and common.binding_path(request_binding) == common.absolute(authority['request_path']),
                   'existing C request ledger mismatch')
    request = common.document(request_binding)
    source = dict(request.get('source_selection', {}))
    arms = source.pop('arms', None)
    common.require(isinstance(arms, dict) and set(arms) == {'off', 'on'} and source == template['source_selection']
                   and set(request) == set(template) and all(request[key] == value for key, value in template.items() if key != 'source_selection'),
                   'existing C request differs from reviewed template')
    expected = dict(schema_version='protocol_c_evaluation_queue_ledger_v1', queue_authority=binding,
                    parent_job_id=38152, wrapper=authority['wrapper'], inputs_manifest=authority['inputs_manifest'],
                    command=common.submission_command(authority, request_binding))
    for record in records:
        common.fixed_fields(record, expected, 'existing C ledger reviewed submission mismatch')
    common.require(type(records[1].get('job_id')) is int and records[1]['job_id'] > 0, 'submitted C job identity required')
    return records[1]


def once(binding):
    authority, template = validate_authority(binding)
    previous = existing_submission(authority, binding, template)
    if previous is not None:
        return previous
    state = common.parent_state(38152)
    if state != 'COMPLETED':
        return dict(status='waiting', protocol='C', parent_job_id=38152, parent_state=state)
    arms = common.completed_arms(authority, template['source_selection'])
    validate_authority(binding)
    payload = common.encode(dict(template, source_selection=dict(template['source_selection'], arms=arms)))
    request = dict(path=authority['request_path'], sha256=common.digest(payload))
    common.require(not common.absolute(template['output_root']).exists() and not common.absolute(request['path']).exists(),
                   'new exclusive C request/output required')
    command = common.submission_command(authority, request)
    record = dict(schema_version='protocol_c_evaluation_queue_ledger_v1', queue_authority=binding, parent_job_id=38152,
                  request=request, wrapper=authority['wrapper'], inputs_manifest=authority['inputs_manifest'], command=command)
    ledger = common.absolute(authority['ledger_path'])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open('x') as stream:
        common.append(stream, dict(record, status='intent', utc=datetime.now(timezone.utc).isoformat()))
        try:
            path = common.absolute(request['path'])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as output:
                output.write(payload)
                output.flush()
                common.os.fsync(output.fileno())
            response = common.subprocess.run(command, cwd=authority['submit_cwd'], check=True, capture_output=True, text=True)
            match = re.fullmatch(r'([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?\s*', response.stdout)
            common.require(match is not None, 'ambiguous C submission; manual review required, no resubmission')
            result = dict(record, status='submitted', job_id=int(match.group(1)), utc=datetime.now(timezone.utc).isoformat())
            common.append(stream, result)
            return result
        except Exception as error:
            common.append(stream, dict(record, status='failed_or_unknown', error_type=type(error).__name__, utc=datetime.now(timezone.utc).isoformat()))
            raise


class WatchTimeout(TimeoutError):
    def __init__(self, report):
        self.report = report
        super().__init__('Protocol C metadata watch deadline reached without submission')


def watch(binding, *, poll_seconds=300, timeout_seconds=72000, sleeper=None, clock=None):
    common.require(type(poll_seconds) is int and poll_seconds > 0 and type(timeout_seconds) is int
                   and 0 < timeout_seconds <= 72000, 'positive polling interval and at most 20-hour deadline required')
    sleeper, clock = sleeper or time.sleep, clock or time.monotonic
    deadline, polls = clock() + timeout_seconds, 0
    while True:
        if clock() >= deadline:
            raise WatchTimeout(dict(status='timeout', protocol='C', parent_job_id=38152, polls=polls,
                                    timeout_seconds=timeout_seconds, submission_attempted=False))
        result = once(binding)
        polls += 1
        if result.get('status') == 'submitted':
            return result
        common.require(result.get('status') == 'waiting', 'unexpected C transition')
        print(json.dumps(dict(result, polls=polls), sort_keys=True), file=sys.stderr, flush=True)
        remaining = deadline - clock()
        if remaining > 0:
            sleeper(min(poll_seconds, remaining))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--authority-sha256', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--once', action='store_true')
    mode.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=300)
    parser.add_argument('--timeout-seconds', type=int, default=72000)
    args = parser.parse_args(argv)
    binding = dict(path=args.authority, sha256=args.authority_sha256)
    try:
        result = watch(binding, poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_seconds) if args.watch else once(binding)
    except WatchTimeout as error:
        print(json.dumps(error.report, sort_keys=True), file=sys.stderr, flush=True)
        raise SystemExit(2) from error
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
