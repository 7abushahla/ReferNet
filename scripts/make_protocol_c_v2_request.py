"""Write a protocol_c_v2_score_v1 request for scripts/score_protocol_c_v2.py.

Login-node safe: plain python3, no torch, no product or pixel access. The C
target bindings (target_split, target_enrollment, enrollment, root, raw_root)
are copied verbatim from the reviewed protocol_c_current_score_v1 template.
The A request must be a fold-0 protocol_a_request_v1 whose source_sha256 is
SOURCE_SHA, and the gate must belong to that request. SOURCE_SHA is checked
against the frozen SOURCE_DIR inventory with anchor_job_manifest.source_entries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from anchor_job_manifest import object_sha256, source_entries

TEMPLATE = ROOT / 'results/runs/protocol-c-current/requests/score-template-4aeb0d2.json'
C_FIELDS = ('target_split', 'target_enrollment', 'enrollment', 'root', 'raw_root')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def absolute(value):
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts, f'absolute path without traversal required: {value}')
    return path


def document(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), f'JSON object required: {path}')
    return value


def build(source_dir, source_sha, a_request, a_gate, output_root, template=TEMPLATE):
    require(re.fullmatch('[0-9a-f]{64}', source_sha) is not None, 'lowercase SOURCE_SHA required')
    require(object_sha256(source_entries(Path(source_dir))) == source_sha, 'SOURCE_SHA differs from SOURCE_DIR inventory')
    a_request = absolute(a_request)
    a_gate = absolute(a_gate)
    output_root = absolute(output_root)
    study = document(a_request)
    require(study.get('schema_version') == 'protocol_a_request_v1' and study.get('protocol') == 'A'
            and type(study.get('fold')) is int and study['fold'] == 0, 'fold-0 protocol_a_request_v1 required')
    require(study.get('source_sha256') == source_sha, 'A request source differs from SOURCE_SHA')
    a_binding = dict(path=str(a_request), sha256=file_sha256(a_request))
    gate = document(a_gate)
    require(gate.get('schema_version') == 'protocol_a_training_gate_v1' and gate.get('passed') is True
            and gate.get('request') == a_binding, 'passed gate bound to this A request required')
    require(a_gate == absolute(study['output_root']) / 'gate/gate.json', 'gate must be the A study gate/gate.json')
    for protected in (absolute(study['output_root']), Path(source_dir).resolve(), absolute(study['raw_root'])):
        require(not output_root.is_relative_to(protected) and not protected.is_relative_to(output_root),
                'output root overlaps A study, source or raw inputs')
    reviewed = document(template)
    require(reviewed.get('schema_version') == 'protocol_c_current_score_v1' and reviewed.get('protocol') == 'C',
            'reviewed C score template required')
    request = dict(schema_version='protocol_c_v2_score_v1', protocol='C', seed=42, eval_batch_size=1,
                   source_sha256=source_sha, a_request=a_binding, a_gate_sha256=file_sha256(a_gate))
    for name in C_FIELDS:
        request[name] = json.loads(json.dumps(reviewed[name]))
    for name in C_FIELDS[:-1]:
        require(isinstance(request[name], dict) and set(request[name]) == {'path', 'sha256'}
                and re.fullmatch('[0-9a-f]{64}', request[name]['sha256']) is not None, f'template {name} binding required')
    require(isinstance(request['raw_root'], str) and request['raw_root'].startswith('/'), 'template raw_root required')
    request['output_root'] = str(output_root)
    return request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_dir')
    parser.add_argument('source_sha')
    parser.add_argument('a_request')
    parser.add_argument('a_gate')
    parser.add_argument('output_root')
    parser.add_argument('--template', default=str(TEMPLATE))
    parser.add_argument('--output', required=True, help='new request JSON path (must not exist)')
    args = parser.parse_args(argv)
    request = build(args.source_dir, args.source_sha, args.a_request, args.a_gate, args.output_root, args.template)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(request, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'{file_sha256(output)}  {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
