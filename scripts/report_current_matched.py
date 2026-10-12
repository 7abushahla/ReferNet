"""Report completed matched runs from explicit hash-bound JSON/CSV metadata only.

The current_matched_report_request_v1 JSON has protocol (A/B/C), index
(fold/group), records, memberships {val,test}, and arms {off,on}. All input
bindings are {path,sha256}. Each arm has completion (selected train.json),
val and test, each containing predictions and summary bindings. Membership
CSVs explicitly enumerate the already scored populations with image_id and
role. Test summaries must bind source_checkpoint and predictions. Checkpoint
bindings are compared as receipts only, never opened. This request is a
reporting input, not evaluation authorization or population-selection policy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from refer_or_clear import metrics


LIMITATIONS = (
    'Metadata analysis only. No test authorization, inference, checkpoint loading, '
    'selection, architecture selection, or cross-validation mean. Membership is '
    'the explicit hash-bound scored population, not proof of the full published '
    'population or independent verification of source, checkpoint, or pixel bytes.'
)
BOOTSTRAP_INTERPRETATION = (
    'Paired cluster percentile intervals conditional on fixed trained checkpoints '
    'and fixed designs. They do not include training, checkpoint-selection, '
    'design-generalization, or cross-validation uncertainty. Source-val thresholds '
    'remain fixed in all resamples. Differences are ON minus OFF.'
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_binding(value):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'path/SHA256 binding required')
    require(isinstance(value['path'], str) and Path(value['path']).is_absolute(), 'absolute input path required')
    require(isinstance(value['sha256'], str) and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'invalid SHA256')
    return value


def json_object(content):
    def pairs(entries):
        result = {}
        for key, value in entries:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f'nonfinite JSON value: {value}')
    value = json.loads(content, object_pairs_hook=pairs, parse_constant=invalid_constant)
    require(isinstance(value, dict), 'JSON object required')
    return value


def csv_rows(content):
    reader = csv.DictReader(StringIO(content.decode('utf-8-sig'), newline=''), strict=True)
    require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), 'unique CSV header required')
    rows = list(reader)
    require(rows and all(None not in row and None not in row.values() for row in rows), 'complete nonempty CSV required')
    return rows


def unique_rows(rows, name):
    require(all(isinstance(row.get('image_id'), str) and row['image_id'].strip() for row in rows), f'{name}: image IDs required')
    result = {row['image_id']: row for row in rows}
    require(len(result) == len(rows), f'{name}: duplicate image IDs')
    return result


def binary_label(value):
    require(value in ('0', '1'), 'CSV label must be 0 or 1')
    return int(value)


def finite_probability(value):
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError('numeric saved probability required') from error
    require(math.isfinite(number) and 0 <= number <= 1, 'finite probability in [0,1] required')
    return number


def native_metrics(rows):
    scores = [row['bonafide_score'] for row in rows]
    labels = [row['label'] for row in rows]
    genuine = [row for row in rows if row['label'] == 0]
    altered = [row for row in rows if row['label'] == 1]
    false_positive_ids = sorted(row['image_id'] for row in genuine if row['bonafide_score'] < .5)
    miss_ids = sorted(row['image_id'] for row in altered if row['bonafide_score'] >= .5)
    result = dict(n_images=len(rows), n_genuine=len(genuine), n_altered=len(altered),
                  accuracy=metrics.deepid_acc(scores, labels), auc=metrics.auc(scores, labels),
                  genuine_false_positive_count=len(false_positive_ids), genuine_false_positives=false_positive_ids,
                  genuine_false_positive_rate=len(false_positive_ids) / len(genuine) if genuine else None,
                  altered_miss_count=len(miss_ids), altered_misses=miss_ids, subgroups={})
    for name, family in (('copy', 'text_copy'), ('inpainting', 'text_inpaint')):
        members = [row for row in altered if row['attack_family'] == family]
        population = genuine + members
        subgroup_scores = [row['bonafide_score'] for row in population]
        subgroup_labels = [row['label'] for row in population]
        result['subgroups'][name] = dict(n_genuine=len(genuine), n_altered=len(members),
            n_images=len(population), auc=metrics.auc(subgroup_scores, subgroup_labels),
            accuracy=metrics.deepid_acc(subgroup_scores, subgroup_labels) if genuine and members else None,
            altered_miss_count=sum(row['bonafide_score'] >= .5 for row in members))
    result['other_altered_count'] = sum(row['attack_family'] not in ('text_copy', 'text_inpaint') for row in altered)
    return result


def operational_rates(rows, threshold):
    scores = [row['bonafide_score'] for row in rows]
    labels = [row['label'] for row in rows]
    transported = metrics.transported(scores, labels, threshold)
    return dict(apcer=transported['miss'], bpcer=transported['referral'],
                hter=None if threshold is None else metrics.hter(scores, labels, threshold))


def check_summary(summary, rows, role, epoch):
    require(summary.get('role') == role and type(summary.get('selected_epoch')) is int
            and summary['selected_epoch'] == epoch, 'summary role/selected epoch mismatch')
    expected = native_metrics(rows)
    for key in ('n_images', 'accuracy', 'auc'):
        require(key in summary, f'summary missing {key}')
    for key in ('n_images', 'n_genuine', 'n_altered', 'accuracy', 'auc'):
        if key not in summary:
            continue
        value = expected[key]
        if type(value) is int:
            require(type(summary[key]) is int and summary[key] == value, f'summary {key} mismatch')
        elif value is None:
            require(summary[key] is None, f'summary {key} must be undefined')
        else:
            require(type(summary[key]) in (int, float) and math.isclose(summary[key], value, abs_tol=1e-9, rel_tol=1e-9),
                    f'summary {key} mismatch')
    for key in ('genuine_false_positives', 'altered_misses'):
        if key in summary:
            require(isinstance(summary[key], list) and sorted(summary[key]) == expected[key], f'summary {key} mismatch')
    if 'subgroups' in summary:
        require(isinstance(summary['subgroups'], dict), 'summary subgroups must be a mapping')
        for name, subgroup in summary['subgroups'].items():
            if name not in expected['subgroups']:
                continue
            require(isinstance(subgroup, dict), 'summary subgroup must be a mapping')
            for key in ('n_genuine', 'n_altered', 'accuracy', 'auc'):
                if key not in subgroup:
                    continue
                value = expected['subgroups'][name][key]
                if value is None or type(value) is int:
                    require(type(subgroup[key]) is type(value) and subgroup[key] == value, f'summary {name}/{key} mismatch')
                else:
                    require(type(subgroup[key]) in (int, float) and math.isclose(subgroup[key], value, abs_tol=1e-9, rel_tol=1e-9),
                            f'summary {name}/{key} mismatch')
    return expected


def paired_bootstrap(populations, thresholds):
    result = dict(seed=42, replicates=1000, confidence_level=.95, method='paired_cluster_percentile',
                  cluster_fields=['dataset', 'design_id', 'component_id'], interpretation=BOOTSTRAP_INTERPRETATION, roles={})
    generator = np.random.default_rng(42)
    for role, arms in populations.items():
        rows = arms['off']
        if any(not row.get(key, '').strip() or row[key].strip().lower() in ('unknown', 'unresolved', 'nan', 'none')
               for row in rows for key in result['cluster_fields']):
            result['roles'][role] = dict(status='skipped', reason='unresolved dataset/design/component lineage')
            continue
        clusters = {}
        for index, row in enumerate(rows):
            key = tuple(row[field] for field in result['cluster_fields'])
            clusters.setdefault(key, []).append(index)
        keys = sorted(clusters)
        if len(keys) < 2:
            result['roles'][role] = dict(status='skipped', reason='fewer than two resolved clusters', n_clusters=len(keys))
            continue
        labels = np.asarray([row['label'] for row in rows])
        scores = {arm: np.asarray([row['bonafide_score'] for row in members]) for arm, members in arms.items()}
        def values(arm, indices):
            selected_scores, selected_labels = scores[arm][indices], labels[indices]
            threshold = thresholds[arm]
            apcer, bpcer = (None, None) if threshold is None else metrics.apcer_bpcer(selected_scores, selected_labels, threshold)
            genuine = selected_labels == 0
            return dict(accuracy=metrics.deepid_acc(selected_scores, selected_labels), auc=metrics.auc(selected_scores, selected_labels),
                        genuine_false_positive_rate=float((selected_scores[genuine] < .5).mean()) if genuine.any() else None,
                        apcer=apcer, bpcer=bpcer)
        original = {arm: values(arm, np.arange(len(rows))) for arm in ('off', 'on')}
        differences = {name: [] for name in original['off']}
        for unused in range(1000):
            drawn = generator.integers(0, len(keys), size=len(keys))
            indices = np.concatenate([clusters[keys[int(position)]] for position in drawn])
            sampled = {arm: values(arm, indices) for arm in ('off', 'on')}
            for name in differences:
                if sampled['off'][name] is not None and sampled['on'][name] is not None:
                    differences[name].append(sampled['on'][name] - sampled['off'][name])
        intervals = {}
        for name, samples in differences.items():
            estimate = None if original['off'][name] is None or original['on'][name] is None else original['on'][name] - original['off'][name]
            intervals[name] = dict(estimate_on_minus_off=estimate, valid_replicates=len(samples),
                undefined_replicates=1000 - len(samples),
                interval_95=np.quantile(samples, [.025, .975]).tolist() if samples else None)
        result['roles'][role] = dict(status='complete', n_clusters=len(keys), n_images=len(rows), metrics=intervals)
    return result


def build_report(request_binding, *, bootstrap=False):
    require(type(bootstrap) is bool, 'boolean bootstrap option required')
    consumed = {}
    def artifact(value, suffix):
        validate_binding(value)
        path = Path(value['path'])
        require(path.suffix.lower() == suffix, 'only explicit metadata JSON/CSV files may be read')
        content = path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        require(checksum == value['sha256'], 'artifact SHA256 mismatch')
        require(str(path) not in consumed or consumed[str(path)] == checksum, 'conflicting file bindings')
        consumed[str(path)] = checksum
        return json_object(content) if suffix == '.json' else csv_rows(content)
    request = artifact(request_binding, '.json')
    require(request.get('schema_version') == 'current_matched_report_request_v1', 'unsupported report request')
    require(request.get('protocol') in ('A', 'B', 'C') and type(request.get('index')) is int
            and 0 <= request['index'] <= 9, 'explicit protocol/fold/group index required')
    require(isinstance(request.get('arms'), dict) and set(request['arms']) == {'off', 'on'}, 'matched OFF/ON arms required')
    require(isinstance(request.get('memberships'), dict) and set(request['memberships']) == {'val', 'test'}, 'separate val/test membership required')
    records = unique_rows(artifact(request['records'], '.csv'), 'records')
    members = {}
    for role in ('val', 'test'):
        selected = unique_rows(artifact(request['memberships'][role], '.csv'), role)
        members[role] = {}
        for identifier in sorted(selected):
            member = selected[identifier]
            require(member.get('role') == role and identifier in records, 'foreign membership role/ID')
            record = records[identifier]
            binary_label(record.get('label'))
            require(record.get('design_id', '').strip() and 'attack_family' in record, 'record metadata missing')
            for key in ('label', 'design_id', 'attack_family', 'dataset', 'component_id'):
                if key in member:
                    require(member[key] == record.get(key, ''), f'membership {key} mismatch')
            members[role][identifier] = record
    require(not set(members['val']) & set(members['test']), 'val/test IDs overlap')
    populations = {role: {} for role in ('val', 'test')}
    report_arms, shared_identity, checkpoints, thresholds = {}, None, {}, {}
    for arm in ('off', 'on'):
        inputs = request['arms'][arm]
        require(isinstance(inputs, dict) and set(inputs) == {'completion', 'val', 'test'}, 'explicit completion and role bindings required')
        completion = artifact(inputs['completion'], '.json')
        require(completion.get('passed') is True and completion.get('diagnostic_only') is False
                and completion.get('phase') == 'train' and completion.get('arm') == arm
                and type(completion.get('epochs_completed')) is int and completion['epochs_completed'] == 100,
                'completed non-diagnostic matched 100-epoch training required')
        require(isinstance(completion.get('replay'), dict) and completion['replay'].get('passed') is True,
                'selected validation replay must have passed')
        epoch = completion.get('selected_epoch')
        require(type(epoch) is int and 1 <= epoch <= 100, 'selected epoch must be recorded, never reselected')
        identity = completion.get('identities', {})
        require(isinstance(identity, dict), 'training identities required')
        common = {key: identity.get(key) for key in ('source_sha256', 'records_sha256', 'split_sha256', 'enrollment_sha256')}
        common.update({key: identity[key] for key in ('registration_sha256', 'preprocess_sha256') if key in identity})
        require(all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) for value in common.values()), 'shared training hash identities required')
        require(common['records_sha256'] == request['records']['sha256'], 'training records identity mismatch')
        if shared_identity is None:
            shared_identity = common
        require(common == shared_identity, 'OFF/ON source/data identities differ')
        checkpoints[arm] = validate_binding(completion.get('checkpoint'))
        report_arms[arm] = dict(selected_epoch=epoch, checkpoint=checkpoints[arm], completion=inputs['completion'])
        for role in ('val', 'test'):
            role_inputs = inputs[role]
            require(isinstance(role_inputs, dict) and set(role_inputs) == {'predictions', 'summary'}, 'explicit role prediction/summary bindings required')
            require(role != 'val' or completion.get('predictions') == role_inputs['predictions'], 'training prediction binding mismatch')
            saved = unique_rows(artifact(role_inputs['predictions'], '.csv'), f'{arm}/{role}')
            require(set(saved) == set(members[role]), 'exact scored population membership required')
            rows = []
            for identifier, record in members[role].items():
                row = saved[identifier]
                require(row.get('role', role) == role, 'prediction role mismatch')
                require(binary_label(row.get('label')) == binary_label(record['label']), 'prediction label mismatch')
                for key in ('design_id', 'attack_family'):
                    require(row.get(key) == record[key], f'prediction {key} mismatch')
                for key in ('dataset', 'component_id'):
                    if key in row:
                        require(row[key] == record.get(key, ''), f'prediction {key} mismatch')
                score = finite_probability(row.get('bonafide_score'))
                if 'attack_probability' in row:
                    require(math.isclose(score + finite_probability(row['attack_probability']), 1., abs_tol=1e-7, rel_tol=0), 'saved probability orientation mismatch')
                rows.append(dict(record, label=binary_label(record['label']), bonafide_score=score))
            summary = artifact(role_inputs['summary'], '.json')
            if role == 'test':
                require(summary.get('source_checkpoint') == checkpoints[arm]
                        and summary.get('predictions') == role_inputs['predictions'], 'test checkpoint/prediction binding mismatch')
            if 'protocol' in summary:
                require(summary['protocol'] == request['protocol'], 'summary protocol mismatch')
            for key in ('fold', 'group'):
                if key in summary:
                    require(type(summary[key]) is int and summary[key] == request['index'], 'summary fold/group mismatch')
            populations[role][arm] = rows
            report_arms[arm][role] = dict(native=check_summary(summary, rows, role, epoch),
                                         predictions=role_inputs['predictions'], summary=role_inputs['summary'])
        val_rows = populations['val'][arm]
        threshold = metrics.far_threshold([row['bonafide_score'] for row in val_rows], [row['label'] for row in val_rows], .05)
        thresholds[arm] = threshold
        report_arms[arm]['operating_point'] = dict(target_apcer=.05, threshold=threshold, frozen_on='source_val',
                                                 acceptance_rule='bonafide_score >= threshold', status='undefined' if threshold is None else 'defined')
        for role in ('val', 'test'):
            report_arms[arm][role]['operational'] = operational_rates(populations[role][arm], threshold)
    for path, checksum in consumed.items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == checksum, 'metadata changed during report')
    return dict(schema_version='current_matched_report_v1', protocol=request['protocol'], index=request['index'],
                authorization_granted=False, identities=shared_identity, arms=report_arms,
                score_policy='saved bonafide scores; native threshold .5; ties genuine; no sigmoid recomputation',
                membership_order='joined by unique image_id and sorted; input CSV order is not a pairing authority',
                paired_bootstrap=paired_bootstrap(populations, thresholds) if bootstrap else dict(status='not_requested'),
                metadata_sha256=consumed, limitations=LIMITATIONS)


def render_table(report):
    lines = ['| Arm | Role | N | Accuracy | AUC | Genuine FP | Frozen-val threshold | Test APCER | Test BPCER |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ('off', 'on'):
        value = report['arms'][arm]
        for role in ('val', 'test'):
            native = value[role]['native']
            operational = value[role]['operational'] if role == 'test' else dict(apcer=None, bpcer=None)
            lines.append(f"| {arm} | {role} | {native['n_images']} | {native['accuracy']} | {native['auc']} | "
                         f"{native['genuine_false_positive_count']} | {value['operating_point']['threshold']} | "
                         f"{operational['apcer']} | {operational['bpcer']} |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--bootstrap', action='store_true', help='1000 paired cluster resamples, seed 42')
    parser.add_argument('--out', type=Path, help='New JSON output; never overwrite an existing file')
    parser.add_argument('--format', choices=('json', 'table'), default='json')
    args = parser.parse_args(argv)
    try:
        result = build_report(dict(path=str(Path(args.request).absolute()), sha256=args.request_sha256), bootstrap=args.bootstrap)
        content = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
        if args.out is not None:
            with args.out.open('x') as stream:
                stream.write(content)
        print(content if args.format == 'json' else render_table(result), end='')
    except (ValueError, OSError, KeyError, TypeError, csv.Error) as error:
        parser.exit(2, f'matched report failed: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
