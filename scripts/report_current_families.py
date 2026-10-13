"""Descriptive family errors from the frozen matched reporter's authenticated inputs.

Use the same current_matched_report_request_v1 and external request SHA256.
No scientific code, images, checkpoints, target threshold selection, or confidence
intervals are involved. All rates here are classifier-only, not deployment
clearance rates with mandatory context referral.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_current_matched as reporter


def _read(binding):
    reporter.validate_binding(binding)
    path = Path(binding['path'])
    reporter.require(path.suffix.lower() in ('.json', '.csv'), 'JSON/CSV metadata only')
    content = path.read_bytes()
    reporter.require(hashlib.sha256(content).hexdigest() == binding['sha256'], 'metadata SHA256 mismatch')
    return reporter.json_object(content) if path.suffix.lower() == '.json' else reporter.csv_rows(content)


def _errors(rows, threshold, *, altered):
    name = 'altered_miss' if altered else 'false_positive'
    count = None if threshold is None else sum(
        row['bonafide_score'] >= threshold if altered else row['bonafide_score'] < threshold for row in rows)
    rate = None if count is None or not rows else count / len(rows)
    return {name + '_count': count, name + '_rate': rate}


def _population(rows, threshold):
    genuine = [row for row in rows if row['label'] == 0]
    altered = [row for row in rows if row['label'] == 1]
    families = {}
    for family in sorted({row['attack_family'] for row in altered}):
        attacks = [row for row in altered if row['attack_family'] == family]
        population = genuine + attacks
        families[family] = dict(n_altered=len(attacks), n_genuine_shared=len(genuine),
            n_images=len(population), family_vs_all_genuine_auc=reporter.metrics.auc(
                [row['bonafide_score'] for row in population], [row['label'] for row in population]),
            native=_errors(attacks, .5, altered=True), transported=_errors(attacks, threshold, altered=True))
    return dict(n_images=len(rows), n_altered=len(altered), families=families,
        genuine=dict(n_genuine=len(genuine), native=_errors(genuine, .5, altered=False),
                     transported=_errors(genuine, threshold, altered=False)))


def build_family_report(request_binding):
    request = _read(request_binding)
    validated = reporter.build_report(request_binding, bootstrap=False)
    records = reporter.unique_rows(_read(request['records']), 'records')
    arms = {}
    for arm in ('off', 'on'):
        selected = validated['arms'][arm]
        threshold = selected['operating_point']['threshold']
        arms[arm] = dict(selected_epoch=selected['selected_epoch'], checkpoint=selected['checkpoint'],
                         operating_point=selected['operating_point'])
        for role in ('val', 'test'):
            saved = _read(request['arms'][arm][role]['predictions'])
            rows = [dict(image_id=row['image_id'], label=reporter.binary_label(records[row['image_id']]['label']),
                attack_family=records[row['image_id']]['attack_family'],
                bonafide_score=reporter.finite_probability(row['bonafide_score'])) for row in saved]
            arms[arm][role] = _population(rows, threshold)
    for path, checksum in validated['metadata_sha256'].items():
        _read(dict(path=path, sha256=checksum))
    return dict(schema_version='current_matched_family_report_v1', protocol=validated['protocol'],
        index=validated['index'], authorization_granted=False, request=request_binding,
        validated_report_schema=validated['schema_version'], identities=validated['identities'],
        metadata_sha256=validated['metadata_sha256'], arms=arms, native_threshold=.5,
        score_policy='Classifier-only saved bonafide scores; >= threshold accepts genuine, including ties. '
                     'Transported thresholds are frozen per arm on source validation at 5% APCER, never per family or TEST.',
        auc_orientation='Genuine positive; every scored genuine query versus that recorded altered family.',
        family_policy='All observed recorded altered families; exact names, including empty names; no merging or outcome selection.',
        limitations=['Shared genuine queries recur in every family AUC denominator and are not independent family evidence.',
            'Undefined threshold gives undefined transported counts/rates; AUC requires both classes.',
            'Descriptive conditional-on-checkpoint results, not independent confidence intervals or significance tests.',
            'Design mixtures and recorded-component lineage limits remain; no new source, checkpoint or product byte proof.',
            'These are not mandatory-referral deployment metrics. The frozen matched reporter reports that policy separately.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    try:
        result = build_family_report(dict(path=args.request, sha256=args.request_sha256))
        payload = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n'
        if args.out is not None:
            with args.out.open('x') as stream:
                stream.write(payload)
        print(payload, end='')
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f'family reporting failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
