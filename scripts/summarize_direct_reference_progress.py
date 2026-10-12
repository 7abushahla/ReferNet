"""Observe known run metadata without inference, test reads, or authorization.

Pass training/off, training/on, or a global-only output directory. A staged run
root expands only its existing training/off and training/on directories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


FAMILIES = {'copy': 'text_copy', 'inpaint': 'text_inpaint'}
COMPLETION_SCHEMAS = {
    'direct_reference_control_result_v1', 'direct_reference_on_result_v1',
    'direct_reference_on_study_result_v1', 'direct_reference_global_result_v1',
}


def finite_number(value, *, probability=False):
    if isinstance(value, bool):
        raise ValueError('boolean is not a numeric metric')
    number = float(value)
    if not math.isfinite(number) or (probability and not 0 <= number <= 1):
        raise ValueError('invalid finite metric range')
    return number


def auc(rows):
    genuine = sum(row['label'] == 0 for row in rows)
    altered = len(rows) - genuine
    if not genuine or not altered:
        return None
    ordered = sorted(rows, key=lambda row: row['bonafide_score'])
    preceding_altered, favorable, index = 0, 0., 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end]['bonafide_score'] == ordered[index]['bonafide_score']:
            end += 1
        tied = ordered[index:end]
        tied_genuine = sum(row['label'] == 0 for row in tied)
        tied_altered = len(tied) - tied_genuine
        favorable += tied_genuine * (preceding_altered + tied_altered / 2)
        preceding_altered += tied_altered
        index = end
    return favorable / (genuine * altered)


def prediction_summary(rows):
    genuine = [row for row in rows if row['label'] == 0]
    altered = [row for row in rows if row['label'] == 1]
    false_positives = [row['image_id'] for row in genuine if row['bonafide_score'] < .5]
    misses = [row['image_id'] for row in altered if row['bonafide_score'] >= .5]
    result = dict(n_images=len(rows), n_genuine=len(genuine), n_altered=len(altered),
                  accuracy=1 - (len(false_positives) + len(misses)) / len(rows), auc=auc(rows),
                  genuine_false_positives=false_positives, altered_misses=misses,
                  genuine_false_positive_rate=len(false_positives) / len(genuine) if genuine else None,
                  altered_miss_rate=len(misses) / len(altered) if altered else None, subgroups={})
    for name, family in FAMILIES.items():
        members = [row for row in altered if row['attack_family'] == family]
        population = genuine + members
        correct = sum((row['bonafide_score'] < .5) == bool(row['label']) for row in population)
        result['subgroups'][name] = dict(n_genuine=len(genuine), n_altered=len(members), auc=auc(population),
                                        accuracy=correct / len(population) if genuine and members else None)
    return result


def summarize_progress(directory, *, total_epochs=100):
    if type(total_epochs) is not int or total_epochs <= 0:
        raise ValueError('positive integer total_epochs required')
    directory = Path(directory).absolute()
    issues, evidence = [], {}

    def read(name):
        path = directory / name
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            issues.append(f'{name}: symlink metadata not read')
            return None
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as error:
            issues.append(f'{name}: metadata temporarily unreadable ({type(error).__name__})')
            return None
        evidence[name] = hashlib.sha256(payload).hexdigest()
        try:
            return payload.decode('utf-8-sig')
        except UnicodeError:
            issues.append(f'{name}: incomplete or invalid UTF-8 metadata')
            return None

    def rows(name, required):
        text = read(name)
        if text is None:
            return []
        parsed = []
        try:
            reader = csv.DictReader(io.StringIO(text), strict=True)
            fields = reader.fieldnames or []
            if len(set(fields)) != len(fields) or not required <= set(fields):
                raise ValueError('incomplete or invalid CSV header')
            for row in reader:
                parsed.append(row)
            if text and not text.endswith(('\n', '\r')) and parsed:
                parsed[-1] = {}
            return parsed
        except (csv.Error, ValueError) as error:
            issues.append(f'{name}: {error}; transfer or writer may be in progress')
            return parsed + [{}] if parsed else []

    def document(name):
        text = read(name)
        if text is None:
            return None

        def pairs(entries):
            value = {}
            for key, item in entries:
                if key in value:
                    raise ValueError('duplicate JSON key')
                value[key] = item
            return value

        try:
            value = json.loads(text, object_pairs_hook=pairs)
            json.dumps(value, allow_nan=False)
            if not isinstance(value, dict):
                raise ValueError('JSON object required')
            return value
        except (ValueError, TypeError):
            issues.append(f'{name}: incomplete or invalid JSON; transfer or writer may be in progress')
            return None

    complete_rows, selected = [], None
    for row in rows('log.csv', {'epoch', 'val_accuracy', 'val_auc'}):
        try:
            number = int(row['epoch'])
            if number != len(complete_rows) + 1 or number > total_epochs or None in row or any(value is None for value in row.values()):
                raise ValueError('noncontiguous or incomplete epoch')
            accuracy = finite_number(row['val_accuracy'], probability=True)
            area = finite_number(row['val_auc'], probability=True)
            complete_rows.append(row)
            if selected is None or (accuracy, area) > (selected['accuracy'], selected['auc']):
                selected = dict(epoch=number, accuracy=accuracy, auc=area, provisional=True)
        except (ValueError, TypeError, KeyError):
            issues.append('log.csv: incomplete or malformed epoch; only the preceding contiguous prefix is reported')
            break
    elapsed = dict(seconds=None, source=None, scope='unavailable')
    if complete_rows and all('seconds' in row for row in complete_rows):
        try:
            durations = [finite_number(row['seconds']) for row in complete_rows]
            if any(value < 0 for value in durations):
                raise ValueError('negative duration')
            elapsed = dict(seconds=finite_number(sum(durations)), source='log.csv:sum(seconds)', scope='completed_epochs_only')
        except (ValueError, TypeError):
            issues.append('log.csv: elapsed duration unavailable; timestamps are not substituted')

    documents = {name: document(name) for name in ('train.json', 'control.json', 'on.json', 'result.json', 'val_summary.json', 'completion.json')}
    chosen_receipt = documents['train.json'] or documents['result.json'] or documents['val_summary.json']
    recorded_epoch = None
    if chosen_receipt is not None and 'selected_epoch' in chosen_receipt:
        candidate = chosen_receipt['selected_epoch']
        if chosen_receipt.get('role', 'val') != 'val' or type(candidate) is not int or selected is None or candidate != selected['epoch']:
            issues.append('selected validation receipt disagrees with observed history; not independently authenticated')
        else:
            recorded_epoch = candidate
    job_completion = documents['completion.json']
    if job_completion is not None and 'elapsed_seconds' in job_completion:
        try:
            duration = finite_number(job_completion['elapsed_seconds'])
            if duration < 0:
                raise ValueError('negative duration')
            elapsed = dict(seconds=duration, source='completion.json:elapsed_seconds', scope='recorded_job_duration')
        except (ValueError, TypeError):
            issues.append('completion.json: invalid elapsed_seconds ignored')

    prediction_rows = rows('val_predictions.csv', {'image_id', 'label', 'attack_family', 'bonafide_score'})
    parsed_predictions, identifiers = [], set()
    metrics = None
    if prediction_rows:
        try:
            for row in prediction_rows:
                if None in row or any(value is None for value in row.values()) or row.get('role', 'val') != 'val':
                    raise ValueError('incomplete or non-validation prediction')
                identifier = row['image_id']
                if not identifier or identifier in identifiers or row['label'] not in ('0', '1'):
                    raise ValueError('invalid or repeated prediction identity/label')
                label = int(row['label'])
                if label == 1 and row['attack_family'] not in FAMILIES.values():
                    raise ValueError('unknown attack subgroup')
                parsed_predictions.append(dict(image_id=identifier, label=label, attack_family=row['attack_family'],
                                               bonafide_score=finite_number(row['bonafide_score'], probability=True)))
                identifiers.add(identifier)
            metrics = prediction_summary(parsed_predictions)
        except (ValueError, TypeError, KeyError):
            issues.append('val_predictions.csv: incomplete, malformed, or non-validation rows; metrics withheld')

    completion = None
    for name in ('control.json', 'on.json', 'result.json'):
        receipt = documents[name]
        if receipt is not None:
            completion = dict(path=str(directory / name), schema_version=receipt.get('schema_version'), passed_recorded=receipt.get('passed') is True)
            break
    finished = (completion is not None and completion['schema_version'] in COMPLETION_SCHEMAS
                and completion['passed_recorded'] and type(receipt.get('epochs_completed')) is int
                and receipt['epochs_completed'] == total_epochs and len(complete_rows) == total_epochs
                and chosen_receipt is not None and chosen_receipt.get('passed') is True
                and type(chosen_receipt.get('epochs_completed')) is int and chosen_receipt['epochs_completed'] == total_epochs
                and recorded_epoch is not None and not issues)
    if selected is not None:
        selected['provisional'] = not finished
    status = 'completion_recorded' if finished else 'in_progress' if evidence or issues else 'pending'
    arm = directory.name if directory.name in ('off', 'on') else (chosen_receipt or {}).get('arm', 'unknown')
    return dict(path=str(directory), arm=arm, status=status, epochs_completed=len(complete_rows), total_epochs=total_epochs,
                selected_validation=selected, validation_predictions=dict(metrics=metrics, selected_epoch=recorded_epoch,
                decision_rule='stored bonafide_score < 0.5 predicts altered; ties genuine', membership='not independently authenticated'),
                test=dict(status='not_read'), elapsed=elapsed, completion=completion, issues=issues, metadata_sha256=evidence,
                authorization='none; observation only', limitations='No checkpoint, pixels, inference, threshold selection, job-state verification, or external authority verification. Missing or partial metadata is not a scientific failure.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--format', choices=('json', 'text'), default='json')
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error('--epochs must be positive')
    reports = []
    for path in args.paths:
        directories = [path / 'training' / arm for arm in ('off', 'on') if (path / 'training' / arm).is_dir()]
        for directory in directories or [path]:
            reports.append(summarize_progress(directory, total_epochs=args.epochs))
    if args.format == 'json':
        print(json.dumps(dict(schema_version='direct_reference_progress_v1', runs=reports), allow_nan=False))
    else:
        for report in reports:
            selected = report['selected_validation']
            metrics = f"epoch={selected['epoch']} accuracy={selected['accuracy']:.6f} AUC={selected['auc']:.6f}" if selected else 'unavailable'
            print(f"{report['path']} {report['status']} {report['epochs_completed']}/{report['total_epochs']} selected validation {metrics}; warnings={len(report['issues'])}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
