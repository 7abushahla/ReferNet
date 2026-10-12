"""Analyze completed OFF100 saved validation metadata; never authorize training."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_anchor_pilots import file_sha256, read_csv, require, same_metrics, selected_history
from refer_or_clear import metrics as metric_helpers
from refer_or_clear.data.query import load_role_rows
from refer_or_clear.pilots import ATOL, RTOL, compare_replay, summarize_predictions

LIMITATIONS = (
    "Analysis only, not training authorization or family selection. Query-component-only scores are from the same "
    "saved selected checkpoint, not an independently trained no-adapter ablation. No new inference, threshold, "
    "checkpoint selection, test pixels, or source/checkpoint tensor verification is performed. CPU float32 sigmoid "
    "rounding/saturation can produce score ties. Only 16 official fold-0 copy forgeries limit subgroup interpretation."
)
QUERY_METHOD = "CPU torch.float32 sigmoid(query_logit); bonafide_score=1-probability; ties predict genuine"


def build_report(control_binding, manifest_binding, *, relocations=(), metadata_paths=None):
    metadata_paths = metadata_paths or {}
    require(set(metadata_paths) <= {'records', 'split'}, "only explicit records/split CSV copies permitted")
    mappings = [(Path(original), Path(local)) for original, local in relocations]
    require(all(original.is_absolute() and local.is_absolute() and original != Path(original.anchor)
                and '..' not in original.parts + local.parts for original, local in mappings), "absolute explicit relocation required")
    for index, (original, local) in enumerate(mappings):
        for other, destination in mappings[index + 1:]:
            require(not original.is_relative_to(other) and not other.is_relative_to(original)
                    and not local.is_relative_to(destination) and not destination.is_relative_to(local), "ambiguous relocation")

    def located(value):
        path = Path(value)
        require(path.is_absolute() and '..' not in path.parts, "absolute artifact path required")
        for original, local in mappings:
            if path.is_relative_to(original):
                path = local / path.relative_to(original)
                break
        require(not any(parent.is_symlink() for parent in (path, *path.parents)), "symlink metadata path forbidden")
        return path

    consumed = {}

    def artifact(binding, expected=None):
        require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}
                and isinstance(binding['sha256'], str) and re.fullmatch('[0-9a-f]{64}', binding['sha256']), "external path/hash required")
        path = located(binding['path'])
        require(expected is None or path == located(expected), "unexpected artifact filename/path")
        require(file_sha256(path) == binding['sha256'], "artifact SHA256 mismatch")
        consumed[str(path)] = binding['sha256']
        return path

    def document(binding, expected=None):
        def pairs(entries):
            value = {}
            for name, item in entries:
                require(name not in value, "duplicate metadata JSON key")
                value[name] = item
            return value
        value = json.loads(artifact(binding, expected).read_bytes(), object_pairs_hook=pairs)
        json.dumps(value, allow_nan=False)
        require(isinstance(value, dict), "metadata JSON object required")
        return value

    def fixed(value, expected, message):
        require(all(type(value.get(name)) is type(item) and value[name] == item for name, item in expected.items()), message)

    completion, manifest = document(control_binding), document(manifest_binding)
    fixed(completion, dict(schema_version='direct_reference_control_result_v1', passed=True, phase='train', arm='off', epochs_completed=100),
          "completed passing OFF100 control required")
    fixed(manifest, dict(schema_version='direct_reference_job_v1', diagnostic_only=True), "original staged metadata required")
    run = Path(manifest['paths']['run_dir'])
    directory = run / 'training/off'
    artifact(control_binding, directory / 'control.json')
    artifact(manifest_binding, run / 'input_manifest.json')
    train = document(completion['train_result'], directory / 'train.json')
    fixed(train, dict(passed=True, phase='train', diagnostic_only=False, epochs_completed=100, optimizer_steps=22200, role='val', arm='off'),
          "complete selected OFF100 validation required")
    require(train.get('scheduler') == dict(total_updates=22200, warmup_updates=1000), "incomplete scheduler receipt")
    fixed(train.get('replay', {}), dict(passed=True, fresh_process=False, n_images=222, phase='train'), "in-process replay missing")
    identities = train['identities']
    require(identities == completion['identities'] and identities['source_sha256'] == manifest['source_sha256'], "mixed artifact identities")
    require(all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) for value in identities.values()), "invalid identity hash")
    checkpoint = train['checkpoint']
    require(checkpoint == completion['checkpoint'] and checkpoint.get('path') == str(directory / 'best.pt')
            and isinstance(checkpoint.get('sha256'), str) and re.fullmatch('[0-9a-f]{64}', checkpoint['sha256']), "checkpoint receipt binding mismatch")
    authorities = manifest['authorities']
    metadata = {}
    for name in ('records', 'split'):
        require(authorities[name]['sha256'] == identities[name + '_sha256'], "membership identity mismatch")
        require(Path(authorities[name]['path']).name == name + '.csv', "known metadata CSV filename required")
        csv_binding = dict(authorities[name], path=str(metadata_paths.get(name, authorities[name]['path'])))
        require(Path(csv_binding['path']).name == name + '.csv', "known local metadata CSV filename required")
        metadata[name] = BytesIO(artifact(csv_binding).read_bytes())
    members = load_role_rows(metadata['records'], metadata['split'], 'val').to_dict('records')
    predictions = read_csv(artifact(train['predictions'], directory / 'val_predictions.csv'))
    require(len(predictions) == 222 and [row['image_id'] for row in predictions] == [row['image_id'] for row in members],
            "exact ordered official VAL membership required")
    for row, member in zip(predictions, members):
        require(row.get('role', 'val') == 'val' and row['label'] in ('0', '1'), "VAL binary label required")
        row['label'] = int(row['label'])
        require(all(row[name] == member[name] for name in ('image_id', 'label', 'design_id', 'attack_family')), "prediction membership metadata mismatch")
        for name in ('attack_logit', 'query_logit', 'correction_logit', 'attack_probability', 'bonafide_score'):
            row[name] = float(row[name])
            require(math.isfinite(row[name]), "nonfinite saved prediction")
        for name in ('context_available', 'referral_required'):
            require(row[name] in ('True', 'False'), "boolean availability required")
            row[name] = row[name] == 'True'
        require(row['context_available'] is not row['referral_required'], "context/referral mismatch")
        probability = torch.tensor(row['attack_logit'], dtype=torch.float32).sigmoid().item()
        same_metrics({'attack_probability': probability, 'bonafide_score': 1. - probability}, row, 'saved score/logit')
        combined = (torch.tensor(row['query_logit'], dtype=torch.float32) + torch.tensor(row['correction_logit'], dtype=torch.float32)).item()
        require(abs(combined - row['attack_logit']) <= ATOL + RTOL * abs(row['attack_logit']), "saved logit components disagree")
        require(row['context_available'] or row['correction_logit'] == 0., "unavailable correction must be zero")
    combined_summary = summarize_predictions(predictions)
    history_path = directory / 'log.csv'
    history = read_csv(located(history_path))
    history_binding = dict(path=str(history_path), sha256=file_sha256(located(history_path)))
    artifact(history_binding)
    require(all(row.get('n_images') == '1776' and row.get('optimizer_steps') == '222' for row in history), "incomplete epoch history")
    selected_history(train, history)
    same_metrics(train['selection_key'], [combined_summary['accuracy'], combined_summary['auc']], "selected CSV metrics")
    replay = document(completion['fresh_replay'], directory / 'fresh-replay.json')
    fixed(replay, dict(schema_version='direct_reference_control_replay_v1', passed=True, fresh_process=True, phase='train', arm='off',
          n_images=222, buffers_unchanged=True, train_result=completion['train_result'], control_request=completion['control_request'],
          checkpoint=checkpoint, identities=identities, history=history_binding, image_ids=[row['image_id'] for row in predictions]), "complete bound fresh replay required")
    require(type(replay.get('pid')) is int and replay['pid'] > 0, "fresh replay PID required")
    comparison = compare_replay(predictions, replay.get('predictions', []))
    require(comparison['passed'] and comparison == replay.get('replay') and comparison['max_abs_difference'] == replay.get('max_abs_difference'), "saved replay comparison mismatch")
    for expected, actual in zip(predictions, replay['predictions']):
        for name, value in expected.items():
            if type(value) is float:
                require(type(actual.get(name)) in (int, float) and math.isclose(value, actual[name], rel_tol=RTOL, abs_tol=ATOL),
                        "saved replay numeric mismatch")
            else:
                require(type(actual.get(name)) is type(value) and actual[name] == value, "saved replay metadata mismatch")
    same_metrics(combined_summary, replay['summary'], "replay summary")
    require(replay['summary'].get('buffers_unchanged') is True, "replay buffer receipt missing")
    query_rows = []
    for row in predictions:
        probability = torch.tensor(row['query_logit'], dtype=torch.float32).sigmoid().item()
        query_rows.append(dict(row, attack_logit=row['query_logit'], attack_probability=probability, bonafide_score=1. - probability))

    def metrics(rows):
        summary = summarize_predictions(rows)
        apcer, bpcer = metric_helpers.apcer_bpcer([row['bonafide_score'] for row in rows], [row['label'] for row in rows], .5)
        return dict(summary, genuine_false_positive_rate=bpcer, altered_miss_rate=apcer)

    per_image = [dict(image_id=row['image_id'], label=row['label'], context_available=row['context_available'],
                      combined_bonafide_score=row['bonafide_score'], query_bonafide_score=query['bonafide_score'],
                      combined_prediction=int(row['bonafide_score'] < .5), query_prediction=int(query['bonafide_score'] < .5))
                 for row, query in zip(predictions, query_rows)]
    changed = [row['image_id'] for row in per_image if row['combined_prediction'] != row['query_prediction']]
    available = sum(row['context_available'] for row in predictions)
    require(all(file_sha256(Path(path)) == checksum for path, checksum in consumed.items()), "metadata changed during report")
    return dict(schema_version='direct_reference_control_report_v1', status='complete', role='selected_validation', arm='off',
                selected_epoch=train['selected_epoch'], identities=identities, control=control_binding, manifest=manifest_binding,
                competence=dict(accuracy_target=.97, auc_target=.99, passed=combined_summary['accuracy'] >= .97 and combined_summary['auc'] >= .99),
                combined=metrics(predictions), query_component_only=metrics(query_rows), query_score_method=QUERY_METHOD,
                context_availability=dict(available=available, unavailable=222 - available, total=222),
                context_subsets={name: metrics([row for row in predictions if row['context_available'] is enabled])
                                 for name, enabled in (('available', True), ('unavailable', False))},
                decision_changes=dict(count=len(changed), image_ids=changed), per_image=per_image,
                checkpoint_verification='receipt-binding-only; tensor bytes not opened', family_selection='not-authorized',
                metadata_sha256=consumed, metadata_path_overrides=metadata_paths,
                relocations=[dict(original=str(original), local=str(local)) for original, local in mappings], limitations=LIMITATIONS)


def render_markdown(report):
    lines = ['# Completed OFF100 saved-validation analysis', '', report['limitations'], '',
             f"Selected epoch {report['selected_epoch']}. Fixed competence targets met: {report['competence']['passed']}.", '',
             '| Saved component | Accuracy | AUC | Genuine FPR |', '|---|---|---|---|']
    for name in ('combined', 'query_component_only'):
        value = report[name]
        lines.append(f"| {name} | {value['accuracy']} | {value['auc']} | {value['genuine_false_positive_rate']} |")
    lines.extend(['', report['query_score_method'], '', f"Context availability: {json.dumps(report['context_availability'])}.",
                  f"Changed decisions: {report['decision_changes']['count']}.", '', '## Official combined subsets'])
    for name, value in report['combined']['subgroups'].items():
        lines.append(f'- {name}: {json.dumps(value, sort_keys=True)}')
    for name, value in report['context_subsets'].items():
        lines.append(f'- Context {name}: {json.dumps(value, sort_keys=True)}')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('control', 'control-sha256', 'manifest', 'manifest-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--relocate', nargs=2, action='append', default=[], metavar=('ORIGINAL', 'LOCAL'))
    parser.add_argument('--records', help='Explicit local records.csv copy, checked against manifest hash')
    parser.add_argument('--split', help='Explicit local split.csv copy, checked against manifest hash')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    markdown = args.out.with_suffix('.md')
    require(args.out != markdown and not args.out.exists() and not markdown.exists(), 'new output paths required')
    require(not any(path.is_symlink() for target in (args.out, markdown) for path in (target, *target.parents)), 'symlink output forbidden')
    result = build_report(dict(path=args.control, sha256=args.control_sha256), dict(path=args.manifest, sha256=args.manifest_sha256),
                          relocations=args.relocate, metadata_paths={name: getattr(args, name) for name in ('records', 'split') if getattr(args, name)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    with markdown.open('x') as stream:
        stream.write(render_markdown(result))
    return result


if __name__ == '__main__':
    main()
