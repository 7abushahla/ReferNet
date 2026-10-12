"""Optional figures from authenticated current-matched report/request pairs.

Pass --input REPORT REPORT_SHA REQUEST REQUEST_SHA once per completed unit.
Report validation is delegated to report_current_matched.build_report. No
actual figure is made without those inputs. Each fold/group stays separate,
with no cross-validation averaging or checkpoint selection. --roc adds test
curves from authenticated saved CSV scores, genuine as the positive class.
Matplotlib is a lazy runtime dependency from requirements-train.txt. Bootstrap
intervals describe paired differences, not arm-wise uncertainty, and are not
drawn as error bars. Existing output directories are never reused.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_current_matched as reporter


INTERPRETATION = (
    'Separate observed folds/groups, not a cross-validation mean. Fixed selected '
    'checkpoints and saved predictions only. Native error panels use the fixed '
    '.5 bona fide threshold; they are not transported operating-point errors. '
    'No bootstrap interval is presented as arm-wise uncertainty. No benchmark '
    'comparison, architecture selection, inference, or test authorization.'
)
COLORS = {'off': '#0072B2', 'on': '#D55E00'}


def authenticated(binding, suffix):
    reporter.validate_binding(binding)
    path = Path(binding['path'])
    reporter.require(path.suffix.lower() == suffix, 'explicit JSON/CSV metadata only')
    content = path.read_bytes()
    reporter.require(hashlib.sha256(content).hexdigest() == binding['sha256'], 'figure input SHA256 mismatch')
    return reporter.json_object(content) if suffix == '.json' else reporter.csv_rows(content)


def load_inputs(pairs, *, include_roc=False):
    reporter.require(pairs and type(include_roc) is bool, 'at least one explicit report/request pair required')
    validated, seen = [], set()
    for report_binding, request_binding in pairs:
        report = authenticated(report_binding, '.json')
        current = reporter.build_report(request_binding, bootstrap=False)
        reporter.require({key: value for key, value in report.items() if key != 'paired_bootstrap'} ==
                         {key: value for key, value in current.items() if key != 'paired_bootstrap'},
                         'frozen report differs from authenticated current metadata')
        unit = (report['protocol'], report['index'])
        reporter.require(unit not in seen, 'duplicate protocol/fold/group unit')
        seen.add(unit)
        kind = 'group' if report['protocol'] == 'B' else 'fold'
        entry = dict(report=report, label=f"{report['protocol']} {kind} {report['index']}",
                     report_binding=report_binding, request_binding=request_binding, roc={})
        if include_roc:
            from sklearn.metrics import roc_curve
            for arm in ('off', 'on'):
                rows = authenticated(report['arms'][arm]['test']['predictions'], '.csv')
                labels = [reporter.binary_label(row['label']) for row in rows]
                if set(labels) != {0, 1}:
                    entry['roc'][arm] = dict(status='undefined', reason='both classes required')
                    continue
                scores = [reporter.finite_probability(row['bonafide_score']) for row in rows]
                false_positive, true_positive, unused_thresholds = roc_curve([1 - label for label in labels], scores)
                entry['roc'][arm] = dict(status='defined', x=false_positive.tolist(), y=true_positive.tolist(),
                    auc=report['arms'][arm]['test']['native']['auc'],
                    x_label='Altered accepted (APCER)', y_label='Genuine accepted (1 - BPCER)')
        validated.append(entry)
    return sorted(validated, key=lambda value: (value['report']['protocol'], value['report']['index']))


def plotting_runtime():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as pyplot
    except ImportError as error:
        raise ValueError('Matplotlib is required for rendering (requirements-train.txt); validation needs no plotting runtime') from error
    return pyplot


def unit_axis(axis, title, ylabel):
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, 1)
    axis.grid(axis='y', alpha=.2)
    axis.set_axisbelow(True)


def paired_bars(axis, labels, values, annotations=None):
    positions = list(range(len(labels)))
    width = .36
    for arm, offset in (('off', -width / 2), ('on', width / 2)):
        observed = [(position + offset, value, index) for index, (position, value) in enumerate(zip(positions, values[arm])) if value is not None]
        axis.bar([position for position, value, index in observed], [value for position, value, index in observed],
                 width=width, label=arm.upper(), color=COLORS[arm])
        for position, value, index in observed:
            label = f'{value:.3f}' if annotations is None else annotations[arm][index]
            axis.text(position, max(.035, value - .035), label, ha='center', va='top', fontsize=8,
                      rotation=90 if annotations is not None else 0)
        for index, value in enumerate(values[arm]):
            if value is None:
                axis.text(index + offset, .02, 'N/A', ha='center', va='bottom', fontsize=8, rotation=90)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.legend(frameon=False)


def performance_figure(entries, plotting):
    figure, axes = plotting.subplots(2, 2, figsize=(max(8, len(entries) * 2.5), 7), squeeze=False, constrained_layout=True)
    labels = [entry['label'] for entry in entries]
    for row_index, role in enumerate(('val', 'test')):
        title = 'Source validation' if role == 'val' else 'Scored test queries'
        for column_index, name in enumerate(('accuracy', 'auc')):
            axis = axes[row_index][column_index]
            unit_axis(axis, title, 'Native accuracy (.5)' if name == 'accuracy' else 'ROC AUC')
            values = {arm: [entry['report']['arms'][arm][role]['native'][name] for entry in entries] for arm in ('off', 'on')}
            paired_bars(axis, labels, values)
    figure.suptitle('Matched OFF / ON, individual units')
    return figure


def error_rates(native):
    denominator = native['n_genuine']
    numerator = native['genuine_false_positive_count']
    rates = dict(genuine=(numerator / denominator, f'{numerator}/{denominator}') if denominator else (None, 'N/A (n=0)'))
    for name in ('copy', 'inpainting'):
        subgroup = native['subgroups'][name]
        denominator, numerator = subgroup['n_altered'], subgroup['altered_miss_count']
        rates[name] = (numerator / denominator, f'{numerator}/{denominator}') if denominator else (None, 'N/A (n=0)')
    return rates


def errors_figure(entries, plotting):
    figure, axis = plotting.subplots(figsize=(max(9, len(entries) * 4), 4), constrained_layout=True)
    values, counts = {arm: [] for arm in ('off', 'on')}, {arm: [] for arm in ('off', 'on')}
    labels = []
    for entry in entries:
        for name, label in (('genuine', 'Genuine FP'), ('copy', 'Copy miss'), ('inpainting', 'Inpainting miss')):
            labels.append(f"{entry['label']}\n{label}")
            for arm in ('off', 'on'):
                rate, count = error_rates(entry['report']['arms'][arm]['test']['native'])[name]
                values[arm].append(rate)
                counts[arm].append(count)
    unit_axis(axis, 'Scored test errors at native bona fide threshold .5', 'Error fraction (labels show numerator/denominator)')
    paired_bars(axis, labels, values, counts)
    return figure


def roc_figure(entries, plotting):
    figure, axis = plotting.subplots(figsize=(6, 5), constrained_layout=True)
    unit_axis(axis, 'Scored test ROC, genuine-positive convention', 'Genuine accepted (1 - BPCER)')
    axis.set_xlim(0, 1)
    axis.set_xlabel('Altered accepted (APCER)')
    for entry in entries:
        for arm in ('off', 'on'):
            curve = entry['roc'].get(arm, {})
            if curve.get('status') == 'defined':
                axis.plot(curve['x'], curve['y'], label=f"{entry['label']} {arm.upper()} (AUC {curve['auc']:.3f})",
                          color=COLORS[arm], linestyle='--' if arm == 'off' else '-')
    axis.legend(frameon=False, fontsize=8)
    return figure


def render(entries, output, *, formats=('svg', 'png')):
    reporter.require(entries and formats and len(set(formats)) == len(formats) and set(formats) <= {'svg', 'png'},
                     'unique SVG/PNG formats required')
    output = Path(output)
    reporter.require(not output.exists(), 'new output directory required; never overwrite figures')
    plotting = plotting_runtime()
    output.mkdir(parents=True, exist_ok=False)
    builders = [('performance', performance_figure), ('native-errors', errors_figure)]
    if any(entry['roc'] for entry in entries):
        builders.append(('test-roc', roc_figure))
    files = []
    for name, builder in builders:
        figure = builder(entries, plotting)
        try:
            for suffix in formats:
                path = output / f'{name}.{suffix}'
                figure.savefig(path, format=suffix, dpi=300, bbox_inches='tight')
                files.append(dict(path=str(path.absolute()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        finally:
            plotting.close(figure)
    receipt = dict(schema_version='current_matched_figures_v1', authorization_granted=False,
                   inputs=[dict(report=entry['report_binding'], request=entry['request_binding'], unit=entry['label']) for entry in entries],
                   files=files, interpretation=INTERPRETATION, runtime_validation='authenticated metadata; no checkpoint/pixel access')
    with (output / 'figures.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--input', nargs=4, action='append', required=True,
                        metavar=('REPORT', 'REPORT_SHA', 'REQUEST', 'REQUEST_SHA'))
    parser.add_argument('--roc', action='store_true', help='Test ROC from authenticated saved CSV scores')
    parser.add_argument('--formats', nargs='+', choices=('svg', 'png'), default=['svg', 'png'])
    parser.add_argument('--out', type=Path, required=True, help='New figure directory')
    args = parser.parse_args(argv)
    try:
        pairs = [(dict(path=str(Path(report).absolute()), sha256=report_sha),
                  dict(path=str(Path(request).absolute()), sha256=request_sha))
                 for report, report_sha, request, request_sha in args.input]
        entries = load_inputs(pairs, include_roc=args.roc)
        receipt = render(entries, args.out, formats=tuple(args.formats))
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError, csv.Error) as error:
        print(f'matched figures failed: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
