import importlib.util
from pathlib import Path

import pytest

from test_report_current_matched import bundle, reporter, request_binding, document, binding


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plotter():
    path = ROOT / 'scripts/plot_current_matched.py'
    assert path.is_file(), 'matched figure generator is not implemented'
    specification = importlib.util.spec_from_file_location('plot_current_matched', path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(reporter, bundle):
    request = request_binding(bundle)
    report = reporter.build_report(request)
    report_binding = document(bundle['root'] / 'report.json', report)
    return dict(bundle=bundle, report=report, report_binding=report_binding, request=request)


def test_plotter_module_exists(plotter):
    assert callable(plotter.load_inputs)


def test_load_reuses_report_validation_and_no_plotting_import(plotter, inputs, monkeypatch):
    calls = []
    original = plotter.reporter.build_report
    def recorded(request, **kwargs):
        calls.append((request, kwargs))
        return original(request, **kwargs)
    monkeypatch.setattr(plotter.reporter, 'build_report', recorded)
    validated = plotter.load_inputs([(inputs['report_binding'], inputs['request'])], include_roc=True)
    assert calls == [(inputs['request'], dict(bootstrap=False))]
    assert validated[0]['label'] == 'B group 0'
    assert validated[0]['report'] == inputs['report']
    assert validated[0]['roc']['off']['auc'] == .75
    assert validated[0]['roc']['on']['auc'] == 1
    assert validated[0]['roc']['off']['x_label'] == 'Altered accepted (APCER)'
    assert validated[0]['roc']['off']['y_label'] == 'Genuine accepted (1 - BPCER)'


@pytest.mark.parametrize('failure', ['report-hash', 'report-number', 'csv-hash', 'request-hash', 'duplicate'])
def test_invalid_inputs_rejected_before_rendering(plotter, inputs, failure):
    pair = (inputs['report_binding'], inputs['request'])
    pairs = [pair]
    if failure == 'report-hash':
        pair[0]['sha256'] = '0' * 64
    elif failure == 'report-number':
        inputs['report']['arms']['on']['test']['native']['accuracy'] = .1
        pair = (document(Path(pair[0]['path']), inputs['report']), pair[1])
        pairs = [pair]
    elif failure == 'csv-hash':
        path = Path(inputs['report']['arms']['on']['test']['predictions']['path'])
        path.write_text(path.read_text() + '\nchanged')
    elif failure == 'request-hash':
        pair[1]['sha256'] = '0' * 64
    else:
        pairs.append(pair)
    with pytest.raises(ValueError):
        plotter.load_inputs(pairs)


def test_missing_attack_rates_are_undefined_not_zero(plotter, inputs):
    native = inputs['report']['arms']['off']['test']['native']
    assert plotter.error_rates(native)['copy'] == (.0, '0/1')
    assert plotter.error_rates(native)['inpainting'] == (1., '1/1')
    native['subgroups']['copy'].update(n_altered=0, altered_miss_count=0)
    assert plotter.error_rates(native)['copy'] == (None, 'N/A (n=0)')


class Axis:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def recorded(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return recorded


class Figure:
    def __init__(self):
        self.paths = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

    def savefig(self, path, **kwargs):
        self.paths.append(Path(path))
        Path(path).write_bytes(b'mocked synthetic figure, not an outcome')


class Plotting:
    def __init__(self):
        self.axes, self.figures, self.closed = [], [], []

    def subplots(self, rows=1, columns=1, **kwargs):
        figure = Figure()
        axes = [[Axis() for unused in range(columns)] for unused in range(rows)]
        self.figures.append(figure)
        self.axes.extend(axis for row in axes for axis in row)
        return figure, axes if kwargs.get('squeeze') is False else axes[0][0]

    def close(self, figure):
        self.closed.append(figure)


def test_render_full_axes_separate_units_native_errors_and_no_ci(plotter, inputs, tmp_path, monkeypatch):
    plots = Plotting()
    monkeypatch.setattr(plotter, 'plotting_runtime', lambda: plots)
    validated = plotter.load_inputs([(inputs['report_binding'], inputs['request'])], include_roc=True)
    before = {path: path.read_bytes() for path in inputs['bundle']['root'].iterdir() if path.is_file()}
    output = tmp_path / 'figures'
    result = plotter.render(validated, output, formats=('svg', 'png'))
    assert len(result['files']) == 6
    assert all(Path(item['path']).is_file() and binding(Path(item['path'])) == item for item in result['files'])
    assert len(plots.closed) == len(plots.figures) == 3
    assert all(('set_ylim', (0, 1), {}) in axis.calls for axis in plots.axes)
    assert not any(name == 'errorbar' for axis in plots.axes for name, args, kwargs in axis.calls)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert 'not a cross-validation mean' in result['interpretation']
    assert result['authorization_granted'] is False
    with pytest.raises(ValueError):
        plotter.render(validated, output)


def test_bad_input_creates_no_output_directory(plotter, inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(plotter, 'plotting_runtime', lambda: pytest.fail('must authenticate before rendering'))
    output = tmp_path / 'never-created'
    result = plotter.main(['--input', inputs['report_binding']['path'], '0' * 64,
                          inputs['request']['path'], inputs['request']['sha256'], '--out', str(output)])
    assert result == 2
    assert not output.exists()


def test_two_units_stay_separate_without_averaging(plotter, inputs):
    second = dict(inputs['report'], protocol='A')
    second['metadata_sha256'] = dict(inputs['report']['metadata_sha256'])
    first_pair = (inputs['report_binding'], inputs['request'])
    initial = plotter.load_inputs([first_pair])
    assert len(initial) == 1
    initial.append(dict(report=second, label='A fold 0', report_binding=inputs['report_binding'], request_binding=inputs['request'], roc={}))
    plots = Plotting()
    plotter.performance_figure(initial, plots)
    accuracy_bars = [args[1] for name, args, kwargs in plots.axes[0].calls if name == 'bar']
    assert accuracy_bars == [[.75, .75], [1., 1.]]


def test_missing_plotting_runtime_creates_no_output(plotter, inputs, tmp_path, monkeypatch):
    def missing():
        raise ValueError('Matplotlib is required for rendering')
    monkeypatch.setattr(plotter, 'plotting_runtime', missing)
    validated = plotter.load_inputs([(inputs['report_binding'], inputs['request'])])
    output = tmp_path / 'missing-runtime'
    with pytest.raises(ValueError, match='Matplotlib'):
        plotter.render(validated, output)
    assert not output.exists()


@pytest.mark.parametrize('formats', [(), ('pdf',), ('svg', 'svg')])
def test_bad_formats_fail_before_runtime_or_writes(plotter, inputs, tmp_path, monkeypatch, formats):
    monkeypatch.setattr(plotter, 'plotting_runtime', lambda: pytest.fail('formats must validate first'))
    validated = plotter.load_inputs([(inputs['report_binding'], inputs['request'])])
    output = tmp_path / 'invalid-format'
    with pytest.raises(ValueError):
        plotter.render(validated, output, formats=formats)
    assert not output.exists()
