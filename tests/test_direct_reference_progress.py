import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/summarize_direct_reference_progress.py'


@pytest.fixture
def progress():
    assert SCRIPT.is_file(), 'metadata-only progress script missing'
    specification = importlib.util.spec_from_file_location('direct_reference_progress_test', SCRIPT)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def history(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def epoch(number, accuracy=.8, auc=.9, **extra):
    return dict(epoch=number, loss=.2, val_accuracy=accuracy, val_auc=auc, val_image_loss=.3, **extra)


def predictions(path, role=None):
    rows = [dict(image_id=identifier, label=label, attack_family=family, bonafide_score=score)
            for identifier, label, family, score in (
                ('genuine_fp', 0, '', .4), ('genuine_tie', 0, '', .5),
                ('copy_tie', 1, 'text_copy', .5), ('inpaint_detected', 1, 'text_inpaint', .2))]
    if role is not None:
        rows = [dict(row, role=role) for row in rows]
    history(path, rows)


def test_selection_is_accuracy_first_then_auc_with_exact_earlier_ties(progress, tmp_path):
    history(tmp_path / 'log.csv', [epoch(1, .8, .99), epoch(2, .9, .7), epoch(3, .9, .8), epoch(4, .9, .8)])
    result = progress.summarize_progress(tmp_path)
    assert result['epochs_completed'] == 4
    assert result['selected_validation'] == dict(epoch=3, accuracy=.9, auc=.8, provisional=True)
    assert result['status'] == 'in_progress'
    assert result['elapsed']['seconds'] is None


def test_close_but_not_exact_ties_remain_distinct(progress, tmp_path):
    history(tmp_path / 'log.csv', [epoch(1, .9, .8), epoch(2, .9000000000000001, .7)])
    assert progress.summarize_progress(tmp_path)['selected_validation']['epoch'] == 2


@pytest.mark.parametrize('trailing', ['5,.2,.9', '5,.2,nan,.9,.3\n', '5,.2,.9,2,.3\n', 'wrong,.2,.9,.9,.3\n'])
def test_partial_or_malformed_tail_preserves_completed_prefix_as_in_progress(progress, tmp_path, trailing):
    history(tmp_path / 'log.csv', [epoch(number) for number in range(1, 5)])
    with (tmp_path / 'log.csv').open('a') as stream:
        stream.write(trailing)
    result = progress.summarize_progress(tmp_path)
    assert result['epochs_completed'] == 4 and result['status'] == 'in_progress'
    assert result['issues'] and result['selected_validation']['epoch'] == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('rows', [[epoch(2)], [epoch(1), epoch(3)], [epoch(1), epoch(1)]])
def test_noncontiguous_or_duplicate_epochs_do_not_inflate_progress(progress, tmp_path, rows):
    history(tmp_path / 'log.csv', rows)
    result = progress.summarize_progress(tmp_path)
    assert result['epochs_completed'] <= 1 and result['issues']


@pytest.mark.parametrize('content', ['', 'epoch,val_accuracy', 'epoch,epoch,val_accuracy,val_auc\n1,1,.9,.9\n'])
def test_inflight_empty_or_bad_header_does_not_crash(progress, tmp_path, content):
    (tmp_path / 'log.csv').write_text(content)
    result = progress.summarize_progress(tmp_path)
    assert result['epochs_completed'] == 0 and result['status'] == 'in_progress'
    assert result['selected_validation'] is None


def test_missing_run_is_pending_without_creating_files(progress, tmp_path):
    missing = tmp_path / 'not transferred yet'
    result = progress.summarize_progress(missing)
    assert result['status'] == 'pending' and result['epochs_completed'] == 0
    assert not missing.exists()


def test_predictions_use_saved_scores_fixed_tie_rule_and_subgroups(progress, tmp_path):
    history(tmp_path / 'log.csv', [epoch(1)])
    predictions(tmp_path / 'val_predictions.csv')
    result = progress.summarize_progress(tmp_path)
    summary = result['validation_predictions']['metrics']
    assert summary['accuracy'] == .5 and summary['auc'] == .625
    assert summary['genuine_false_positives'] == ['genuine_fp']
    assert summary['altered_misses'] == ['copy_tie']
    assert summary['subgroups']['copy']['n_genuine'] == 2
    assert summary['subgroups']['copy']['n_altered'] == 1
    assert summary['subgroups']['copy']['accuracy'] == pytest.approx(1 / 3)
    assert summary['subgroups']['inpaint']['accuracy'] == pytest.approx(2 / 3)
    assert result['validation_predictions']['selected_epoch'] is None
    assert result['test'] == dict(status='not_read')


@pytest.mark.parametrize('failure', ['test-role', 'bad-score', 'duplicate-id', 'partial-row'])
def test_invalid_predictions_are_not_reported_as_complete_metrics(progress, tmp_path, failure):
    predictions(tmp_path / 'val_predictions.csv', role='test' if failure == 'test-role' else None)
    path = tmp_path / 'val_predictions.csv'
    content = path.read_text()
    if failure == 'bad-score': content = content.replace('0.4', 'nan')
    if failure == 'duplicate-id': content = content.replace('genuine_tie', 'genuine_fp')
    if failure == 'partial-row': content += 'unfinished,1'
    path.write_text(content)
    result = progress.summarize_progress(tmp_path)
    assert result['validation_predictions']['metrics'] is None and result['issues']
    assert result['status'] == 'in_progress'


def test_empty_context_and_single_class_predictions_have_undefined_auc(progress, tmp_path):
    history(tmp_path / 'val_predictions.csv', [dict(image_id='genuine', label=0, attack_family='', bonafide_score=.5)])
    summary = progress.summarize_progress(tmp_path)['validation_predictions']['metrics']
    assert summary['auc'] is None and summary['accuracy'] == 1.
    assert summary['subgroups']['copy']['accuracy'] is None


def test_only_recorded_elapsed_is_used_not_mtime(progress, tmp_path):
    history(tmp_path / 'log.csv', [epoch(1, seconds=10), epoch(2, seconds=12)])
    result = progress.summarize_progress(tmp_path)
    assert result['elapsed'] == dict(seconds=22., source='log.csv:sum(seconds)', scope='completed_epochs_only')


def test_completion_receipt_is_observed_not_training_authorization(progress, tmp_path):
    history(tmp_path / 'log.csv', [epoch(1, .95, .98), epoch(2, .95, .98)])
    (tmp_path / 'train.json').write_text(json.dumps(dict(passed=True, role='val', epochs_completed=2, selected_epoch=1)))
    (tmp_path / 'control.json').write_text(json.dumps(dict(schema_version='direct_reference_control_result_v1', passed=True, epochs_completed=2)))
    predictions(tmp_path / 'val_predictions.csv')
    result = progress.summarize_progress(tmp_path, total_epochs=2)
    assert result['status'] == 'completion_recorded'
    assert result['selected_validation']['provisional'] is False
    assert result['validation_predictions']['selected_epoch'] == 1
    assert result['authorization'] == 'none; observation only'


@pytest.mark.parametrize('receipt', ['{"passed":', '{"passed":NaN}', '{"passed":true,"passed":false}'])
def test_partial_or_invalid_receipt_is_a_warning_not_a_crash(progress, tmp_path, receipt):
    history(tmp_path / 'log.csv', [epoch(1)])
    (tmp_path / 'train.json').write_text(receipt)
    result = progress.summarize_progress(tmp_path)
    assert result['status'] == 'in_progress' and result['issues']


def test_complete_epoch_count_without_completion_does_not_imply_execution_pass(progress, tmp_path):
    history(tmp_path / 'log.csv', [epoch(1), epoch(2)])
    assert progress.summarize_progress(tmp_path, total_epochs=2)['status'] == 'in_progress'


def test_selection_receipt_disagreement_is_visible(progress, tmp_path):
    history(tmp_path / 'log.csv', [epoch(1), epoch(2)])
    (tmp_path / 'train.json').write_text(json.dumps(dict(passed=True, role='val', epochs_completed=2, selected_epoch=2)))
    result = progress.summarize_progress(tmp_path, total_epochs=2)
    assert result['issues'] and result['selected_validation']['epoch'] == 1


def test_reader_never_opens_test_pixels_checkpoints_or_unknown_receipt_paths(progress, tmp_path, monkeypatch):
    history(tmp_path / 'log.csv', [epoch(1)])
    predictions(tmp_path / 'val_predictions.csv')
    forbidden = ['best.pt', 'test_predictions.csv', 'query.png', 'unknown.json']
    for name in forbidden:
        (tmp_path / name).write_bytes(b'must not open')
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    original = Path.read_bytes

    def guarded(path):
        assert path.name not in forbidden
        return original(path)

    monkeypatch.setattr(Path, 'read_bytes', guarded)
    progress.summarize_progress(tmp_path)
    assert all(original(path) == value for path, value in before.items())


def test_symlink_metadata_is_ignored_with_warning(progress, tmp_path):
    target = tmp_path / 'outside.csv'
    history(target, [epoch(1)])
    (tmp_path / 'log.csv').symlink_to(target)
    result = progress.summarize_progress(tmp_path)
    assert result['epochs_completed'] == 0 and result['issues']


def test_cli_json_and_text_are_stdlib_only_and_separate_off_on(progress, tmp_path):
    for arm in ('off', 'on'):
        directory = tmp_path / 'training' / arm
        directory.mkdir(parents=True)
        history(directory / 'log.csv', [epoch(1)])
    command = [sys.executable, '-S', str(SCRIPT), str(tmp_path)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    document = json.loads(result.stdout)
    assert document['schema_version'] == 'direct_reference_progress_v1'
    assert [report['arm'] for report in document['runs']] == ['off', 'on']
    text = subprocess.run([*command, '--format', 'text'], text=True, capture_output=True, timeout=10)
    assert text.returncode == 0 and '1/100' in text.stdout and 'validation' in text.stdout


def test_unclosed_csv_tail_retains_preceding_epochs(progress, tmp_path):
    history(tmp_path / 'log.csv', [epoch(1), epoch(2)])
    with (tmp_path / 'log.csv').open('a') as stream:
        stream.write('3,.2,"unfinished')
    result = progress.summarize_progress(tmp_path)
    assert result['epochs_completed'] == 2 and result['issues']


@pytest.mark.parametrize('mutation', [dict(passed=False), dict(epochs_completed=1)])
def test_root_flag_does_not_override_incomplete_training_receipt(progress, tmp_path, mutation):
    history(tmp_path / 'log.csv', [epoch(1), epoch(2)])
    train = dict(passed=True, role='val', epochs_completed=2, selected_epoch=1)
    (tmp_path / 'train.json').write_text(json.dumps(dict(train, **mutation)))
    (tmp_path / 'control.json').write_text(json.dumps(dict(schema_version='direct_reference_control_result_v1', passed=True, epochs_completed=2)))
    result = progress.summarize_progress(tmp_path, total_epochs=2)
    assert result['status'] == 'in_progress'
