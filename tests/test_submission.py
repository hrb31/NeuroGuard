"""Guard the saved final readout against metric drift and private asset leakage."""
import json
from pathlib import Path


def test_saved_readout_provenance_points_and_exact_reports():
    summary = json.loads(Path('results/final-summary.json').read_text(encoding='utf-8'))
    protocol = json.loads(Path('configs/final_evaluation.json').read_text(encoding='utf-8'))
    assert summary['operating_points'] == protocol['operating_points']
    assert summary['provenance']['manifest_sha256'] == protocol['manifest_sha256']
    assert summary['metrics']['tversky_default'] == summary['metrics']['tversky_selected']
    for report in summary['metrics'].values():
        assert report['positive_patient_macro']['patients'] == 17
        assert report['positive_patient_macro']['slices'] == 230
        assert report['empty']['slices'] == 359
        assert sum(group['slices'] for group in report['size_groups'].values()) == 230
    assert abs(summary['metrics']['bce_default']['positive_patient_macro']['dice'] - .76726368) < 5e-9
    assert abs(summary['metrics']['tversky_default']['empty']['false_positive_rate'] - .85793872) < 5e-9
    # The public readout contains aggregate measures, no per-patient record list.
    assert 'records' not in summary and 'patients' not in summary['metrics']


def test_final_submission_notebook_has_real_readout_outputs_only():
    nb = json.loads(Path('notebooks/NeuroGuard_Final_Submission.ipynb').read_text(encoding='utf-8'))
    count, images = 0, 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        compile(source, 'submission_readout_cell', 'exec')
        assert not any(term in source for term in ('torch', 'Segmenter', 'run_final_evaluation',
                       'fit(', 'freeze_split', 'select_threshold', 'local-assets', 'drive.mount'))
        assert cell['execution_count'] is not None and cell['outputs']
        count += 1
        for output in cell['outputs']:
            assert output['output_type'] != 'error'
            images += 'image/png' in output.get('data', {})
    assert count >= 5 and images == 1


def test_original_notebooks_and_sources_unchanged_from_completed_evaluation():
    import subprocess
    frozen = ['src/model.py', 'src/preprocessing.py', 'src/metrics.py',
              'src/training.py', 'src/losses.py', 'src/final_evaluation.py',
              'configs/bce.json', 'configs/tversky.json', 'configs/final_evaluation.json',
              'docs/evaluation-protocol.md']
    frozen += [p.as_posix() for p in Path('notebooks').glob('0[1-4]_*.ipynb')]
    for name in frozen:
        original = subprocess.check_output(['git','show',f'd26811abb564680075030d7970633b4e681a98d4:{name}'])
        assert Path(name).read_bytes().replace(b'\r\n', b'\n') == original.replace(b'\r\n', b'\n'), name
