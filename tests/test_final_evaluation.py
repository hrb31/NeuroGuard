"""Synthetic integrity, metric, pairing and explicit final-evaluation gate tests."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip('torch')
from torch.utils.data import Dataset

from src.data import digest, write_json
from src.dataset import file_sha256
import src.final_evaluation as final
from src.metrics import aggregate, metric_row
from src.model import UNet


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def saved_run(tmp_path):
    """Generated weights and synthetic metadata; never real MRI evidence."""
    config = final.read_json('configs/bce.json')
    protocol = final.read_json('configs/final_evaluation.json')
    env = {'python': 'synthetic', 'torch': 'synthetic', 'cuda': None, 'cudnn': None,
           'gpu': None, 'deterministic_algorithms': True, 'cublas_workspace_config': ':4096:8',
           'packages': 'numpy==1\npillow==1\ntifffile==1\nimagecodecs==1'}
    manifest = {'manifest_sha256': protocol['manifest_sha256'], 'dataset_fingerprint': 'synthetic',
                'size_cutoffs': [.01, .1]}
    identity = {'config': config, 'commit': protocol['training_commits']['bce'],
                **manifest, 'parameter_count': 1942577, 'environment': env}
    write_json(tmp_path/'identity.json', identity)
    write_json(tmp_path/'config.json', config)
    report = {'slices': 621, 'positive_patient_macro': {'slices': 209}, 'empty': {'slices': 412}}
    write_json(tmp_path/'validation_reports.json', {'default_threshold': .5,
               'selection': {'threshold': .8, 'status': 'constraint_met'}, 'default': report, 'selected': report})
    history = [{'epoch': i, 'validation': {'positive_patient_macro': {'dice': .8 if i == 18 else .7}}}
               for i in range(1, 26)]
    write_json(tmp_path/'history.json', history)
    write_json(tmp_path/'review_summary.json', {'completed_epochs': 25, 'best_epoch': 18})
    state = {'model': UNet().state_dict(), 'identity': identity, 'validation_dice': .8}
    torch.save(state, tmp_path/'best.pt')
    write_json(tmp_path/'best_checksum.json', {'sha256': file_sha256(tmp_path/'best.pt')})
    return tmp_path, config, manifest, protocol, env, state


def load_fixture(fixture):
    run, config, manifest, protocol, env, _ = fixture
    return final.load_frozen_model(run, 'bce', config, manifest, protocol, env, torch.device('cpu'))


def test_original_checkpoint_reload_is_exact_and_inference_only(saved_run):
    model, evidence = load_fixture(saved_run)
    reference = UNet().eval()
    reference.load_state_dict(saved_run[-1]['model'])
    x = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert evidence['best_epoch'] == 18


def test_tversky_original_best_epoch_and_constraint_failure_fallback(saved_run):
    run, _, manifest, protocol, env, state = saved_run
    config = final.read_json('configs/tversky.json')
    identity = dict(state['identity'], config=config, commit=protocol['training_commits']['tversky'])
    write_json(run/'identity.json', identity)
    write_json(run/'config.json', config)
    validation = final.read_json(run/'validation_reports.json')
    validation['selection'] = {'threshold': .5, 'status': 'fp_constraint_unmet_fallback'}
    write_json(run/'validation_reports.json', validation)
    history = [{'epoch': i, 'validation': {'positive_patient_macro': {'dice': .8 if i == 23 else .7}}}
               for i in range(1, 31)]
    write_json(run/'history.json', history)
    write_json(run/'review_summary.json', {'completed_epochs': 30, 'best_epoch': 23})
    torch.save(dict(state, identity=identity), run/'best.pt')
    write_json(run/'best_checksum.json', {'sha256': file_sha256(run/'best.pt')})
    _, evidence = final.load_frozen_model(run, 'tversky', config, manifest, protocol, env, 'cpu')
    assert evidence['best_epoch'] == 23 and evidence['completed_epochs'] == 30
    assert evidence['selection'] == {'threshold': .5, 'status': 'fp_constraint_unmet_fallback'}


@pytest.mark.parametrize('field,value', [('commit', 'wrong'), ('parameter_count', 1),
                                      ('manifest_sha256', 'wrong'), ('size_cutoffs', [.2, .4]),
                                      ('dataset_fingerprint', 'wrong')])
def test_checkpoint_identity_mismatch_stops_before_deserialization(saved_run, monkeypatch, field, value):
    run = saved_run[0]
    identity = final.read_json(run/'identity.json')
    identity[field] = value
    write_json(run/'identity.json', identity)
    def forbidden(*args, **kwargs):
        pytest.fail('Checkpoint should not be deserialized after metadata mismatch')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='identity mismatch'):
        load_fixture(saved_run)


@pytest.mark.parametrize('kind', ['checksum', 'config', 'threshold', 'status', 'epoch', 'runtime', 'dependency'])
def test_checkpoint_preflight_rejects_tampering(saved_run, kind):
    run = saved_run[0]
    if kind == 'checksum':
        with (run/'best.pt').open('ab') as stream:
            stream.write(b'changed')
    elif kind == 'config':
        write_json(run/'config.json', dict(saved_run[1], batch_size=8))
    elif kind in ('threshold', 'status'):
        reports = final.read_json(run/'validation_reports.json')
        reports['selection'][kind] = .7 if kind == 'threshold' else 'fp_constraint_unmet_fallback'
        write_json(run/'validation_reports.json', reports)
    elif kind == 'epoch':
        write_json(run/'review_summary.json', {'completed_epochs': 25, 'best_epoch': 19})
    elif kind == 'runtime':
        saved_run[4]['torch'] = 'different'
    else:
        saved_run[4]['packages'] = saved_run[4]['packages'].replace('pillow==1', 'pillow==2')
    with pytest.raises(ValueError):
        load_fixture(saved_run)


@pytest.mark.parametrize('kind', ['embedded_identity', 'wrong_architecture', 'nonfinite'])
def test_checksum_alone_does_not_establish_checkpoint_correctness(saved_run, kind):
    state, run = copy.deepcopy(saved_run[-1]), saved_run[0]
    if kind == 'embedded_identity':
        state['identity']['commit'] = 'wrong'
    elif kind == 'wrong_architecture':
        state['model'].pop(next(iter(state['model'])))
    else:
        next(t for t in state['model'].values() if t.is_floating_point()).fill_(float('nan'))
    torch.save(state, run/'best.pt')
    write_json(run/'best_checksum.json', {'sha256': file_sha256(run/'best.pt')})
    with pytest.raises((ValueError, RuntimeError)):
        load_fixture(saved_run)


class ToyDataset(Dataset):
    split = 'test'
    epoch = 0
    def __len__(self):
        return 3
    def __getitem__(self, i):
        return {'image': torch.tensor([[[2., 1.]]]).repeat(3, 1, 1),
                'mask': torch.tensor([[[1., 0.]]]) if i < 2 else torch.zeros(1, 1, 2),
                'patient': 'A' if i != 1 else 'B', 'area_fraction': .5 if i < 2 else 0., 'index': i}


class ToyModel(torch.nn.Module):
    def forward(self, x):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        return x[:, :1]


def synthetic_rows():
    models = {a: ToyModel().eval() for a in ('bce', 'tversky')}
    protocol = final.read_json('configs/final_evaluation.json')
    config = {'seed': 42, 'batch_size': 2, 'num_workers': 0}
    return final.infer_rows(models, ToyDataset(), config, [.2, .7], protocol['operating_points'], 'cpu')


def test_single_pass_metrics_and_no_threshold_search():
    rows = synthetic_rows()
    reports = {k: aggregate(v) for k, v in rows.items()}
    assert len(reports) == 5
    assert reports['always_empty']['positive_patient_macro']['dice'] == 0
    assert reports['always_empty']['positive_patient_macro']['precision'] == 0
    assert reports['always_empty']['empty']['false_positive_rate'] == 0
    assert reports['bce_default']['positive_patient_macro']['dice'] == pytest.approx(2/3)
    assert reports['bce_selected']['positive_patient_macro']['dice'] == 1
    assert reports['bce_default']['empty']['false_positive_rate'] == 1
    assert reports['bce_default']['empty']['mean_foreground_fraction'] == 1
    assert reports['bce_selected']['empty']['mean_foreground_fraction'] == .5
    assert rows['tversky_selected'] == rows['tversky_default']
    assert reports['bce_default']['positive_patient_macro']['patients'] == 2
    with pytest.raises(ValueError, match='eval mode'):
        final.infer_rows({'bce': ToyModel()}, ToyDataset(), {'seed': 42}, [.2, .7], {}, 'cpu')


def test_paired_patient_bootstrap_preserves_duplicate_patient_weights_and_na():
    left, right = [], []
    # A has one slice and +1 improvement; B has nine slices and zero improvement.
    # Patient mean difference=.5, not the slice-weighted .1.
    for i in range(10):
        patient = 'A' if i == 0 else 'B'
        a = metric_row(np.array([0]), np.array([1]), patient, .1, [.2, .7])
        b = metric_row(np.array([i == 0]), np.array([1]), patient, .1, [.2, .7])
        left.append(dict(a, index=i))
        right.append(dict(b, index=i))
    settings = final.read_json('configs/final_evaluation.json')['bootstrap']
    result = final.paired_bootstrap(left, right, settings)
    assert result == final.paired_bootstrap(left, right, settings)
    assert result['metrics']['positive_dice']['difference'] == .5
    assert result['metrics']['positive_dice']['ci'] == [0., 1.]
    assert result['metrics']['medium_recall']['undefined_replicates'] == 1000
    with pytest.raises(ValueError, match='aligned'):
        final.paired_bootstrap(left, list(reversed(right)), settings)
    # Some cohort patients have only empty slices: undefined draws are counted.
    left[1:] = [dict(r, positive=False, size_group='empty') for r in left[1:]]
    right[1:] = [dict(r, positive=False, size_group='empty') for r in right[1:]]
    result = final.paired_bootstrap(left, right, settings)
    assert 0 < result['metrics']['positive_dice']['undefined_replicates'] < 1000


def test_comparison_reports_partial_support_and_distinguishes_units():
    reports = {k: aggregate(v) for k, v in synthetic_rows().items()}
    # Construct clear synthetic recall gain with unacceptable FPs.
    a = reports['bce_default']
    b = copy.deepcopy(a)
    a['positive_patient_macro']['recall'] = .5
    a['size_groups']['small']['recall'] = .5
    b['size_groups']['small']['recall'] = .6
    a['empty']['false_positive_rate'] = .05
    b['empty']['false_positive_rate'] = .7
    c = final.comparison({'a': a, 'b': b}, 'a', 'b', final.read_json('configs/final_evaluation.json'))
    assert c['metrics']['recall']['difference_pp'] == 50
    assert c['metrics']['recall']['relative_change_percent'] == 100
    assert c['research_fp_guardrail_met'] is False
    assert c['hypothesis_assessment'].startswith('partially supported')


def test_gate_and_prior_results_block_before_preflight_or_prediction(tmp_path, monkeypatch):
    protocol = final.read_json('configs/final_evaluation.json')
    ticket = {'protocol': protocol}
    def forbidden(*args, **kwargs):
        pytest.fail('Gate must stop before preflight/model/test access')
    original_preflight = final.preflight
    monkeypatch.setattr(final, 'preflight', forbidden)
    for phrase, identity in [('', digest(ticket)), (protocol['approval_phrase'], 'wrong')]:
        with pytest.raises(ValueError, match='Explicit approval'):
            final.run_final_evaluation(ticket, approval=phrase, reviewed_preflight_sha256=identity, device='cpu')
    final.assert_unused(tmp_path)
    (tmp_path/'old-run').mkdir()
    (tmp_path/'old-run'/'final_test_metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='already started'):
        final.assert_unused(tmp_path)
    with pytest.raises(ValueError, match='trusted'):
        original_preflight('.', '', '', {}, tmp_path, {}, 'cpu')


@pytest.mark.parametrize('fail_inference', [False, True])
def test_gated_synthetic_workflow_persists_once_and_changed_ticket_blocks(tmp_path, monkeypatch, fail_inference):
    protocol = final.read_json('configs/final_evaluation.json')
    manifest_path = tmp_path/'synthetic_manifest.json'
    manifest_path.write_text('{}')
    records = [{'foreground': 1 if i < 2 else 0, 'area_fraction': .5 if i < 2 else 0.,
                'patient': 'A' if i != 1 else 'B', 'slice': i, 'image': f'synthetic_{i}'}
               for i in range(3)]
    models = {a: ToyModel().eval() for a in ('bce', 'tversky')}
    ticket = {'protocol': protocol, 'checkpoints': {a: {
        'run': str(tmp_path/a), 'hashes': {}, 'identity': {'config': {
            'seed': 42, 'batch_size': 2, 'num_workers': 0, 'preprocessing': {}}}}
        for a in models}, 'manifest_path': str(manifest_path), 'manifest_sha256': 'synthetic',
        'manifest_file_sha256': file_sha256(manifest_path), 'data_root': str(tmp_path),
        'results_root': str(tmp_path/'phase5'), 'size_cutoffs': [.2, .7],
        'test_cohort': {'slices': 3, 'patients': 2, 'positive_slices': 2, 'empty_slices': 1,
                        'patients_with_positive': 2, 'patients_with_empty': 1},
        'environment': {}, 'source': {'commit': 'synthetic'}}
    def preflight(*args, **kwargs):
        final.assert_unused(ticket['results_root'])
        return copy.deepcopy(ticket), models
    monkeypatch.setattr(final, 'preflight', preflight)
    monkeypatch.setattr(final, 'current_environment', lambda: {})
    monkeypatch.setattr(final, 'load_manifest', lambda *args: {'records': records})
    data = ToyDataset()
    data.records = records
    monkeypatch.setattr(final, 'FinalTestDataset', lambda *args: data)
    altered = copy.deepcopy(ticket)
    altered['size_cutoffs'] = [.3, .6]
    with pytest.raises(ValueError, match='changed since review'):
        final.run_final_evaluation(altered, approval=protocol['approval_phrase'],
                                   reviewed_preflight_sha256=digest(altered), device='cpu')
    assert not Path(ticket['results_root']).exists()
    # The production gate/persistence/reporting is exercised with synthetic inference.
    # No real preflight/model identity verification is bypassed in the notebook.
    for arm in ticket['checkpoints'].values():
        arm['hashes']['best.pt'] = 'synthetic'
    monkeypatch.setattr(final, 'file_sha256', lambda p: 'synthetic' if Path(p).name == 'best.pt' else file_sha256(p))
    if fail_inference:
        def fail(*args, **kwargs):
            raise RuntimeError('Injected synthetic inference failure')
        monkeypatch.setattr(final, 'infer_rows', fail)
        with pytest.raises(RuntimeError, match='Injected synthetic'):
            final.run_final_evaluation(ticket, approval=protocol['approval_phrase'],
                                       reviewed_preflight_sha256=digest(ticket), device='cpu')
        output = Path(ticket['results_root'])/'final-test-seed42-01'
        assert (output/'evaluation_failure.json').is_file()
        assert not (output/'review_summary.json').exists()
        with pytest.raises(ValueError, match='already started'):
            final.assert_unused(ticket['results_root'])
        return
    output, summary = final.run_final_evaluation(ticket, approval=protocol['approval_phrase'],
                                                reviewed_preflight_sha256=digest(ticket), device='cpu')
    assert summary['test_evaluated'] is True and summary['cohort']['slices'] == 3
    assert (output/'review_summary.json').is_file()
    assert (output/'figures_PRIVATE'/'example_00.png').is_file()
    with pytest.raises(ValueError, match='already started'):
        final.run_final_evaluation(ticket, approval=protocol['approval_phrase'],
                                   reviewed_preflight_sha256=digest(ticket), device='cpu')


def test_data_changed_at_use_is_rejected_before_decode(tmp_path, monkeypatch):
    import tifffile
    image = np.ones((16, 16, 3), np.uint8)
    mask = np.zeros((16, 16), np.uint8)
    tifffile.imwrite(tmp_path/'image.tif', image, photometric='rgb')
    tifffile.imwrite(tmp_path/'mask.tif', mask)
    row = {'patient': 'synthetic', 'slice': 1, 'split': 'test', 'image': 'image.tif', 'mask': 'mask.tif',
           'shape': [16, 16, 3], 'foreground': 0, 'area_fraction': 0.,
           'image_sha256': file_sha256(tmp_path/'image.tif'), 'mask_sha256': file_sha256(tmp_path/'mask.tif')}
    data = final.FinalTestDataset(tmp_path, {'records': [row]}, {'size': 16, 'channels': [0, 1, 2]})
    assert not data[0]['mask'].any()
    with (tmp_path/'mask.tif').open('ab') as stream:
        stream.write(b'changed')
    monkeypatch.setattr(final, 'read_pair', lambda *args: pytest.fail('Must reject before decoding'))
    with pytest.raises(ValueError, match='changed after preflight'):
        data[0]


def test_example_policy_ties_and_failures_and_reporting(tmp_path):
    rows = synthetic_rows()
    records = [{'foreground': 1 if i < 2 else 0, 'area_fraction': .5 if i < 2 else 0.}
               for i in range(3)]
    chosen = final.example_indices(records, rows, [.2, .7])
    assert set(chosen) == {0, 2}  # lower median and stable tie-breaking
    assert any('lowest positive Dice' in reason for reason in chosen[0])
    reports = {k: aggregate(v) for k, v in rows.items()}
    c = final.comparison(reports, 'bce_default', 'tversky_default', final.read_json('configs/final_evaluation.json'))
    final.save_tables(reports, {'default': c}, tmp_path,
                      final.read_json('configs/final_evaluation.json')['operating_points'])
    assert (tmp_path/'aggregate_comparison.png').stat().st_size > 1000
    assert 'empty_false_positive_rate' in (tmp_path/'empty_and_pooled_metrics.csv').read_text()
    assert 'fp_constraint_unmet_fallback' in (tmp_path/'positive_size_metrics.csv').read_text()


def test_final_notebook_clean_and_compiles():
    notebook = json.loads(Path('notebooks/04_final_evaluation.ipynb').read_text(encoding='utf-8'))
    code = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is None and not cell['outputs']
            source = ''.join(cell['source'])
            compile(source, 'final_notebook_cell', 'exec')
            code.append(source)
    joined = '\n'.join(code)
    assert 'run_final_evaluation' in joined and 'PREFLIGHT_REVIEWED = False' in joined
    assert 'fit(' not in joined and 'select_threshold(' not in joined and 'freeze_split(' not in joined
