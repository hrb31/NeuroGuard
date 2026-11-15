"""Locked, read-only model preflight and explicitly gated private test evaluation.

Development loaders/training are deliberately unchanged. Only this module opens
test pixels, after the review gate. No fitting or threshold search is implemented.
"""
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import digest, read_pair, summarize, write_json
from .dataset import file_sha256, load_manifest, make_loader, safe_path
from .metrics import aggregate, metric_row
from .model import UNet
from .preprocessing import preprocess


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def current_environment():
    return {'python': sys.version, 'torch': str(torch.__version__),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'packages': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def assert_unused(directory):
    """A started/failed evaluation also blocks an automatic second evaluation."""
    directory = Path(directory)
    for name in ('EVALUATION_STARTED.json', 'final_test_metrics.json', 'review_summary.json'):
        if next(directory.rglob(name), None) is not None:
            raise ValueError('Test evaluation already started or results exist; preserve and review them')


def verify_source(repo, protocol):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    if git('status', '--porcelain'):
        raise ValueError('Evaluation source checkout is not clean')
    # Compare actual implementation, not just architecture labels/parameter count.
    immutable = ('src/model.py', 'src/preprocessing.py', 'src/metrics.py',
                 'src/data.py', 'src/dataset.py')
    for arm, commit in protocol['training_commits'].items():
        for relative in (*immutable, f'configs/{arm}.json'):
            original = git('show', f'{commit}:{relative}')
            if (Path(repo)/relative).read_text(encoding='utf-8').strip() != original:
                raise ValueError(f'Frozen implementation differs: {arm} {relative}')
    paths = (*immutable, 'src/final_evaluation.py', 'configs/final_evaluation.json',
             'docs/evaluation-protocol.md', 'docs/phase5-evaluation.md')
    return {'commit': git('rev-parse', 'HEAD'),
            'files': {p: file_sha256(Path(repo)/p) for p in paths}}


def verify_runtime(original, current):
    core = ('python', 'torch', 'cuda', 'cudnn', 'gpu',
            'deterministic_algorithms', 'cublas_workspace_config')
    for key in core:
        if key not in original or original[key] != current.get(key):
            raise ValueError(f'Frozen inference runtime differs: {key}; stop for review')
    def versions(text):
        return {name.lower(): version for line in text.splitlines() if '==' in line
                for name, version in [line.split('==', 1)]}
    old, new = versions(original['packages']), versions(current['packages'])
    for package in ('numpy', 'pillow', 'tifffile', 'imagecodecs'):
        if not old.get(package) or old[package] != new.get(package):
            raise ValueError(f'Frozen preprocessing dependency differs: {package}')
    a, b = set(original['packages'].splitlines()), set(current['packages'].splitlines())
    return {'removed_or_changed': sorted(a-b), 'added_or_changed': sorted(b-a)}


def load_frozen_model(run, arm, config, manifest, protocol, environment, device):
    """Verify all metadata/checksum before safe deserialization of trusted best.pt.

    A colocated checksum detects corruption, not authenticity. The user must
    confirm these are their own original trusted private training directories.
    """
    run = Path(run)
    names = ('identity.json', 'config.json', 'best.pt', 'best_checksum.json',
             'history.json', 'validation_reports.json', 'review_summary.json')
    hashes = {name: file_sha256(run/name) for name in names}
    identity = read_json(run/'identity.json')
    if identity['config'] != config or read_json(run/'config.json') != config:
        raise ValueError(f'{arm}: checkpoint configuration mismatch')
    for key, expected in {'commit': protocol['training_commits'][arm],
                          'manifest_sha256': manifest['manifest_sha256'],
                          'dataset_fingerprint': manifest['dataset_fingerprint'],
                          'size_cutoffs': manifest['size_cutoffs'],
                          'parameter_count': 1942577}.items():
        if identity.get(key) != expected:
            raise ValueError(f'{arm}: checkpoint identity mismatch: {key}')
    packages = verify_runtime(identity['environment'], environment)
    if hashes['best.pt'] != read_json(run/'best_checksum.json')['sha256']:
        raise ValueError(f'{arm}: best checkpoint checksum mismatch')
    validation = read_json(run/'validation_reports.json')
    expected = protocol['operating_points'][arm+'_selected']
    if validation['selection'] != {'threshold': expected['threshold'], 'status': expected['status']}:
        raise ValueError(f'{arm}: original validation-selected threshold/status mismatch')
    if validation['default_threshold'] != .5:
        raise ValueError(f'{arm}: common threshold mismatch')
    for point in ('default', 'selected'):
        report = validation[point]
        if (report['slices'], report['positive_patient_macro']['slices'], report['empty']['slices']) != (621, 209, 412):
            raise ValueError(f'{arm}: validation denominator mismatch')
    history = read_json(run/'history.json')
    if not history or [r['epoch'] for r in history] != list(range(1, len(history)+1)) or len(history) > 30:
        raise ValueError(f'{arm}: invalid completed-epoch history')
    scores = [r['validation']['positive_patient_macro']['dice'] for r in history]
    if not all(s is not None and np.isfinite(s) and 0 <= s <= 1 for s in scores):
        raise ValueError(f'{arm}: invalid validation selection history')
    best_epoch = int(np.argmax(scores))+1  # earliest on ties, frozen rule
    summary = read_json(run/'review_summary.json')
    if summary['best_epoch'] != best_epoch or summary['completed_epochs'] != len(history):
        raise ValueError(f'{arm}: best epoch disagrees with saved review/history')
    if arm == 'bce' and (len(history), best_epoch) != (25, 18):
        raise ValueError('BCE run differs from user-verified original run')
    if arm == 'tversky' and (len(history), best_epoch) != (30, 23):
        raise ValueError('Tversky run differs from user-verified original run')
    state = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
    if state['identity'] != identity or state['validation_dice'] != scores[best_epoch-1]:
        raise ValueError(f'{arm}: embedded checkpoint provenance/selection mismatch')
    model = UNet(**config['model'])
    if sum(p.numel() for p in model.parameters()) != identity['parameter_count']:
        raise ValueError(f'{arm}: architecture parameter count mismatch')
    model.load_state_dict(state['model'], strict=True)
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError(f'{arm}: nonfinite saved weights')
    model.to(device).eval().requires_grad_(False)
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 256, 256, device=device))
        if logits.shape != (1, 1, 256, 256) or not torch.isfinite(logits).all():
            raise ValueError(f'{arm}: synthetic inference check failed')
    return model, {'run': str(run.resolve()), 'identity': identity, 'hashes': hashes,
                   'best_epoch': best_epoch, 'completed_epochs': len(history),
                   'selection': validation['selection'], 'package_differences': packages}


def verify_test_bytes(root, manifest):
    """One hash-only pass over test files; no repeat of the full Phase 2 audit."""
    records = [r for r in manifest['records'] if r['split'] == 'test']
    for row in records:
        if row['shape'] != [256, 256, 3]:
            raise ValueError('Unexpected frozen test dimensions')
        for kind in ('image', 'mask'):
            if file_sha256(safe_path(root, row[kind])) != row[kind+'_sha256']:
                raise ValueError('Original test file bytes differ from frozen manifest')
    return summarize(records)


def preflight(repo, manifest_path, root, runs, results_root, environment, device,
              *, trusted_checkpoints=False):
    if not trusted_checkpoints:
        raise ValueError('Confirm these are your own original trusted checkpoints')
    protocol = read_json(Path(repo)/'configs/final_evaluation.json')
    assert_unused(results_root)
    source = verify_source(repo, protocol)
    manifest = load_manifest(manifest_path, protocol['manifest_sha256'])
    for split, expected in protocol['cohorts'].items():
        actual = manifest['summary'][split]
        if [actual[k] for k in ('patients', 'slices', 'positive_slices')] != expected:
            raise ValueError('Original frozen cohort counts differ')
    cohort = verify_test_bytes(root, manifest)
    models, evidence = {}, {}
    configs = {a: read_json(Path(repo)/f'configs/{a}.json') for a in ('bce', 'tversky')}
    if {k for k in configs['bce'] if configs['bce'][k] != configs['tversky'][k]} != {'experiment', 'loss'}:
        raise ValueError('Original arms differ beyond experiment/loss')
    for arm in configs:
        models[arm], evidence[arm] = load_frozen_model(
            runs[arm], arm, configs[arm], manifest, protocol, environment, device)
        if evidence[arm]['identity']['manifest_file_sha256'] != file_sha256(manifest_path):
            raise ValueError('Original manifest file bytes differ from checkpoint identity')
    tv = evidence['tversky']['identity']
    if (tv['bce_reference_commit'] != evidence['bce']['identity']['commit'] or
        tv['bce_identity_file_sha256'] != evidence['bce']['hashes']['identity.json']):
        raise ValueError('Tversky original BCE reference identity mismatch')
    ticket = {'stage': 'preflight only', 'test_evaluated': False, 'source': source,
              'protocol': protocol, 'protocol_sha256': digest(protocol),
              'manifest_path': str(Path(manifest_path).resolve()),
              'manifest_file_sha256': file_sha256(manifest_path),
              'manifest_sha256': manifest['manifest_sha256'],
              'dataset_fingerprint': manifest['dataset_fingerprint'],
              'size_cutoffs': manifest['size_cutoffs'], 'test_cohort': cohort,
              'test_assignments_sha256': digest([r for r in manifest['records'] if r['split'] == 'test']),
              'data_root': str(Path(root).resolve()),
              'results_root': str(Path(results_root).resolve()),
              'checkpoints': evidence, 'environment': environment}
    return ticket, models


class FinalTestDataset(Dataset):
    """Constructed by the gated evaluator only; never changes MRIDataset rules."""
    split = 'test'

    def __init__(self, root, manifest, preprocessing):
        self.root = Path(root)
        self.records = sorted([r for r in manifest['records'] if r['split'] == 'test'],
                              key=lambda r: (r['patient'], r['slice'], r['image']))
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        # Recheck bytes at use, including after an approval pause/runtime restart.
        for kind in ('image', 'mask'):
            if file_sha256(safe_path(self.root, row[kind])) != row[kind+'_sha256']:
                raise ValueError('Test bytes changed after preflight')
        image, mask = read_pair(self.root, row)
        if list(image.shape) != row['shape'] or image.shape[:2] != mask.shape:
            raise ValueError('Test pair dimensions differ')
        if int(np.count_nonzero(mask)) != row['foreground']:
            raise ValueError('Test foreground differs from manifest')
        x, y = preprocess(image, mask, training=False, **self.preprocessing)
        return {'image': torch.from_numpy(x.copy()), 'mask': torch.from_numpy(y.copy()),
                'patient': row['patient'], 'area_fraction': row['area_fraction'], 'index': index}


def infer_rows(models, dataset, config, cutoffs, points, device):
    """Single pass; one forward per model/batch, reuse probabilities at fixed t."""
    for model in models.values():
        if any(module.training for module in model.modules()):
            raise ValueError('Models must already be in eval mode')
    rows = {name: [] for name in points}
    with torch.inference_mode():
        for batch in make_loader(dataset, config):
            x, y = batch['image'].to(device), batch['mask'].numpy()
            probabilities = {}
            for arm, model in models.items():
                logits = model(x)
                if logits.shape != batch['mask'].shape or not torch.isfinite(logits).all():
                    raise ValueError('Invalid final test logits')
                probabilities[arm] = logits.sigmoid().cpu().numpy()
            for name, point in points.items():
                pred = (probabilities[point['model']] >= point['threshold']
                        if point['model'] else np.zeros_like(y))
                for i in range(len(x)):
                    row = metric_row(pred[i], y[i], batch['patient'][i],
                                     float(batch['area_fraction'][i]), cutoffs)
                    row['index'] = int(batch['index'][i])
                    rows[name].append(row)
    return rows


def paired_bootstrap(left, right, settings):
    """Tversky minus BCE patient-macro differences; duplicate draws keep weight.

    Draw the entire test patient cohort, including patients without positives.
    Average eligible patient means per draw, not rows grouped under repeated IDs.
    """
    keys = lambda rows: [(r['patient'], r['index'], r['positive'], r['size_group']) for r in rows]
    if keys(left) != keys(right):
        raise ValueError('Paired patient bootstrap requires aligned slices')
    patients = sorted({r['patient'] for r in left})
    if not patients:
        raise ValueError('No patients to bootstrap')
    rng = np.random.default_rng(settings['seed'])
    draws = rng.integers(0, len(patients), size=(settings['replicates'], len(patients)))
    output = {}
    for group in ('positive', 'small', 'medium', 'large'):
        for metric in ('dice', 'recall'):
            values = []
            for patient in patients:
                def eligible(r):
                    return r['patient'] == patient and r['positive'] and (group == 'positive' or r['size_group'] == group)
                a, b = [r[metric] for r in left if eligible(r)], [r[metric] for r in right if eligible(r)]
                values.append(float(np.mean(b)-np.mean(a)) if a else np.nan)
            values = np.asarray(values)
            sampled = values[draws]
            counts = np.isfinite(sampled).sum(axis=1)
            estimates = np.divide(np.nansum(sampled, axis=1), counts,
                                  out=np.full(len(draws), np.nan), where=counts > 0)
            finite = estimates[np.isfinite(estimates)]
            tail = (1-settings['confidence'])/2
            interval = np.quantile(finite, [tail, 1-tail]).tolist() if len(finite) else [None, None]
            output[group+'_'+metric] = {
                'difference': float(np.nanmean(values)) if np.isfinite(values).any() else None,
                'ci': interval, 'defined_replicates': len(finite),
                'undefined_replicates': int(len(draws)-len(finite)),
                'eligible_patients': int(np.isfinite(values).sum())}
    return {'settings': settings, 'units': 'proportion difference (Tversky minus BCE)',
            'cohort_patients': len(patients), 'metrics': output,
            'limitation': 'Exploratory paired percentile intervals on one small cohort; no clinical or significance claim.'}


def comparison(reports, left, right, protocol):
    a, b = reports[left], reports[right]
    fields = {'dice': ('positive_patient_macro', 'dice'), 'recall': ('positive_patient_macro', 'recall'),
              'precision': ('positive_patient_macro', 'precision'), 'iou': ('positive_patient_macro', 'iou'),
              'empty_fp_rate': ('empty', 'false_positive_rate')}
    values = {}
    for name, path in fields.items():
        values[name] = (a[path[0]][path[1]], b[path[0]][path[1]])
    for group in ('small', 'medium', 'large'):
        for metric in ('dice', 'recall', 'precision', 'iou'):
            values[group+'_'+metric] = (a['size_groups'][group][metric], b['size_groups'][group][metric])
    deltas = {k: {'bce': x, 'tversky': y,
                 'difference_pp': 100*(y-x) if x is not None and y is not None else None,
                 'relative_change_percent': 100*(y-x)/x if x not in (None, 0) and y is not None else None}
              for k, (x, y) in values.items()}
    x, y = values['empty_fp_rate']
    guardrail = (y <= protocol['empty_fp_limit'] and y-x <= protocol['comparison_fp_increase_limit']
                 if x is not None and y is not None else None)
    recall_gains = all(deltas[k]['difference_pp'] is not None and deltas[k]['difference_pp'] > 0
                       for k in ('recall', 'small_recall'))
    conclusion = ('partially supported: recall and small-region recall improved, FP guardrail failed'
                  if recall_gains and guardrail is False else
                  'supported on this cohort only: both recall gains and research FP guardrail met'
                  if recall_gains and guardrail is True else
                  'not fully supported or unevaluable; inspect recall requirements and FP guardrail separately')
    return {'bce_point': left, 'tversky_point': right, 'metrics': deltas,
            'recall_and_small_recall_improved': recall_gains,
            'research_fp_guardrail_met': guardrail, 'hypothesis_assessment': conclusion}


def example_indices(records, rows, cutoffs):
    chosen = defaultdict(list)
    for group in ('small', 'medium', 'large', 'empty'):
        indices = [i for i, r in enumerate(records) if
                   ('empty' if not r['foreground'] else 'small' if r['area_fraction'] <= cutoffs[0]
                    else 'medium' if r['area_fraction'] <= cutoffs[1] else 'large') == group]
        if indices:
            indices.sort(key=lambda i: (records[i]['area_fraction'], i))
            chosen[indices[0] if group == 'empty' else indices[(len(indices)-1)//2]].append('prespecified '+group)
    for arm in ('bce_default', 'tversky_default'):
        positive = [r for r in rows[arm] if r['positive']]
        empty_fp = [r for r in rows[arm] if not r['positive'] and r['fp'] > 0]
        if positive:
            chosen[min(positive, key=lambda r: (r['dice'], r['index']))['index']].append(arm+' lowest positive Dice')
        if empty_fp:
            chosen[min(empty_fp, key=lambda r: (-r['fp'], r['index']))['index']].append(arm+' largest empty FP')
    return dict(sorted(chosen.items()))


def save_test_figures(models, dataset, rows, points, cutoffs, directory, device):
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir()
    selection = example_indices(dataset.records, rows, cutoffs)
    write_json(directory/'selection_private.json', {str(i): {'reasons': reasons, 'record': dataset.records[i]}
                                                   for i, reasons in selection.items()})
    with torch.inference_mode():
        for number, (index, reasons) in enumerate(selection.items()):
            sample = dataset[index]
            x, y = sample['image'], sample['mask'][0].numpy().astype(bool)
            probabilities = {a: m(x[None].to(device)).sigmoid()[0, 0].cpu().numpy() for a, m in models.items()}
            panels = [('Ground truth', y)] + [
                (f"{name} t={p['threshold']:g}\n{p['status']}", probabilities[p['model']] >= p['threshold'])
                for name, p in points.items() if p['model']]
            fig, axes = plt.subplots(1, len(panels), figsize=(18, 4))
            for ax, (title, mask) in zip(axes, panels):
                ax.imshow(x[1].numpy(), cmap='gray')
                ax.imshow(np.ma.masked_where(~mask, mask), cmap='autumn', alpha=.5, vmin=0, vmax=1)
                ax.set_title(title, fontsize=9)
                ax.axis('off')
            fig.suptitle('; '.join(reasons)+'\nPrivate test example, FLAIR channel 1; research prototype, not medical advice', fontsize=9)
            fig.tight_layout()
            fig.savefig(directory/f'example_{number:02d}.png', dpi=130)
            plt.close(fig)
    return {'examples': len(selection), 'failure_selection': 'Outcome-selected extremes labelled explicitly; no prevalence inference'}


def save_tables(reports, comparisons, directory, points):
    rows = []
    for name, report in reports.items():
        for group, values in [('positive', report['positive_patient_macro']), *report['size_groups'].items()]:
            rows.append({'point': name, **points[name], 'group': group, **values})
    with (Path(directory)/'positive_size_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    differences = [{'comparison': name, 'metric': metric,
                    'bce_threshold': points[result['bce_point']]['threshold'],
                    'tversky_threshold': points[result['tversky_point']]['threshold'],
                    'tversky_status': points[result['tversky_point']]['status'], **values}
                   for name, result in comparisons.items() for metric, values in result['metrics'].items()]
    with (Path(directory)/'differences.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(differences[0]))
        writer.writeheader()
        writer.writerows(differences)
    supplementary = []
    for name, report in reports.items():
        supplementary.append({'point': name, **points[name], 'slices': report['slices'],
                              **{'empty_'+k: v for k, v in report['empty'].items()},
                              **{'pooled_'+k: v for k, v in report['pooled_all_slices'].items()}})
    with (Path(directory)/'empty_and_pooled_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(supplementary[0]))
        writer.writeheader()
        writer.writerows(supplementary)
    import matplotlib.pyplot as plt
    names = list(reports)
    labels = []
    for name in names:
        point = points[name]
        suffix = ('fallback (FP unmet)' if point['status'] == 'fp_constraint_unmet_fallback'
                  else 'validation-selected' if point['status'] == 'constraint_met' else 'common threshold')
        labels.append(f"{point['model'].upper()} t={point['threshold']:g}\n{suffix}"
                      if point['model'] else 'Always empty')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (group, metric, title) in zip(axes, (
            ('positive_patient_macro', 'dice', 'Patient-macro positive Dice'),
            ('positive_patient_macro', 'recall', 'Patient-macro positive recall'),
            ('empty', 'false_positive_rate', 'Empty-mask FP slice rate'))):
        values = [reports[n][group][metric] for n in names]
        ax.bar(range(len(names)), [v if v is not None else np.nan for v in values])
        ax.set_xticks(range(len(names)), labels, rotation=35, ha='right', fontsize=8)
        ax.set(title=title, ylim=(0, 1))
    axes[2].axhline(.1, color='red', linestyle='--', label='Research 10% guardrail')
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(directory)/'aggregate_comparison.png', dpi=130)
    plt.close(fig)


def run_final_evaluation(ticket, *, approval, reviewed_preflight_sha256, device):
    """Revalidate the ticket/artifacts, reserve once, then open test predictions.

    Reload the verified best.pt after approval: mutable in-memory model weights
    cannot accidentally become the evaluated checkpoint. No automatic resume.
    """
    if (approval != ticket['protocol']['approval_phrase'] or
        reviewed_preflight_sha256 != digest(ticket)):
        raise ValueError('Explicit approval of this exact preflight is required')
    repo = Path.cwd()
    runs = {a: e['run'] for a, e in ticket['checkpoints'].items()}
    refreshed, models = preflight(repo, ticket['manifest_path'], ticket['data_root'], runs,
                                  ticket['results_root'], current_environment(), device,
                                  trusted_checkpoints=True)
    if refreshed != ticket:
        raise ValueError('Preflight changed since review; stop without test predictions')
    results_root = Path(ticket['results_root'])
    # Canonical claim across ALL run IDs. Never remove after failure automatically.
    results_root.mkdir(parents=True, exist_ok=True)
    marker = results_root/'EVALUATION_STARTED.json'
    with marker.open('x', encoding='utf-8') as stream:
        json.dump({'preflight_sha256': digest(ticket), 'started_utc': datetime.now(timezone.utc).isoformat(),
                   'status': 'started; may be incomplete, do not rerun automatically'}, stream, indent=2)
    output = results_root/'final-test-seed42-01'
    output.mkdir(exist_ok=False)
    write_json(output/'evaluation_provenance.json', ticket)
    (output/'metric_definitions.md').write_text(
        (repo/'docs/evaluation-protocol.md').read_text(encoding='utf-8'), encoding='utf-8')
    protocol = ticket['protocol']
    try:
        manifest = load_manifest(ticket['manifest_path'], ticket['manifest_sha256'])
        config = ticket['checkpoints']['bce']['identity']['config']
        dataset = FinalTestDataset(ticket['data_root'], manifest, config['preprocessing'])
        rows = infer_rows(models, dataset, config, ticket['size_cutoffs'], protocol['operating_points'], device)
        reports = {name: aggregate(values) for name, values in rows.items()}
        for report in reports.values():
            if (report['slices'] != ticket['test_cohort']['slices'] or
                report['positive_patient_macro']['slices'] != ticket['test_cohort']['positive_slices'] or
                report['empty']['slices'] != ticket['test_cohort']['empty_slices'] or
                report['positive_patient_macro']['patients'] != ticket['test_cohort']['patients_with_positive'] or
                report['empty']['patients'] != ticket['test_cohort']['patients_with_empty']):
                raise ValueError('Actual evaluated denominators differ from preflight')
        if any(len({r['patient'] for r in values}) != ticket['test_cohort']['patients']
               for values in rows.values()):
            raise ValueError('Actual evaluated patient count differs from preflight')
        comparisons, uncertainty = {}, {}
        for point in ('default', 'selected'):
            a, b = 'bce_'+point, 'tversky_'+point
            comparisons[point] = comparison(reports, a, b, protocol)
            uncertainty[point] = paired_bootstrap(rows[a], rows[b], protocol['bootstrap'])
        write_json(output/'slice_metrics_PRIVATE.json', rows)
        write_json(output/'final_test_metrics.json', reports)
        write_json(output/'comparison.json', comparisons)
        write_json(output/'paired_patient_bootstrap.json', uncertainty)
        save_tables(reports, comparisons, output, protocol['operating_points'])
        figures = save_test_figures(models, dataset, rows, protocol['operating_points'],
                                    ticket['size_cutoffs'], output/'figures_PRIVATE', device)
        for arm, evidence in ticket['checkpoints'].items():
            for name, checksum in evidence['hashes'].items():
                if file_sha256(Path(evidence['run'])/name) != checksum:
                    raise ValueError(f'{arm}: original artifact changed during evaluation')
        if file_sha256(ticket['manifest_path']) != ticket['manifest_file_sha256']:
            raise ValueError('Original manifest changed during evaluation')
        summary = {'stage': 'Phase 5 final held-out test', 'test_evaluated': True,
                   'preflight_sha256': digest(ticket), 'source_commit': ticket['source']['commit'],
                   'manifest_sha256': ticket['manifest_sha256'], 'cohort': ticket['test_cohort'],
                   'operating_points': protocol['operating_points'],
                   'checkpoint_sha256': {a: e['hashes']['best.pt'] for a, e in ticket['checkpoints'].items()},
                   'size_cutoffs': ticket['size_cutoffs'], 'metrics': reports,
                   'comparisons': comparisons, 'uncertainty': uncertainty, 'figures': figures,
                   'clinical_validation': False,
                   'limitations': 'One small held-out cohort; correlated 2D slices; research FP limits; no external or clinical validation. Compare to validation honestly; no model/threshold changes after this evaluation.'}
        write_json(output/'review_summary.json', summary)
        write_json(output/'artifact_hashes.json', {p.relative_to(output).as_posix(): file_sha256(p)
                   for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})
        return output, summary
    except BaseException as error:
        write_json(output/'evaluation_failure.json', {'type': type(error).__name__, 'message': str(error),
                   'test_may_have_been_accessed': True, 'instruction': 'Preserve all files; request review before any recovery.'})
        raise
