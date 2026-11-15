"""Private validation curves and FLAIR predictions for review, not diagnosis."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


def compare_validation(reports, bce_reference, *, fp_limit=.1, fp_increase_limit=.05):
    """Compare measured validation reports to rounded user-verified BCE aggregates.

    Deltas are Tversky minus BCE, multiplied by 100: percentage points, not
    relative percentage changes. A positive FP delta is more false alarms.
    Missing BCE values are never reconstructed or invented.
    """
    if bce_reference['split'] != 'validation':
        raise ValueError('Comparison reference must be validation only')
    result = {'stage': 'preliminary validation comparison',
              'reference_provenance': bce_reference['provenance'],
              'difference_units': 'percentage points (Tversky minus rounded BCE)',
              'test_evaluated': False, 'operating_points': {}}
    for point in ('default', 'selected'):
        report = reports[point]
        cohort = bce_reference['cohort']
        if report['slices'] != cohort['slices'] or report['positive_patient_macro']['slices'] != cohort['positive_slices']:
            raise ValueError('Validation cohort counts differ from BCE reference')
        positive, small = report['positive_patient_macro'], report['size_groups']['small']
        measured = {k: positive[k] for k in ('dice', 'recall', 'precision')}
        measured.update(empty_fp_rate=report['empty']['false_positive_rate'],
                        small_dice=small['dice'], small_recall=small['recall'])
        reference = bce_reference[point]
        deltas = {k: 100*(v-reference[k]) if v is not None else None for k, v in measured.items()}
        fp = measured['empty_fp_rate']
        guardrail = (fp <= fp_limit and fp-reference['empty_fp_rate'] <= fp_increase_limit
                     if fp is not None else None)
        result['operating_points'][point] = {
            'bce_threshold': reference['threshold'],
            'tversky_threshold': reports['default_threshold'] if point == 'default' else reports['selection']['threshold'],
            'bce': {k: reference[k] for k in measured}, 'tversky': measured,
            'difference_pp': deltas, 'research_fp_guardrail_met': guardrail}
    result['selection_status'] = reports['selection']['status']
    result['interpretation'] = ('Inspect Dice/recall/precision and FP differences together; higher recall alone '
        'does not establish improvement. FP guardrails are research choices, not clinical safety limits. '
        'No uncertainty/significance claim from rounded aggregates or one split. Training losses are different objectives.')
    return result


def save_curves(history, directory, loss_name='BCE'):
    directory = Path(directory)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r['epoch'] for r in history]
    for key in ('train_loss', 'validation_loss'):
        axes[0].plot(epochs, [r[key] for r in history], label=key)
    axes[0].set(xlabel='Completed epoch', ylabel=f'{loss_name} loss')
    axes[0].legend()
    axes[1].plot(epochs, [r['validation']['positive_patient_macro']['dice'] for r in history])
    axes[1].set(xlabel='Completed epoch', ylabel='Validation positive-slice patient-macro Dice', ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(directory/'training_curves.png', dpi=130)
    plt.close(fig)


@torch.no_grad()
def save_predictions(model, dataset, device, threshold, directory):
    if dataset.split != 'validation':
        raise ValueError('Development predictions must be validation only')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    positives = sorted([i for i, r in enumerate(dataset.records) if r['foreground'] > 0],
                       key=lambda i: dataset.records[i]['area_fraction'])
    empty = [i for i, r in enumerate(dataset.records) if not r['foreground']]
    indices = ([positives[0], positives[len(positives)//2], positives[-1]] if positives else []) + empty[:1]
    model.eval()
    for number, index in enumerate(dict.fromkeys(indices)):
        sample = dataset[index]
        x, y = sample['image'], sample['mask'][0].numpy()
        pred = (model(x[None].to(device)).sigmoid()[0, 0] >= threshold).cpu().numpy()
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for ax in axes:
            ax.imshow(x[1].numpy(), cmap='gray')
            ax.axis('off')
        axes[0].set_title('FLAIR (stored channel 1)')
        axes[1].imshow(np.ma.masked_where(y == 0, y), cmap='autumn', alpha=.5, vmin=0, vmax=1)
        axes[1].set_title('Annotation')
        axes[2].imshow(np.ma.masked_where(~pred, pred), cmap='winter', alpha=.5, vmin=0, vmax=1)
        axes[2].set_title(f'Prediction t={threshold:g}')
        fig.suptitle('Validation example; research prototype, not medical advice')
        fig.tight_layout()
        fig.savefig(directory/f'validation_example_{number}.png', dpi=130)
        plt.close(fig)
