"""Private validation curves and FLAIR predictions for review, not diagnosis."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_curves(history, directory):
    directory = Path(directory)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r['epoch'] for r in history]
    for key in ('train_loss', 'validation_loss'):
        axes[0].plot(epochs, [r[key] for r in history], label=key)
    axes[0].set(xlabel='Completed epoch', ylabel='BCE loss')
    axes[0].legend()
    axes[1].plot(epochs, [r['validation']['positive_patient_macro']['dice'] for r in history])
    axes[1].set(xlabel='Completed epoch', ylabel='Validation positive-slice patient-macro Dice', ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(directory/'training_curves.png', dpi=130)
    plt.close(fig)


@torch.no_grad()
def save_predictions(model, dataset, device, threshold, directory):
    if dataset.split != 'validation':
        raise ValueError('Phase 3 predictions must be validation only')
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
