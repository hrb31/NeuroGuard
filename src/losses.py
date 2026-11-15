"""Binary logit losses for the matched BCE/Tversky experiments."""
import math

import torch
from torch import nn


class TverskyLoss(nn.Module):
    """Mean of per-slice ``1 - (TP+s)/(TP+alpha*FP+beta*FN+s)``.

    TP/FP/FN are soft probability sums over C,H,W after sigmoid(logits).
    Alpha weights false positives; beta weights false negatives. Smoothing is
    added to numerator AND denominator, in pixel-sum units. Empty slices remain
    in the mean. With s=1e-6 their FP gradients can be extremely small: a finite
    gradient is not evidence of a strong background penalty. No hard threshold,
    hybrid term or foreground-only filtering is used. Metrics use no smoothing.
    """

    def __init__(self, alpha=.3, beta=.7, smooth=1e-6, reduction='mean'):
        super().__init__()
        if any(not math.isfinite(v) or v <= 0 for v in (alpha, beta, smooth)):
            raise ValueError('alpha, beta and smooth must be finite and positive')
        if reduction != 'mean':
            raise ValueError('Only per-slice batch-mean reduction is supported')
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, targets):
        if (logits.shape != targets.shape or logits.ndim != 4 or
                logits.shape[1] != 1 or not logits.numel()):
            raise ValueError('Expected matching nonempty B1HW logits and targets')
        if not logits.is_floating_point() or not targets.is_floating_point():
            raise ValueError('Logits and binary targets must be floating point')
        if not torch.isfinite(logits).all() or not ((targets == 0) | (targets == 1)).all():
            raise ValueError('Expected finite logits and binary targets')
        probabilities = logits.sigmoid()
        dims = (1, 2, 3)
        tp = (probabilities * targets).sum(dims)
        fp = (probabilities * (1 - targets)).sum(dims)
        fn = ((1 - probabilities) * targets).sum(dims)
        index = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        return (1 - index).mean()


def make_criterion(config):
    """Explicit loss selection; the original BCE config remains valid unchanged."""
    loss = config['loss']
    if config['experiment'] == 'bce' and loss == 'BCEWithLogitsLoss(mean)':
        return nn.BCEWithLogitsLoss()
    if config['experiment'] == 'tversky' and isinstance(loss, dict):
        if set(loss) == {'name', 'alpha', 'beta', 'smooth', 'reduction'} and loss['name'] == 'TverskyLoss':
            return TverskyLoss(**{k: v for k, v in loss.items() if k != 'name'})
    raise ValueError('Unsupported experiment/loss configuration')
