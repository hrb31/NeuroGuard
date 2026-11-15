"""Train-only tiny-subset debugging check, never a generalisation result."""
from pathlib import Path
import numpy as np
import torch

from .model import UNet
from .training import seed_everything
from .losses import make_criterion


def tiny_overfit(dataset, config, directory, *, steps=100):
    if dataset.split != 'train':
        raise ValueError('Tiny overfit requires training data')
    seed_everything(config['seed'])
    # Prefer four positive training slices. Fixed tensors remove augmentation noise.
    indices = [i for i, r in enumerate(dataset.records) if r['foreground'] > 0][:4]
    if not indices:
        raise ValueError('No positive training slices for smoke test')
    samples = [dataset[i] for i in indices]
    x = torch.stack([s['image'] for s in samples])
    y = torch.stack([s['mask'] for s in samples])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, y = x.to(device), y.to(device)
    model = UNet(**config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['lr'])
    criterion = make_criterion(config)
    losses = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if logits.shape != y.shape:
            raise ValueError('Smoke logit/mask shapes differ')
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite smoke loss')
        loss.backward()
        gradients = [p.grad for p in model.parameters()]
        if any(g is None for g in gradients) or not torch.stack([torch.isfinite(g).all() for g in gradients]).all():
            raise FloatingPointError('Missing/nonfinite smoke gradient')
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        output = model(x)
        pred = output.sigmoid() >= .5
        tp = (pred*y.bool()).sum().item()
        dice = 2*tp/(pred.sum().item()+y.sum().item())
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].plot(range(1, steps+1), losses)
    axes[0].set(xlabel='Tiny training update', ylabel=f"{config['experiment'].upper()} loss")
    for ax in axes[1:]:
        ax.imshow(x[0, 1].cpu().numpy(), cmap='gray')
        ax.axis('off')
    target = y[0, 0].cpu().numpy()
    prediction = pred[0, 0].cpu().numpy()
    axes[1].imshow(np.ma.masked_where(target == 0, target), cmap='autumn', alpha=.5)
    axes[1].set_title('Tiny training annotation')
    axes[2].imshow(np.ma.masked_where(~prediction, prediction), cmap='winter', alpha=.5)
    axes[2].set_title('Tiny training prediction')
    fig.tight_layout()
    fig.savefig(directory/'smoke_debug.png', dpi=120)
    plt.close(fig)
    path = directory/'smoke.pt'
    torch.save(model.state_dict(), path)
    restored = UNet(**config['model']).to(device)
    restored.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x), output, rtol=0, atol=0)
    decreased = float(np.mean(losses[-min(10, steps):])) < losses[0]
    result = {'steps': steps, 'training_slices': len(samples), 'first_loss': losses[0],
              'final_loss': losses[-1], 'loss_decreased': decreased,
              'tiny_training_pooled_dice': dice, 'checkpoint_roundtrip': True,
              'device': str(device), 'debug_only': True,
              'interpretation': 'Loss decrease is a minimum gate; inspect Dice/overlays for actual fitting. No generalisation claim.'}
    if not decreased:
        raise RuntimeError(f'Tiny train-only loss did not decrease: {result}')
    return result
