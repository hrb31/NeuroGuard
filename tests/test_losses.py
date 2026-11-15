"""Analytic soft-count checks and numerical limitations, without real MRI data."""
import json
from pathlib import Path

import pytest
torch = pytest.importorskip('torch')

from src.losses import TverskyLoss, make_criterion
from src.model import UNet
from src.training import seed_everything


def test_known_counts_and_alpha_beta_order():
    # TP=.8+.6=1.4, FP=.2+.1=.3, FN=.2+.4=.6.
    p = torch.tensor([.8, .6, .2, .1], dtype=torch.float64).reshape(1, 1, 2, 2)
    y = torch.tensor([1., 1., 0., 0.], dtype=torch.float64).reshape_as(p)
    expected = 1 - (1.4+1e-6)/(1.4+.3*.3+.7*.6+1e-6)
    assert TverskyLoss()(torch.logit(p), y).item() == pytest.approx(expected, abs=1e-12)
    # At the same confidence, a missed positive costs more than an extra positive.
    y = torch.tensor([1., 0.]).reshape(1, 1, 1, 2)
    fp = TverskyLoss()(torch.logit(torch.tensor([.9, .5]).reshape_as(y)), y)
    fn = TverskyLoss()(torch.logit(torch.tensor([.5, .1]).reshape_as(y)), y)
    assert fn > fp


@pytest.mark.parametrize('kind', ['empty', 'tiny', 'large', 'background', 'perfect', 'mixed'])
def test_native_grid_finite_loss_and_gradient_direction(kind):
    y = torch.zeros(2, 1, 256, 256)
    if kind != 'empty':
        y[0, 0, 0, 0] = 1
    if kind in ('large', 'background', 'perfect', 'mixed'):
        y[0, 0, :200, :200] = 1
    if kind == 'large':
        y[1] = 1
    logits = torch.zeros_like(y)
    if kind == 'background':
        logits.fill_(-12)
    if kind == 'perfect':
        logits = torch.where(y.bool(), 12., -12.)
    logits.requires_grad_()
    loss = TverskyLoss()(logits, y)
    loss.backward()
    assert torch.isfinite(loss) and 0 <= loss.item() <= 1
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[y == 0] > 0).all()  # gradient descent lowers FP logits
    if y.any():
        assert (logits.grad[y == 1] < 0).all()  # raises positive logits
    if kind == 'perfect':
        # A nearly-perfect positive slice is low-loss. For empty GT even
        # p~6e-6 across 65,536 pixels dwarfs the tiny smoothing constant.
        assert TverskyLoss()(logits[:1], y[:1]).item() < .01
        assert TverskyLoss()(logits[1:], y[1:]).item() > .99
        assert loss.item() == pytest.approx(.5, abs=.01)
    if kind == 'empty':
        # Document the real limitation, not just a finite/nonzero assertion.
        assert loss.item() == 1.0  # float32 rounds 1 - tiny index to 1
        assert 0 < logits.grad.abs().max().item() < 1e-12


def test_per_slice_mean_includes_empty_and_is_not_batch_pooled():
    logits = torch.tensor([0., 1., -2., -.5], dtype=torch.float64).reshape(2, 1, 1, 2)
    y = torch.tensor([1., 0., 0., 0.], dtype=torch.float64).reshape_as(logits)
    criterion = TverskyLoss()
    loss = criterion(logits, y)
    expected = (criterion(logits[:1], y[:1]) + criterion(logits[1:], y[1:])) / 2
    torch.testing.assert_close(loss, expected)
    assert loss.item() != pytest.approx(criterion(logits.reshape(1, 1, 2, 2), y.reshape(1, 1, 2, 2)).item())
    logits.requires_grad_()
    assert torch.autograd.gradcheck(criterion, (logits, y))


def test_empty_prediction_limit_and_input_validation():
    criterion = TverskyLoss()
    empty = torch.zeros(1, 1, 2, 2)
    assert criterion(torch.full_like(empty, -30), empty).item() < 1e-5
    assert criterion(torch.full_like(empty, -30), torch.ones_like(empty)).item() > .99
    for value in (-1, 0, float('nan'), float('inf')):
        with pytest.raises(ValueError):
            TverskyLoss(smooth=value)
    with pytest.raises(ValueError):
        TverskyLoss(reduction='sum')
    for y in (empty + .5, empty + float('nan'), empty.squeeze(0)):
        with pytest.raises(ValueError):
            criterion(empty, y)


def test_frozen_arms_only_differ_in_loss_and_identity_and_seeded_weights():
    bce = json.loads(Path('configs/bce.json').read_text())
    tversky = json.loads(Path('configs/tversky.json').read_text())
    assert {k for k in bce if bce[k] != tversky[k]} == {'experiment', 'loss'}
    assert tversky['loss'] == {'name': 'TverskyLoss', 'alpha': .3, 'beta': .7,
                               'smooth': 1e-6, 'reduction': 'mean'}
    assert isinstance(make_criterion(bce), torch.nn.BCEWithLogitsLoss)
    assert isinstance(make_criterion(tversky), TverskyLoss)
    seed_everything(bce['seed'])
    model = UNet(**bce['model'])
    assert sum(p.numel() for p in model.parameters()) == 1942577
    seed_everything(tversky['seed'])
    other = UNet(**tversky['model'])
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, other.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError):
        make_criterion(dict(bce, experiment='tversky'))
    with pytest.raises(ValueError):
        make_criterion(dict(bce, loss='unsupported'))


def test_bce_factory_retains_exact_original_loss_and_gradient():
    config = json.loads(Path('configs/bce.json').read_text())
    logits = torch.tensor([-.3, 2., -4., .8]).reshape(2, 1, 1, 2).requires_grad_()
    target = torch.tensor([0., 1., 0., 0.]).reshape_as(logits)
    actual = make_criterion(config)(logits, target)
    expected = torch.nn.BCEWithLogitsLoss()(logits, target)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, logits)[0],
                               torch.autograd.grad(expected, logits)[0], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No local CUDA GPU')
def test_tversky_cuda_backward():
    model = UNet().cuda()
    logits = model(torch.randn(2, 3, 64, 64, device='cuda'))
    y = torch.zeros_like(logits)
    y[0, 0, :2, :2] = 1
    loss = TverskyLoss()(logits, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
