import math

import torch
from torch.autograd import gradcheck

import sys

sys.path.append("./src")

from anatgs.dynamic.shroud import amsterdam_shroud


def _corrcoef(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = a - a.mean()
    b = b - b.mean()
    return torch.sum(a * b) / torch.sqrt(torch.sum(a * a) * torch.sum(b * b) + eps)


def test_amsterdam_shroud_synthetic_phase_corr():
    n, h, w = 64, 64, 32
    y = torch.arange(h, dtype=torch.float32)
    projs = []
    phase = []
    for i in range(n):
        ph = 2.0 * math.pi * i / n
        center = h / 2.0 + 10.0 * math.sin(ph)
        g = torch.exp(-0.5 * ((y - center) / 3.0) ** 2)[:, None].repeat(1, w)
        projs.append(g)
        phase.append(math.sin(ph))
    projs_t = torch.stack(projs, dim=0)
    target = torch.tensor(phase, dtype=torch.float32)
    pred = amsterdam_shroud(projs_t)
    corr = _corrcoef(pred, target)
    assert float(corr) > 0.95


def test_amsterdam_shroud_gradcheck():
    n, h, w = 8, 10, 6
    torch.manual_seed(0)
    projs = torch.rand(n, h, w, dtype=torch.float64, requires_grad=True)
    assert gradcheck(amsterdam_shroud, (projs,), eps=1e-6, atol=1e-4, rtol=1e-3)

