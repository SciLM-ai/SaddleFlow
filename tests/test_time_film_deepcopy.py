"""copy.deepcopy of a TimeFiLMBackbone must give a model that uses its own FiLM weights and state."""
import copy

import torch
import torch.nn as nn

from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone


class _Block(nn.Module):
    def forward(self, x):
        return x


class _Backbone(nn.Module):
    sphere_channels, lmax, num_layers = 8, 1, 2

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def test_deepcopy_uses_its_own_films():
    torch.manual_seed(0)
    m = TimeFiLMBackbone(_Backbone(), inject_block_indices=[0, 1])
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.5)
    c = copy.deepcopy(m)
    x, t, b = torch.randn(5, 4, 8), torch.tensor([0.3]), torch.zeros(5, dtype=torch.long)
    y = m(x, t, b)
    assert (y - x).abs().max() > 1e-3  # the FiLMs act
    assert torch.equal(c(x, t, b), y)  # the copy computes the same thing
    with torch.no_grad():
        for p in c.parameters():
            p.add_(1.0)
    assert not torch.allclose(c(x, t, b), y)  # with its own weights
    assert torch.equal(m(x, t, b), y)  # and leaves the original alone
