import torch

from code.model import resnet20


def test_resnet20_output_and_size():
    model = resnet20()
    output = model(torch.randn(4, 3, 32, 32))
    assert output.shape == (4, 10)
    assert 250_000 < sum(p.numel() for p in model.parameters()) < 300_000
