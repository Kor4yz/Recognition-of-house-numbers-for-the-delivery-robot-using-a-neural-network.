import torch
from src.models.resnet import SVHNResNet

def test_forward_shape():
    m = SVHNResNet()
    x = torch.randn(2,3,32,32)
    y = m(x)
    assert y.shape == (2,10)
