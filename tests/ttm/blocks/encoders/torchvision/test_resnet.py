import pytest

import torch

from ttm.blocks.encoders import Encoder
from ttm.blocks.encoders.torchvision import resnet_encoders

ENCODERS = resnet_encoders.keys()

PRETRAINED = [
    'tv-resnet18.in1k',
    'tv-resnet34.in1k',
]

@pytest.mark.resnet
@pytest.mark.encoders
@pytest.mark.parametrize('name', ENCODERS)
def test_ResNetEncoder_CASE_creation(name):
    IN_CHANNELS = 3
    DEPTH = 5

    encoder = Encoder(
        in_channels=IN_CHANNELS,
        depth=DEPTH,
        name=name
    )

    assert len(encoder.out_channels) == DEPTH + 1

    encoder.eval()

    x = torch.randn(1, IN_CHANNELS, 32, 32, 32)

    with torch.no_grad():
        y = encoder(x)

    for nchannels, yi in zip(encoder.out_channels, y):
        assert yi.shape[0] == x.shape[0]
        assert yi.shape[1] == nchannels

@pytest.mark.resnet
@pytest.mark.encoders
@pytest.mark.pretrain
@pytest.mark.parametrize('config', PRETRAINED)
def test_EfficientNetEncoder_CASE_pretrain(config):
    name, pretrain = config.split('.')

    IN_CHANNELS = 3
    DEPTH = 5

    encoder = Encoder(
        in_channels=IN_CHANNELS,
        depth=DEPTH,
        name=name,
        pretrain=pretrain,
    )

    assert len(encoder.out_channels) == DEPTH + 1

    encoder.eval()

    x = torch.randn(1, IN_CHANNELS, 32, 32, 32)

    with torch.no_grad():
        y = encoder(x)

    for nchannels, yi in zip(encoder.out_channels, y):
        assert yi.shape[0] == x.shape[0]
        assert yi.shape[1] == nchannels
