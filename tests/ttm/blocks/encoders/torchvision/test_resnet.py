import pytest

import torch

from ttm.blocks.encoders import Encoder
from ttm.blocks.encoders.torchvision import resnet_encoders

ENCODERS = resnet_encoders.keys()

PRETRAINED = [
    'tv-resnet18.in1kv1',
    'tv-resnet34.in1kv1',
    'tv-resnet50.in1kv1',
    'tv-resnet50.in1kv2',
    'tv-resnet101.in1kv1',
    'tv-resnet101.in1kv2',
    'tv-resnet152.in1kv1',
    'tv-resnet152.in1kv2',
    'tv-resnext50_32x4d.in1kv1',
    'tv-resnext50_32x4d.in1kv2',
    'tv-resnext101_32x8d.in1kv1',
    'tv-resnext101_32x8d.in1kv2',
    'tv-wide_resnet50_2.in1kv1',
    'tv-wide_resnet50_2.in1kv2',
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
