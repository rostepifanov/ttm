import pytest

import torch

from ttm.blocks.encoders import Encoder
from ttm.blocks.encoders.timm import efficientnet_encoders

ENCODERS = efficientnet_encoders.keys()

PRETRAINED = [
    'timm-efficientnetv2-b0.in1k',
    'timm-efficientnetv2-b1.in1k',
    'timm-efficientnetv2-b2.in1k',
    'timm-efficientnetv2-b3.in1k',
    'timm-efficientnetv2-s.in1k',
    'timm-efficientnetv2-s.in21k',
    'timm-efficientnetv2-s.in21k_ft_in1k',
    'timm-efficientnetv2-m.in1k',
    'timm-efficientnetv2-m.in21k',
    'timm-efficientnetv2-m.in21k_ft_in1k',
    'timm-efficientnetv2-l.in1k',
    'timm-efficientnetv2-l.in21k',
    'timm-efficientnetv2-l.in21k_ft_in1k',
    'timm-efficientnetv2-xl.in21k',
    'timm-efficientnetv2-xl.in21k_ft_in1k',
]

@pytest.mark.efficientnet
@pytest.mark.encoders
@pytest.mark.parametrize('name', ENCODERS)
def test_EfficientNetEncoder_CASE_creation(name):
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

@pytest.mark.efficientnet
@pytest.mark.encoders
@pytest.mark.parametrize('name', ENCODERS)
def test_EfficientNetEncoder_CASE_unused_layer_check(name):
    IN_CHANNELS = 3
    DEPTH = 5

    encoder = Encoder(
        in_channels=IN_CHANNELS,
        depth=DEPTH,
        name=name
    )

    encoder.train()

    x = torch.randn(2, IN_CHANNELS, 32, 32, 32)
    y = encoder(x)[-1]

    loss = y.sum()
    loss.backward()

    for n, p in encoder.named_parameters():
        assert p.grad is not None, f'{n} parameter has not a grad'

@pytest.mark.efficientnet
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
