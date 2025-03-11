import torch, torch.nn as nn
import torch.utils.model_zoo as zoo

from ttm.blocks.encoders.torchvision import *
from ttm.blocks.encoders.timm import *
from ttm.blocks.encoders.misc import __classinit

ttm_encoders = {}
ttm_encoders.update(resnet_encoders)
ttm_encoders.update(densenet_encoders)
ttm_encoders.update(efficientnet_encoders)
ttm_encoders.update(convnext_encoders)
ttm_encoders.update(mobilenet_encoders)

@__classinit
class Encoder(object):
    """Fake class for creation of ttm encoders by name
    """
    @classmethod
    def _init__class(cls):
        return cls()

    @staticmethod
    def _patch(encoder, in_channels, default_in_channels=3):
        for node in encoder.modules():
            if isinstance(node, nn.Conv3d) and node.in_channels == default_in_channels:
                break

        encoder.out_channels_ = (in_channels, *encoder.out_channels_[1:])

        weight = node.weight.detach()
        node.in_channels = in_channels

        nweight = torch.Tensor(
            node.out_channels,
            in_channels // node.groups,
            *node.kernel_size
        )

        for i in range(in_channels):
            nweight[:, i] = weight[:, i % default_in_channels]

        node.weight = nn.parameter.Parameter(nweight)

    def __call__(self, in_channels=3, depth=5, name='tv-resnet34', pretrain=None):
        """
            :args:
                in_channels: int
                    number of channels of input tensor
                depth: int
                    depth of encoder
                name: str, optional
                    name of encoder to create
                pretrain: str or None
                    name of pretrained weights

            :return:
                output: ttm.EncoderBase
                    created encoder
        """
        try:
            type = ttm_encoders[name]['encoder']
        except:
            raise KeyError('Wrong encoder name `{}`, supported encoders: {}'.format(name, list(ttm_encoders.keys())))

        params = dict(ttm_encoders[name]['params'])
        params.update(depth=depth)

        if pretrain is not None:
            try:
                url = ttm_encoders[name]['pretrain'][pretrain]
            except KeyError:
                raise KeyError('Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}'.format(
                    pretrain, name, list(ttm_encoders[name]['pretrain'].keys()),
                ))

            params.update(weights=zoo.load_url(url))

        encoder = type(**params)

        self._patch(encoder, in_channels)

        return encoder
