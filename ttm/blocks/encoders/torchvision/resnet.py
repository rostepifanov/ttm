import torch.nn as nn
import torchvision.models.resnet as tvresnet

from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from ttm.blocks.encoders.base import EncoderBase
from ttm.blocks.encoders.converters import Converter3d

class ResNetEncoder(ResNet, EncoderBase):
    """Builder for encoder from ResNet family such as ResNet, ResNeXt, and etc
    """
    def __init__(self, out_channels, depth=5, weights=None, **kwargs):
        """
            :NOTE:

            :args:
                out_channels: list of int
                    channel number of output tensors, including intermediate ones
                depth: int
                    depth of encoder
                weights: None or dict
                    weights to load
        """
        super().__init__(**kwargs)

        self.in_channels = 3
        self.out_channels_ = out_channels

        self.depth = depth

        del self.fc
        del self.avgpool

        if weights is not None:
            del weights['fc.weight']
            del weights['fc.bias']

            self.load_state_dict(weights)

        Converter3d.convert(self)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        """
            :args:
                x: [batch_size, in_channels, length] torch.tensor
                    input tensor

            :return:
                output: list of [batch_size, schannels, slength] torch.tensor
                    latent representations of input
        """
        stages = self.get_stages()
        features = []

        for i in range(self.depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

resnet_encoders = {
    'tv-resnet18': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 64, 128, 256, 512),
            'block': BasicBlock,
            'layers': [2, 2, 2, 2],
        },
        'pretrain': {
            'in1kv1': tvresnet.ResNet18_Weights.IMAGENET1K_V1.url,
        }
    },
    'tv-resnet34': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 64, 128, 256, 512),
            'block': BasicBlock,
            'layers': [3, 4, 6, 3],
        },
        'pretrain': {
            'in1kv1': tvresnet.ResNet34_Weights.IMAGENET1K_V1.url,
        }
    },
    'tv-resnet50': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
        },
        'pretrain': {
            'in1kv1': tvresnet.ResNet50_Weights.IMAGENET1K_V1.url,
            'in1kv2': tvresnet.ResNet50_Weights.IMAGENET1K_V2.url,
        }
    },
    'tv-resnet101': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
        },
        'pretrain': {
            'in1kv1': tvresnet.ResNet101_Weights.IMAGENET1K_V1.url,
            'in1kv2': tvresnet.ResNet101_Weights.IMAGENET1K_V2.url,
        }
    },
    'tv-resnet152': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 8, 36, 3],
        },
        'pretrain': {
            'in1kv1': tvresnet.ResNet152_Weights.IMAGENET1K_V1.url,
            'in1kv2': tvresnet.ResNet152_Weights.IMAGENET1K_V2.url,
        }
    },
    'tv-resnext50_32x4d': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
            'groups': 32,
            'width_per_group': 4,
        },
        'pretrain': {
            'in1kv1': tvresnet.ResNeXt50_32X4D_Weights.IMAGENET1K_V1.url,
            'in1kv2': tvresnet.ResNeXt50_32X4D_Weights.IMAGENET1K_V2.url,
        }
    },
    'tv-resnext101_32x4d': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 4,
        },
        'pretrain': {
        }
    },
    'tv-resnext101_32x8d': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 8,
        },
        'pretrain': {
            'in1kv1': tvresnet.ResNeXt101_32X8D_Weights.IMAGENET1K_V1.url,
            'in1kv2': tvresnet.ResNeXt101_32X8D_Weights.IMAGENET1K_V2.url,
        }
    },
    'tv-resnext101_32x16d': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 16,
        },
        'pretrain': {
        }
    },
    'tv-resnext101_32x32d': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 32,
        },
        'pretrain': {
        }
    },
    'tv-resnext101_32x48d': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 48,
        },
        'pretrain': {
        }
    },
    'tv-wide_resnet50_2': {
        'encoder': ResNetEncoder,
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
            'width_per_group': 128,
        },
        'pretrain': {
            'in1kv1': tvresnet.Wide_ResNet50_2_Weights.IMAGENET1K_V1.url,
            'in1kv2': tvresnet.Wide_ResNet50_2_Weights.IMAGENET1K_V2.url,
        }
    },
}
