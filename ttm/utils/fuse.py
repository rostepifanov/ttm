import torch

def fuse_conv_bn(conv, bn):
    weight, bias = torch.nn.utils.fuse_conv_bn_weights(
        conv.weight,
        conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
    )

    return weight, bias
