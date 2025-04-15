import torch.nn as nn

class ConvNormAct(nn.Module):
    """Convolution layer with Norm and Act
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            stride,
            bias,
        ):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
        )

        self.norm = nn.BatchNorm3d(
            num_features=out_channels,
        )

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x
