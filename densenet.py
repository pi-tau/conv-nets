import torch
import torch.nn as nn

from positional_norm import PositionalNorm


class DenseBlock(nn.Module):
    """DenseBlock as described in https://arxiv.org/abs/1608.06993

    The dense block consists of multiple convolutional layers, each having the
    same number of output channels. However, each layer takes as input the
    original input concatenated with the output of all previous layers from the
    block.
    """

    def __init__(self, in_chan, out_chan, n_layers):
        """Init a DenseBlock.

        Args:
            in_chan: int
                Number of channels of the input tensor.
            out_chan: int
                Number of output channels for each of the conv layers.
            n_layers: int
                Number of conv layers in the dense block.
        """
        super().__init__()

        # The dense block consists of a `1x1` convolutional layer followed by a
        # `3x3` convolutional layers. Each convolutional layer is preceded by a
        # normalization layer and a non-linear activation, utilizing the
        # so-called "pre-activation" structure.
        # See: https://arxiv.org/abs/1603.05027
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Sequential(
                PositionalNorm(in_chan),
                nn.ReLU(),
                nn.Conv2d(in_chan, out_chan, kernel_size=1),
                PositionalNorm(out_chan),
                nn.ReLU(),
                nn.Conv2d(out_chan, out_chan, kernel_size=3, padding="same"),
            ))
            in_chan += out_chan
        self.layers = layers

    def forward(self, x):
        out = x
        for l in self.layers:
            y = l(out)
            out = torch.cat((out, y), dim=1)
        return out


class DenseNet(nn.Module):
    """DenseNet as described in https://arxiv.org/abs/1608.06993"""

    def __init__(self, in_shape, config):
        """Init a ResNet model.

        Args:
            in_shape: tuple(int)
                The shape of the input tensors. Images should be reshaped
                channels first, i.e. input_shape = (C, H ,W).
            config: dict
                Dictionary with configuration parameters, containing:
                stem_chan: int
                    Number of feature channels in the stem of the model.
                modules: list(tuple(int, int))
                    Each tuple contains two values:
                    The first value is the number of convolutional layers in the
                    dense block.
                    The second value is the number of filters for each of the
                    convolutional layers.
                out_classes: int
                    Number of output classes.
        """
        super().__init__()
        C, H, W = in_shape
        stem_chan = config["stem_chan"]
        modules = config["modules"]
        out_classes = config["out_classes"]

        # The body of the DenseNet consists of stacking multiple Dense blocks.
        # Since each dense block will increase the number of channels, a so-called
        # transition layer is added between dense blocks to control the
        # complexity. The transition layer reduces the number of channels by
        # using a `1x1` convolution. It also halves the spatial dimensions using
        # average pooling with stride 2.
        body = []
        in_chan = stem_chan
        for i, (n_layers, out_chan) in enumerate(modules):
            # Dense block.
            body.append(DenseBlock(in_chan, out_chan, n_layers))
            chan = in_chan + n_layers * out_chan

            # Transition layer.
            if i < len(modules)-1:
                body.append(nn.Sequential(
                    PositionalNorm(chan),
                    nn.ReLU(),
                    nn.Conv2d(chan, chan//2, kernel_size=1), # C => C // 2
                    nn.AvgPool2d(kernel_size=2, stride=2)    # (H, W) => (H//2, W//2)
                ))
                in_chan = chan//2
        body.append(PositionalNorm(chan))
        body.append(nn.ReLU())
        body = nn.Sequential(*body)

        self.net = nn.Sequential(
            # Stem.
            nn.Conv2d(in_channels=C, out_channels=stem_chan, kernel_size=3, padding="same"),
            PositionalNorm(stem_chan),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # (H, W) => (H//2, W//2)

            # Body.
            body,

            # Head.
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(chan, out_classes),
        )

    def forward(self, x):
        return self.net(x)


# Default DenseNet model for CIFAR-10.
DenseNet_CIFAR10 = DenseNet(
    in_shape=(3, 32, 32),
    config={
        "stem_chan": 64,
        "modules": [(6, 16), (12, 16), (32, 16), (32, 16)],
        "out_classes": 10,
    },
)
#