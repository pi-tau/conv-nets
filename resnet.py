import torch
import torch.nn as nn

from positional_norm import PositionalNorm


class ResidualBlock(nn.Module):
    """Residual block as described in https://arxiv.org/abs/1512.03385"""

    def __init__(self, in_chan, out_chan, stride=1):
        """Init a ResidualBlock.

        Args:
            in_chan: int
                Number of channels of the input tensor.
            out_chan: int
                Number of channels of the output tensor.
            stride: int, optional
                Stride of the convolving kernel. The stride is applied only to
                the first `3x3` convolutional layer, as well as to the `1x1`
                convolutional layer. Default value is 1.
        """
        super().__init__()

        # The residual function is modelled using two `3x3` convolutional layers
        # with the same number of filters. Each convolution is followed by
        # normalization and nonlinearity.
        self.res = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=stride, padding=1),
            PositionalNorm(out_chan),
            nn.ReLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding="same"),
            PositionalNorm(out_chan),
        )

        # If the residual block changes the number of channels, then the input
        # is forwarded through a `1x1` convolutional layer to transform it into
        # the desired shape for the addition operation.
        if in_chan != out_chan or stride != 1:
            self.id = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride)
        else:
            self.id = lambda x: x

    def forward(self, x):
        return torch.relu(self.res(x) + self.id(x))


class ResNet(nn.Module):
    """ResNet-18 as described in https://arxiv.org/abs/1512.03385"""

    def __init__(self, in_shape, config=None):
        """Init a ResNet model.

        Args:
            in_shape: tuple(int)
                The shape of the input tensors. Images should be reshaped
                channels first, i.e. input_shape = (C, H ,W).
            config: dict, optional
                Dictionary with configuration parameters, containing:
                stem_chan: int
                    Number of feature channels in the stem of the model.
                modules: list(tuple(int, int))
                    Each tuple contains two values:
                    The first value is the number of residual blocks in the module.
                    The second value is the number of filters for each of the
                    residual blocks in the module. All blocks in the module have
                    the same number of filters.
                out_classes: int
                    Number of output classes.
        """
        super().__init__()

        # Default configuration.
        if config is None:
            config = {
                "stem_chan" : 64,
                "modules": [(2, 64), (2, 128), (2, 256), (2, 512)],
                "out_classes": 10,
            }

        C, H, W = in_shape
        stem_chan = config["stem_chan"]
        modules = config["modules"]
        out_classes = config["out_classes"]

        # The body of the residual network consists of modules of residual blocks.
        # In each module there is a specified number of residual blocks all
        # having the same number of filters.
        # At the end of every module of residual blocks we apply a max-pool layer.
        body = []
        in_chan = stem_chan
        stride = 1
        for num_blocks, out_chan in modules:
            # The first residual block of the module is reducing the spatial
            # dimensions in half and increasing the number of channels.
            # Note that for the very first residual block we have `stride=1`,
            # i.e., no down-sampling. The reason for this is because we already
            # have a max-pool layer in the stem.
            body.append(ResidualBlock(in_chan, out_chan, stride=stride)) # (H, W) => (H//2, W//2)
            body.extend([ResidualBlock(out_chan, out_chan) for _ in range(num_blocks-1)])
            in_chan = out_chan
            stride = 2
        body = nn.Sequential(*body)

        self.net = nn.Sequential(
            # Stem.
            nn.Conv2d(in_channels=C, out_channels=stem_chan, kernel_size=7, stride=2, padding=3),
            PositionalNorm(stem_chan),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Body.
            body,

            # Head.
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(out_chan, out_classes),
        )

    def forward(self, x):
        return self.net(x)

#