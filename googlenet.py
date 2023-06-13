import torch
import torch.nn as nn

from positional_norm import PositionalNorm


class InceptionBlock(nn.Module):
    """InceptionBlock as described in https://arxiv.org/abs/1409.4842

    The inception block solves the problem of selecting kernel sizes by simply
    applying several kernels and concatenating the outputs.
    """

    def __init__(self, in_chan, reduce_chan, out_chan):
        """Init an Inception block.

        Args:
            in_chan: int
                The number of channels from the previous layer.
            reduce_chan: dict
                A dictionary with keys "3x3" and "5x5" specifying the
                dimensionality reduction for the respective branches.
            out_chan: dict
                A dictionary with keys "1x1", "3x3", "5x5", "max" specifying the
                output channels for the respective branches.
            ...
        """
        super().__init__()

        # An inception block applies four convolution blocks separately on the
        # same feature map: a 1x1, 3x3, 5x5 convolution, and a max pool operation.
        # This allows the network to look at the same data with different
        # receptive fields. In addition, `1x1` convolutions are used to compute
        # reductions before the expensive `3x3` and `5x5` convolutions, and also
        # after the max pool operation. They also include the use of a normalizing
        # layer and a `relu` non-linearity. Finally, the outputs along each of
        # the four parallel branches are concatenated along the channel dimension.
        #
        # 1x1 convolution branch.
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan["1x1"], kernel_size=1),
            PositionalNorm(out_chan["1x1"]),
            nn.ReLU(),
        )

        # 3x3 convolution branch.
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_chan, reduce_chan["3x3"], kernel_size=1),
            PositionalNorm(reduce_chan["3x3"]),
            nn.ReLU(),
            nn.Conv2d(reduce_chan["3x3"], out_chan["3x3"], kernel_size=3, padding="same"),
            PositionalNorm(out_chan["3x3"]),
            nn.ReLU(),
        )

        # 5x5 convolution branch.
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_chan, reduce_chan["5x5"], kernel_size=1),
            PositionalNorm(reduce_chan["5x5"]),
            nn.ReLU(),
            nn.Conv2d(reduce_chan["5x5"], out_chan["5x5"], kernel_size=5, padding="same"),
            PositionalNorm(out_chan["5x5"]),
            nn.ReLU(),
        )

        # Max-pool branch.
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_chan, out_chan["max"], kernel_size=1),
            PositionalNorm(out_chan["max"]),
            nn.ReLU(),
        )

    def forward(self, x):
        out_1x1 = self.conv_1x1(x)
        out_3x3 = self.conv_3x3(x)
        out_5x5 = self.conv_5x5(x)
        out_max_pool = self.max_pool(x)
        return torch.cat([out_1x1, out_3x3, out_5x5, out_max_pool], dim=1)



class GoogLeNet(nn.Module):
    """GoogLeNet as described in https://arxiv.org/abs/1409.4842"""

    def __init__(self, in_shape, config=None):
        """Init a GoogLeNet model.

        Args:
            in_shape: tuple(int)
                The shape of the input tensors. Images should be reshaped
                channels first, i.e. input_shape = (C, H ,W).
            config: dict, optional
                Dictionary with configuration parameters, containing:
                stem_chan: int
                    Number of feature channels in the stem of the model.
                gr1: dict
                gr2: dict
                gr3: dict
                    Configuration parameters for the Inception blocks in
                    groups 1, 2 and 3 respectively, containing:
                    reduce_chan: dict
                        A dictionary with keys "3x3" and "5x5" specifying the
                        dimensionality reduction for the respective branches.
                    out_chan: dict
                        A dictionary with keys "1x1", "3x3", "5x5", "max"
                        specifying the output channels for the respective branches.
                out_classes: int
                    Number of output classes.
        """
        super().__init__()

        # Default configuration.
        if config is None:
            config = {
                "stem_chan": 64,
                "gr1": {
                    "reduce_chan": {"3x3": 32, "5x5": 16},
                    "out_chan":    {"1x1": 16, "3x3": 32, "5x5":8, "max":8},
                },
                "gr2": {
                    "reduce_chan": {"3x3": 32, "5x5": 16},
                    "out_chan":    {"1x1": 24, "3x3": 48, "5x5":12, "max":12},
                },
                "gr3": {
                    "reduce_chan": {"3x3": 48, "5x5": 16},
                    "out_chan":    {"1x1": 32, "3x3": 64, "5x5":16, "max":16},
                },
                "out_classes": 10,
            }

        C, H, W = in_shape
        stem_chan = config["stem_chan"]
        gr1_chan = sum(v for v in config["gr1"]["out_chan"].values())
        gr2_chan = sum(v for v in config["gr2"]["out_chan"].values())
        gr3_chan = sum(v for v in config["gr3"]["out_chan"].values())
        out_classes = config["out_classes"]

        # The GoogLeNet architecture consists of stacking multiple Inception
        # blocks with occasional max pooling to reduce the height and the width
        # of the feature maps. It exhibits a clear distinction among the stem
        # (data ingest), body (data processing), and head (prediction).
        self.net = nn.Sequential(
            # # Stem. The stem is similar to a regular conv net.
            # nn.Conv2d(in_channels=C, out_channels=stem_chan, kernel_size=7, stride=2, padding=3),
            # PositionalNorm(stem_chan),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # nn.Conv2d(in_channels=stem_chan, out_channels=stem_chan, kernel_size=1),
            # PositionalNorm(stem_chan),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=stem_chan, out_channels=stem_chan, kernel_size=3, padding="same"),
            # PositionalNorm(stem_chan),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Stem for CIFAR-10.
            nn.Conv2d(in_channels=C, out_channels=stem_chan, kernel_size=3, padding="same"),
            PositionalNorm(stem_chan),
            nn.ReLU(),

            # Body. The body consists of stacking a total of nine inception blocks,
            # arranged into three groups with max-pooling layer in between to
            # reduce the dimensionality of the input (always halved).
            # The original GoogLeNet uses different configurations for each and
            # every inception block. Here for brevity we will use the same
            # configuration throughout a given group.
            # Group 1.
            InceptionBlock(in_chan=stem_chan, **config["gr1"]),
            InceptionBlock(in_chan=gr1_chan, **config["gr1"]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Group 2.
            InceptionBlock(in_chan=gr1_chan, **config["gr2"]),
            InceptionBlock(in_chan=gr2_chan, **config["gr2"]),
            InceptionBlock(in_chan=gr2_chan, **config["gr2"]),
            InceptionBlock(in_chan=gr2_chan, **config["gr2"]),
            InceptionBlock(in_chan=gr2_chan, **config["gr2"]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Group 3.
            InceptionBlock(in_chan=gr2_chan, **config["gr3"]),
            InceptionBlock(in_chan=gr3_chan, **config["gr3"]),

            # Head. The head uses a global average pooling layer to change the
            # height and width of each channel to 1, just as in NiN. Finally, a
            # fully-connected layers is applied to produce the class logits.
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=gr3_chan, out_features=out_classes),
        )

    def forward(self, x):
        return self.net(x)
#