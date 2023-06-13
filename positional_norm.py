import torch.nn as nn


class PositionalNorm(nn.LayerNorm):
    """PositionalNorm is a normalization layer used for 3D inputs that
    normalizes exclusively across the channels dimension.
    """

    def forward(self, x):
        # The input is of shape (B, C, H, W). Transpose the input so that the
        # channels are pushed to the last dimension and then run the standard
        # LayerNorm layer.
        x = x.permute(0, 2, 3, 1).contiguous()
        out = super().forward(x)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out

#