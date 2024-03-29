import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TokenReducer(nn.Module):
    """TokenLearner module.

    This is the module used for the experiments in the paper.

    Attributes:
      num_tokens: Number of tokens.
      use_sum_pooling: Whether to use the sum/average to aggregate the spatial feature after spatial attention
    """

    def __init__(self, in_channels, num_tokens, use_sum_pooling=False):
        """Applies learnable tokenization to the 2D inputs.

        Args:
          inputs: Inputs of shape `[bs, h, w, c]`.

        Returns:
          Output of shape `[bs, n_token, c]`.
        """
        super(TokenReducer, self).__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.use_sum_pooling = use_sum_pooling
        self.norm = nn.LayerNorm(self.in_channels)  # Operates on the last axis (c) of the input data.

        self.attention_maps = nn.Sequential(
            # 3 layers of conv with gelu activation as suggested
            # in the paper.
            nn.Conv2d(self.in_channels, self.num_tokens, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.GELU(),
            # This conv layer will generate the attention maps
            nn.Conv2d(self.num_tokens, self.num_tokens, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid()  # Note sigmoid for [0, 1] output
        )

    def forward(self, inputs):
        b, n, c = inputs.shape
        w = h = int(n ** 0.5)
        inputs = inputs.view(b, h, w, c)
        feature_shape = inputs.shape  # Shape:  [bs, h, w, c]

        selected = inputs
        selected = self.norm(selected)
        selected = selected.permute(0, 3, 1, 2)  # Shape:  [bs, c, h, w]
        selected = self.attention_maps(selected)  # Shape:  [bs, n_token, h, w]
        selected = selected.permute(0, 2, 3, 1)  # Shape: [bs, h, w, n_token].
        selected = selected.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2],
                                              -1)  # Shape: [bs, h*w, n_token].
        selected = selected.permute(0, 2, 1)[..., None]  # Shape: [bs, n_token, h*w, 1].

        # Reshape the input to align it with the output of the conv block.
        feat = inputs
        feat = feat.view(feature_shape[0], feature_shape[1] * feature_shape[2], -1)[:, None,
               ...]  # Shape: [bs, 1, h*w, c].

        # Element-Wise multiplication of the attention maps and the inputs
        attended_inputs = feat * selected  # (bs, n_token, h*w, c)

        if self.use_sum_pooling:
            outputs = torch.sum(attended_inputs, dim=2)  # (bs, n_token, c)
        else:
            outputs = torch.mean(attended_inputs, dim=2)

        return outputs


def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params / 1e6:.3}M')


if __name__ == '__main__':
    tklr = TokenReducer(
        in_channels=1024,
        num_tokens=32,
        use_sum_pooling=False
    )
    print_nb_params(tklr)
    x = torch.ones(1, 529, 1024)  # [bs, t, c]
    b, t, c = x.shape
    y1 = tklr(x)
    print(y1.shape)  # [256, 8, 128]
