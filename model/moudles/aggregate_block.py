import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class AggregateBlock(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)


class Aggregator(nn.Module):
    def __init__(self,
                 in_channels=768,
                 token_num=256,
                 out_channels=768,
                 mix_depth=4,
                 mlp_ratio=4,
                 out_rows=3,
                 ):
        super().__init__()

        self.in_channels = in_channels  # depth of input feature maps
        self.token_num = token_num  # depth of input feature maps
        self.out_channels = out_channels  # depth wise projection dimension
        self.out_rows = out_rows  # row wise projection dimesion

        self.mix_depth = mix_depth  # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio  # ratio of the mid projection layer in the mixer block

        self.mix = nn.Sequential(*[
            AggregateBlock(in_dim=token_num, mlp_ratio=mlp_ratio)
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(token_num, out_rows)

    def forward(self, x):  # [B,N,C]
        x = x.permute(0, 2, 1)  # [B,C,N]
        x = self.mix(x)  # [B,C,N]
        x = x.permute(0, 2, 1)  # [B,N,C]
        x = self.channel_proj(x)  # [B,N,out_channels]
        x = x.permute(0, 2, 1)  # [B,out_channels,N]
        x = self.row_proj(x)  # [B,out_channels,out_rows]
        x = F.normalize(x.flatten(1), p=2, dim=-1)  # [B,out_channels * out_rows]
        return x

def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')

if __name__ == '__main__':

    a=torch.randn((1,529,1024))
    agg=Aggregator(
        in_channels=1024,
        token_num=529,
        out_channels=1024,
        mix_depth=6,
        mlp_ratio=2,
        out_rows=4,
    )
    print_nb_params(agg)
    out=agg(a)
    print(out.shape)