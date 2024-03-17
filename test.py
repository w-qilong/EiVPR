import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
from torchsummary import summary
import time

with torch.device('cuda'):
    # init transformer encoder for rerank module
    transformer_encoder_layer = nn.TransformerEncoderLayer(
        d_model=1024,
        nhead=8,
        # dim_feedforward=4 * self.channels_reduced if self.channels_reduced else self.num_channels * 4,
        dim_feedforward=1024,
        dropout=0.1,
        batch_first=True,
        norm_first=False,
    )
    encoder = nn.TransformerEncoder(transformer_encoder_layer, 12)
    # init class head
    head = nn.Linear(1024, 2)

    input1 = torch.randn((1, 1060, 1024))
    time_start = time.time()
    num = 1000
    for i in range(1000):
        out = encoder(input1)
        out = head(out)
    time_end = time.time()
    use_time = (time_end - time_start) / num
    print('use_time:', use_time)

# macs, params = profile(encoder, inputs=(input1,), verbose=False)
# macs, params = clever_format([macs, params], "%.3f")  # clever_format
# print("Macs=", macs)
# print("Params=", params)
#
