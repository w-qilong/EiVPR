import warnings

import torch
from einops import repeat
from timm.models.layers import trunc_normal_
from torch import nn

from model.moudles.aggregate_block import Aggregator
from model.moudles.token_reducer import TokenReducer
from model.moudles import aggregation

warnings.filterwarnings('ignore')

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}


class Dinov2Backbone(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """

    def __init__(
            self,
            pretrained_model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            local_feature_layer=9,
            norm_layer=True,

            # args for mixer
            mix_in_channels=768,
            mix_token_num=256,
            mix_out_channels=768,
            mix_mix_depth=4,
            mix_mlp_ratio=4,
            mix_out_rows=3,

            # args for token reducer
            rerank=False,
            num_learned_tokens=8,
            channels_reduced=128,
    ):
        super().__init__()

        assert pretrained_model_name in DINOV2_ARCHS.keys(), f'Unknown model name {pretrained_model_name}'

        self.model_dir = '/home/cartolab/.cache/torch/hub/facebookresearch_dinov2_main'
        self.model = torch.hub.load(self.model_dir, pretrained_model_name, source='local')
        self.num_channels = DINOV2_ARCHS[pretrained_model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.local_feature_layer = local_feature_layer
        self.norm_layer = norm_layer
        # args for mixer
        self.mix_in_channels = mix_in_channels
        self.mix_token_num = mix_token_num
        self.mix_out_channels = mix_out_channels
        self.mix_mix_depth = mix_mix_depth
        self.mix_mlp_ratio = mix_mlp_ratio
        self.mix_out_rows = mix_out_rows
        # args for token reducer
        self.rerank = rerank
        self.num_learned_tokens = num_learned_tokens
        self.channels_reduced = channels_reduced

        self.mixer = Aggregator(
            in_channels=mix_in_channels,
            token_num=mix_token_num,
            out_channels=mix_out_channels,
            mix_depth=mix_mix_depth,
            mlp_ratio=mix_mlp_ratio,
            out_rows=mix_out_rows,
        )

        if self.rerank:
            # init token reducer
            self.token_reducer = TokenReducer(
                in_channels=self.num_channels,
                num_tokens=self.num_learned_tokens,
                use_sum_pooling=False
            )
            # liner for reduce dimension
            if self.channels_reduced:
                self.local_reduce = nn.Linear(self.num_channels, self.channels_reduced, bias=False)
                torch.nn.init.xavier_uniform_(self.local_reduce.weight)

    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """
        x = self.model.prepare_tokens_with_masks(x)

        # First blocks are frozen
        with torch.no_grad():
            for idx, blk in enumerate(self.model.blocks[:-self.num_trainable_blocks]):
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        # norm
        if self.norm_layer:
            x = self.model.norm(x)

        local_feature = x[:, 1:]
        mix_feature = self.mixer(local_feature)
        local_feats = local_feature.clone().detach()  # local features

        if self.rerank:
            # use token learner generate tokens
            local_feats = self.token_reducer(local_feats)

            if self.channels_reduced:
                local_feats = self.local_reduce(local_feats)

            local_feats = nn.functional.normalize(local_feats, p=2, dim=-1)

            return mix_feature, local_feats
        else:
            return mix_feature, local_feats


class Dinov2Finetune(nn.Module):
    def __init__(self,
                 # args for backbone
                 pretrained_model_name='dinov2_vitb14',
                 num_trainable_blocks=2,
                 norm_layer=True,

                 # args for mixer
                 mix_in_channels=768,
                 mix_token_num=256,
                 mix_out_channels=768,
                 mix_mix_depth=4,
                 mix_mlp_ratio=4,
                 mix_out_rows=3,

                 # args for matcher
                 rerank=True,
                 num_learned_tokens=8,
                 channels_reduced=128,
                 trans_heads=8,
                 trans_dropout=0.1,
                 trans_layers=4,
                 local_match=False,

                 # args for classifier
                 num_classifier=2,
                 ):
        super().__init__()
        # args for backbone
        self.pretrained_model_name = pretrained_model_name
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.num_channels = DINOV2_ARCHS[pretrained_model_name]

        # args for mixer
        self.mix_in_channels = mix_in_channels
        self.mix_token_num = mix_token_num
        self.mix_out_channels = mix_out_channels
        self.mix_mix_depth = mix_mix_depth
        self.mix_mlp_ratio = mix_mlp_ratio
        self.mix_out_rows = mix_out_rows

        # args for token reducer
        self.rerank = rerank
        self.num_learned_tokens = num_learned_tokens
        self.channels_reduced = channels_reduced
        self.local_match = local_match
        # args for match transformer
        self.trans_heads = trans_heads
        self.trans_dropout = trans_dropout
        self.trans_layers = trans_layers

        # args for classifier
        self.num_classifier = num_classifier

        # init backbone
        self.backbone = Dinov2Backbone(
            # backbone
            pretrained_model_name=self.pretrained_model_name,
            num_trainable_blocks=self.num_trainable_blocks,
            norm_layer=self.norm_layer,
            # token reducer
            rerank=self.rerank,
            num_learned_tokens=self.num_learned_tokens,
            channels_reduced=self.channels_reduced,
            # args for mixer
            mix_in_channels=self.mix_in_channels,
            mix_token_num=self.mix_token_num,
            mix_out_channels=self.mix_out_channels,
            mix_mix_depth=self.mix_mix_depth,
            mix_mlp_ratio=self.mix_mlp_ratio,
            mix_out_rows=self.mix_out_rows,
        )

        if self.rerank:
            # init cls token
            if self.channels_reduced:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.channels_reduced))
                self.sep_token = nn.Parameter(torch.zeros(1, 1, self.channels_reduced))
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.num_channels))
                self.sep_token = nn.Parameter(torch.zeros(1, 1, self.num_channels))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.sep_token, std=.02)

            # init transformer encoder for rerank module
            self.transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.channels_reduced if self.channels_reduced else self.num_channels,
                nhead=self.trans_heads,
                # dim_feedforward=4 * self.channels_reduced if self.channels_reduced else self.num_channels * 4,
                dim_feedforward=1024,
                dropout=self.trans_dropout,
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(self.transformer_encoder_layer, self.trans_layers)

            # init class head
            self.head = nn.Linear(self.channels_reduced if self.channels_reduced else self.num_channels,
                                  self.num_classifier)

    def dino_forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """
        g_feats, l_feats = self.backbone.forward(x)
        return g_feats, l_feats

    def forward(self, x_feats, y_feats):
        '''
        Calculate similarity score for two images by using local features.
        As the global cls feature is from local features.
        If the global cls feature is good enough, the local features should good enough to measure the similarity of two images.
        Args:
            x_feats: [batch_size, n, c]
            y_feats: [batch_size, n, c]

        Returns:
        Match score for two images
        '''
        B, N, C = x_feats.shape
        # concat cls x sep y
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        sep_tokens = repeat(self.sep_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_tokens, x_feats, sep_tokens, y_feats], dim=1)
        # input x in transformer encoder
        x = self.encoder(x)
        # input CLS feature  to  class layers
        logits = self.head(x[:, 0])
        return logits


if __name__ == '__main__':
    import time

    with torch.device('cuda'):
        model = Dinov2Finetune(
            pretrained_model_name='dinov2_vitl14',
            num_trainable_blocks=2,
            norm_layer=True,

            # args for mixer
            mix_in_channels=1024,
            mix_token_num=529,
            mix_out_channels=1024,
            mix_mix_depth=4,
            mix_mlp_ratio=2,
            mix_out_rows=4,

            # args for matcher
            rerank=True,
            num_learned_tokens=8,
            channels_reduced=0,
            trans_heads=8,
            trans_dropout=0.1,
            trans_layers=4,
            local_match=False,

            # args for classifier
            num_classifier=2,
        ).cuda()

        # calculate mean inference time

        # input1 = torch.randn((1, 3, 322, 322))
        # time_start = time.time()
        # num = 1000
        # for i in range(1000):
        #     out = model.dino_forward(input1)
        # time_end = time.time()
        # use_time = (time_end - time_start) / num
        # print('use_time:', use_time)

        input = torch.randn((1, 3, 322, 322)).cuda()
        g, l = model.dino_forward(input)
        print(g.size())

        input1 = torch.randn((1, 8, 1024)).cuda()
        input2 = torch.randn((1, 8, 1024)).cuda()
        o = model(input1, input2)
        print(o.size())
