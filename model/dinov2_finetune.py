import warnings

import torch
from einops import repeat
from timm.models.layers import trunc_normal_
from torch import nn

from model.moudles.mlp_mixer import MixAggregator
from model.moudles.token_learner import TokenLearner
from model.moudles.cross_vit import CrossViT

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

            # args for token learner
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
        # args for token learner
        self.rerank = rerank
        self.num_learned_tokens = num_learned_tokens
        self.channels_reduced = channels_reduced

        self.mixer = MixAggregator(
            in_channels=mix_in_channels,
            token_num=mix_token_num,
            out_channels=mix_out_channels,
            mix_depth=mix_mix_depth,
            mlp_ratio=mix_mlp_ratio,
            out_rows=mix_out_rows,
        )

        if self.rerank:
            # init token learner
            self.token_learner = TokenLearner(
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
                if idx == 4:
                    local_feats_ = x[:, 1:]
        x = x.detach()
        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)

        local_feature = x[:, 1:]
        mix_feature = self.mixer(local_feature)
        # local_feats = local_feature.clone().detach()  # local features

        if self.rerank:
            # global_feats = mix_feats * 1
            # # use correlation score to get top k local tokens
            # reduce_global_feats = self.reduce_dim_layer(global_feats).unsqueeze(
            #     1)  # calculate correlation between mix feature and local features
            # correlation = torch.matmul(reduce_global_feats, local_feats.permute((0, 2, 1)))
            # order_f = torch.argsort(correlation, dim=2, descending=True).squeeze()
            # selected_index = order_f[:, :64].unsqueeze(2).repeat(1, 1, 768)
            # local_feats = torch.gather(input=local_feats, index=selected_index, dim=1)

            # use token learner generate tokens
            local_feats = self.token_learner(local_feats_)

            if self.channels_reduced:
                local_feats = self.local_reduce(local_feats)

            local_feats = nn.functional.normalize(local_feats, p=2, dim=-1)
            return mix_feature, local_feats
        else:
            return mix_feature, local_feats_


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

        # args for token learner
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
            # token learner
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

        # B, N, C = x_feats.shape
        #
        # x = torch.cat([x_feats, y_feats], dim=1)
        # # input x in transformer encoder
        # x = self.encoder(x)
        # x = torch.mean(x, dim=1)
        # # input CLS feature  to  class layers
        # o = self.head(x)
        # return o


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
            rerank=False,
            num_learned_tokens=8,
            channels_reduced=0,
            trans_heads=8,
            trans_dropout=0.1,
            trans_layers=4,
            local_match=False,

            # args for classifier
            num_classifier=2,
        ).cuda()

        input1 = torch.randn((1, 3, 322, 322))
        time_start = time.time()
        num = 1000
        for i in range(1000):
            out = model.dino_forward(input1)
        time_end = time.time()
        use_time = (time_end - time_start) / num
        print('use_time:', use_time)

    # from thop import profile
    # from thop import clever_format
    #
    # input = torch.randn((1, 3, 322, 322)).cuda()
    # input1 = torch.randn((1, 8, 1024)).cuda()
    # input2 = torch.randn((1, 8, 1024)).cuda()
    #
    # macs, params = profile(model, inputs=(input1, input2), verbose=False)
    # macs, params = clever_format([macs, params], "%.3f")  # clever_format
    # print("Macs=", macs)
    # print("Params=", params)

    # macs, params = profile(model, inputs=(input1,input2), verbose=False)
    # macs, params = clever_format([macs, params], "%.3f")  # clever_format
    # print("Macs=", macs)
    # print("Params=", params)

    # a = torch.randn((2, 64, 128))
    # b = torch.randn((2, 64, 128))
    #
    # c = matcher.forward(a, b)
    # print(c)
