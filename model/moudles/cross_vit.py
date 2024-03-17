import torch
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from torch import nn, einsum


class TokenEmbedder(nn.Module):
    def __init__(
            self,
            num_patches=529,
            local_input_dim=384,
            local_out_dim=128,
            dropout=0.1
    ):
        super().__init__()
        self.local_reduce = nn.Linear(local_input_dim, local_out_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.local_reduce.weight)

        self.cls_token = nn.Parameter(torch.randn(1, 1, local_out_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, local_out_dim))
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _ = x.shape
        x = self.local_reduce(x)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        return self.dropout(x)


# attention
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)

        context = context if context is not None else x

        if kv_include_self:
            context = torch.cat((x, context),
                                dim=1)  # cross attention requires CLS token, includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# feedforward
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# transformer encoder, for patches of two images
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(lg_dim, heads=heads, dim_head=dim_head, dropout=dropout),
                Attention(sm_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]),
                                                                   (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context=lg_patch_tokens, kv_include_self=True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context=sm_patch_tokens, kv_include_self=True) + lg_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim=1)
        return sm_tokens, lg_tokens


# multi-scale encoder
class MultiScaleEncoder(nn.Module):
    def __init__(
            self,
            depth,
            sm_dim,
            lg_dim,
            sm_enc_params,
            lg_enc_params,
            cross_attn_heads,
            cross_attn_depth,
            cross_attn_dim_head=64,
            dropout=0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim=sm_dim, dropout=dropout, **sm_enc_params),
                Transformer(dim=lg_dim, dropout=dropout, **lg_enc_params),
                CrossTransformer(sm_dim=sm_dim, lg_dim=lg_dim, depth=cross_attn_depth, heads=cross_attn_heads,
                                 dim_head=cross_attn_dim_head, dropout=dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens


class CrossViT(nn.Module):
    def __init__(
            self,
            num_patches=529,
            num_classes=2,
            local_input_dim=384,

            sm_dim=128,
            lg_dim=128,

            sm_enc_depth=1,
            sm_enc_heads=8,
            sm_enc_mlp_dim=1024,
            sm_enc_dim_head=64,

            lg_enc_depth=1,
            lg_enc_heads=8,
            lg_enc_mlp_dim=1024,
            lg_enc_dim_head=64,

            cross_attn_depth=2,
            cross_attn_heads=8,
            cross_attn_dim_head=64,
            depth=1,

            dropout=0.1,
            emb_dropout=0.1
    ):
        super().__init__()
        self.sm_image_embedder = TokenEmbedder(num_patches=num_patches, local_input_dim=local_input_dim,
                                               local_out_dim=sm_dim, dropout=emb_dropout)
        self.lg_image_embedder = TokenEmbedder(num_patches=num_patches, local_input_dim=local_input_dim,
                                               local_out_dim=lg_dim, dropout=emb_dropout)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth=depth,
            sm_dim=sm_dim,
            lg_dim=lg_dim,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dim_head=cross_attn_dim_head,
            cross_attn_depth=cross_attn_depth,
            sm_enc_params=dict(
                depth=sm_enc_depth,
                heads=sm_enc_heads,
                mlp_dim=sm_enc_mlp_dim,
                dim_head=sm_enc_dim_head
            ),
            lg_enc_params=dict(
                depth=lg_enc_depth,
                heads=lg_enc_heads,
                mlp_dim=lg_enc_mlp_dim,
                dim_head=lg_enc_dim_head
            ),
            dropout=dropout
        )

        self.mlp_head = nn.Sequential(nn.LayerNorm(sm_dim * 2), nn.Linear(sm_dim * 2, num_classes))

    def forward(self, x_feats, y_feats):
        sm_tokens = self.sm_image_embedder(x_feats)
        lg_tokens = self.lg_image_embedder(y_feats)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        cat_cls = torch.cat([sm_cls, lg_cls], dim=1)

        logits = self.mlp_head(cat_cls)
        return logits


if __name__ == '__main__':
    m = CrossViT()
    a = torch.randn((2, 64, 384))
    b = torch.randn((2, 64, 384))

    out = m(a, b)
    print(out.shape)
