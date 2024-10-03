import torch
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from typing import Any
from torchinfo import summary


class PatchEmbedding(torch.nn.Module):

    def __init__(self, num_channels: int, num_classes: int, img_res: int,
                 patch_size: int, embed_size: int, **kwargs: dict[str, Any]) -> None:
        super().__init__()

        self.num_patches = (img_res // patch_size)**2

        self.projection = torch.nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> (b h w) t (p1 p2 c)', p1=patch_size, p2=patch_size),
            torch.nn.Linear((num_channels - 1)*patch_size*patch_size, embed_size)
        )

        self.to_temporal_embedding = torch.nn.Linear(365, embed_size)
        self.cls_tokens = torch.nn.Parameter(torch.randn(1, num_classes, embed_size))


    def forward(self, x: torch.tensor) -> torch.tensor:
        b, t, _, _, _ = x.shape

        # Splitting images and dates
        x_t = x[:, :, -1, 0, 0].to(torch.int64)
        x = x[:, :, :-1]

        # Creating patch embeddings
        x = self.projection(x)
        x = rearrange(x, '(b p) t e -> b p t e', b=b)

        # Adding temporal embeddings
        x_t = torch.nn.functional.one_hot(x_t, num_classes=365)
        x_t = x_t.to(torch.float32).reshape(-1, 365)

        temporal_pos = self.to_temporal_embedding(x_t)
        temporal_pos = rearrange(temporal_pos, '(b t) e -> b 1 t e', b=b)

        x += temporal_pos

        # Prepending the class tokens
        x = rearrange(x, 'b p t e -> (b p) t e')
        cls_tokens = repeat(self.cls_tokens, '() n e -> (b p) n e', b=b, p=self.num_patches)

        x = torch.cat((cls_tokens, x), dim=1)

        return x


class MultiHeadAttention(torch.nn.Module):
    '''
    Computes the multi-head attention matrix using queries and values and uses it to "attend" to the values
    '''

    def __init__(self, embed_size: int, num_heads: int, dropout_prob: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size

        # Combining the queries, keys and values in one matrix
        self.qkv = torch.nn.Linear(embed_size, 3*embed_size)
        self.projection = torch.nn.Linear(embed_size, embed_size)

        self.att_drop = torch.nn.Dropout(dropout_prob)
    

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Splitting keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), 'b n (h d qkv) -> (qkv) b h n d', h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        # Computing the attention matrix
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        scaling = self.embed_size**(1/2)
        att = torch.nn.functional.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        # Applying attention to scale the values
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)

        return out


class FeedForwardBlock(torch.nn.Sequential):
    '''
    Fully connected layer of the transformer encoder block
    '''

    def __init__(self, embed_size: int, scale: int, dropout_prob: float):
        super().__init__(
            torch.nn.Linear(embed_size, scale*embed_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(scale*embed_size, embed_size),
        )


class ResidualAdd(torch.nn.Module):
    '''
    Implements the residual connections of the transformer
    '''

    def __init__(self, fn: torch.nn.Sequential) -> None:
        super().__init__()
        self.fn = fn


    def forward(self, x: torch.tensor, **kwargs: dict[str, Any]) -> torch.tensor:
        res = x
        x = self.fn(x, **kwargs)
        x += res

        return x


class TransformerEncoderBlock(torch.nn.Sequential):

    def __init__(self, embed_size: int, num_heads: int, dropout_prob: float, forward_scale: int,
                 forward_dropout_prob: float, **kwargs: dict[str, Any]) -> None:
        super().__init__(
            ResidualAdd(torch.nn.Sequential(
                torch.nn.LayerNorm(embed_size),
                MultiHeadAttention(
                    embed_size, num_heads=num_heads, dropout_prob=dropout_prob
                ),
                torch.nn.Dropout(dropout_prob)
            )),
            ResidualAdd(torch.nn.Sequential(
                torch.nn.LayerNorm(embed_size),
                FeedForwardBlock(
                    embed_size, scale=forward_scale, dropout_prob=forward_dropout_prob
                ),
                torch.nn.Dropout(dropout_prob)
            ))
        )


class TransformerEncoder(torch.nn.Sequential):
    '''
    Implements the transfomer encoder
    '''

    def __init__(self, depth: int, **kwargs: dict[str, Any]) -> None:
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class TSViT(torch.nn.Module):

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.num_classes = int(config['num_classes'])
        self.img_res = int(config['img_res'])
        self.patch_size = int(config['patch_size'])
        self.embed_size = int(config['embed_size'])
        self.temporal_depth = int(config['temporal_depth'])
        self.spatial_depth = int(config['spatial_depth'])
        self.embed_dropout_prob = float(config['embed_dropout_prob'])

        self.num_patches = (self.img_res // self.patch_size)**2

        self.to_temporal_embedding = PatchEmbedding(**config)
        self.temporal_transformer = TransformerEncoder(depth=self.temporal_depth, **config)

        self.space_pos_embedding = torch.nn.Parameter(torch.randn(1, self.num_patches, self.embed_size))
        self.spatial_transformer = TransformerEncoder(depth=self.spatial_depth, **config)

        self.dropout = torch.nn.Dropout(self.embed_dropout_prob)

        self.mlp_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.embed_size),
            torch.nn.Linear(self.embed_size, self.patch_size**2)
        )


    def forward(self, x: torch.tensor) -> torch.tensor:
        assert len(x.shape) == 5
        _, _, _, H, W = x.shape

        # Temporal encoder
        x = self.to_temporal_embedding(x)
        x = self.temporal_transformer(x)

        # Spatial encoder
        x = x[:, :self.num_classes]

        x = rearrange(x, '(b p) n e -> (b n) p e', p=self.num_patches)
        x += self.space_pos_embedding

        x = self.dropout(x)
        x = self.spatial_transformer(x)

        # Segmentation head
        x = rearrange(x, 'bn p e -> (bn p) e')
        x = self.mlp_head(x)

        x = rearrange(x, '(b n p) hw -> b (p hw) n', n=self.num_classes, p=self.num_patches)
        x = rearrange(x, 'b (H W) n -> b n H W', H=H, W=W)

        return x


if __name__ == '__main__':
    config = {
        'img_res': 24,
        'num_channels': 11,
        'num_features': 16,
        'num_classes': 19,
        'ignore_background': False,
        'dropout_prob': 0.,
        'patch_size': 2,
        'embed_size': 128,
        'temporal_depth': 4,
        'spatial_depth': 4,
        'num_heads': 4,
        'pool': 'cls',
        'dim_head': 32,
        'embed_dropout_prob': 0.,
        'forward_dropout_prob': 0.,
        'forward_scale': 4
    }

    model = TSViT(config)
    batch_size = 16

    summary(model, input_size=(batch_size, 60, 11, 24, 24))