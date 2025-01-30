import torch
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from typing import Any


class PatchEmbedding(torch.nn.Module):
    '''
    Turns 2D images with shape (B x C x H x W) into a sequence of flattened 2D patches
    '''

    def __init__(self, num_channels: int=3, patch_size: int=16, img_res: int=224, embed_size: int=768) -> None:
        super().__init__()

        self.projection = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, embed_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )
        self.cls_token = torch.nn.Parameter(torch.rand(1, 1, embed_size))
        self.positions = torch.nn.Parameter(torch.randn((img_res // patch_size)**2 + 1, embed_size))
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 4

        b, _, _, _ = x.shape

        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)

        # Prepending the class token to the input
        x = torch.cat([cls_tokens, x], dim=1)

        # Adding the positional embedding
        x += self.positions

        return x


class MultiHeadAttention(torch.nn.Module):
    '''
    Computes the multi-head attention matrix using queries and values and uses it to "attend" to the values
    '''

    def __init__(self, embed_size: int=768, num_heads: int=8, dropout_prob: float=0.) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size

        # Combining the queries, keys and values in one matrix
        self.qkv = torch.nn.Linear(embed_size, 3*embed_size)
        self.projection = torch.nn.Linear(embed_size, embed_size)

        self.att_drop = torch.nn.Dropout(dropout_prob)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class ResidualAdd(torch.nn.Module):
    '''
    Implements the residual connections of the transformer
    '''

    def __init__(self, fn: torch.nn.Sequential) -> None:
        super().__init__()
        self.fn = fn


    def forward(self, x: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        res = x
        x = self.fn(x, **kwargs)
        x += res

        return x


class FeedForwardBlock(torch.nn.Sequential):
    '''
    Fully connected layer of the transformer encoder block
    '''

    def __init__(self, embed_size: int, scale: int=4, dropout_prob: float=0.):
        super().__init__(
            torch.nn.Linear(embed_size, scale*embed_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_prob),
            torch.nn.Linear(scale*embed_size, embed_size),
        )


class TransformerEncoderBlock(torch.nn.Sequential):
    '''
    Implements a single transformer encoder block
    '''

    def __init__(self, embed_size:int=768, dropout_prob: float=0., forward_scale: int=4,
                 forward_dropout_prob: float=0., **kwargs) -> None:
        super().__init__(
            ResidualAdd(torch.nn.Sequential(
                torch.nn.LayerNorm(embed_size),
                MultiHeadAttention(embed_size, **kwargs),
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

    def __init__(self, depth: int=12, **kwargs) -> None:
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ViT(torch.nn.Sequential):

    def __init__(self, config: dict[str, Any]) -> None:
        num_channels = int(config['num_channels'])
        patch_size = int(config['patch_size'])
        img_res = int(config['img_res'])
        embed_size = int(config['embed_size'])
        depth = int(config['depth'])

        kwargs = {
            'embed_size': embed_size,
            'num_heads': int(config['num_heads']),
            'dropout_prob': float(config['dropout_prob']),
            'forward_scale': int(config['forward_scale']),
            'forward_dropout_prob': float(config['forward_dropout_prob'])
        }

        super().__init__(
            PatchEmbedding(num_channels, patch_size, img_res, embed_size),
            TransformerEncoder(depth, **kwargs)
        )


# if __name__ == '__main__':
#     '''
#     APAGAR DEPOIS!!!
#     '''

#     x = torch.randn(16, 3, 24, 24)

#     config = {
#         'num_channels': 3,
#         'patch_size': 2,
#         'img_res': 24,
#         'embed_size': 128,
#         'depth': 4,
#         'num_heads': 4,
#         'dropout_prob': 0.,
#         'forward_scale': 4,
#         'forward_dropout_prob': 0.
#     }

#     transformer = ViT(config)
#     print(transformer(x).shape)