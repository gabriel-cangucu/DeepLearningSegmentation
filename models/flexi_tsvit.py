import torch
import numpy as np
import random
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from typing import Any

from utils.distributed_utils import get_current_device


class FlexiPatchEmbedding(torch.nn.Module):

    def __init__(self, num_channels: int, num_classes: int, img_res: int, patch_size: int,
                 embed_size: int, **kwargs: dict[str, Any]) -> None:
        super().__init__()

        self.img_res = img_res
        self.patch_size = patch_size

        self.pseudo_inverses = {}

        self.projection = torch.nn.Sequential(
            Rearrange('b t c h w -> (b t) c h w'),
            torch.nn.Conv2d(num_channels-1, embed_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('bt e p1 p2 -> bt (p1 p2) e')
        )

        self.to_temporal_embedding = torch.nn.Linear(365, embed_size)
        self.cls_tokens = torch.nn.Parameter(torch.randn(1, num_classes, embed_size))


    def forward(self, x: torch.Tensor, new_patch_size: int) -> torch.Tensor:
        b, t, _, _, _ = x.shape
        new_num_patches = (self.img_res // new_patch_size)**2

        # Splitting images and dates
        x_t = x[:, :, -1, 0, 0].to(torch.int64)
        x = x[:, :, :-1]

        # Creating patch embeddings
        x = self.resize_and_project(x, new_patch_size)
        x = rearrange(x, '(b t) p e -> b p t e', b=b)

        # Adding temporal embeddings
        x_t = torch.nn.functional.one_hot(x_t, num_classes=365)
        x_t = x_t.to(torch.float32).reshape(-1, 365)

        temporal_pos = self.to_temporal_embedding(x_t)
        temporal_pos = rearrange(temporal_pos, '(b t) e -> b 1 t e', b=b)

        x += temporal_pos

        # Prepending the class tokens
        x = rearrange(x, 'b p t e -> (b p) t e')
        cls_tokens = repeat(self.cls_tokens, '() n e -> (b p) n e', b=b, p=new_num_patches)

        x = torch.cat((cls_tokens, x), dim=1)

        return x
    

    def _resize(self, x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        x_resized = torch.nn.functional.interpolate(
            x[None, None, ...],
            shape,
            mode='bicubic',
            antialias=True,
        )

        return x_resized[0, 0, ...]
    

    def _calculate_pseudo_inverse(self, old_shape: tuple[int, int], new_shape: tuple[int, int]) -> torch.Tensor:
        matrix = []

        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0

            matrix.append(self._resize(basis_vec, new_shape).reshape(-1))

        resized_matrix = torch.stack(matrix)

        return torch.linalg.pinv(resized_matrix)
    

    def _resize_embeddings(self, patch_embed: torch.Tensor, new_patch_size: int) -> torch.Tensor:
        if new_patch_size not in self.pseudo_inverses.keys():
            old_shape = (self.patch_size, self.patch_size)
            new_shape = (new_patch_size, new_patch_size)

            self.pseudo_inverses[new_patch_size] = self._calculate_pseudo_inverse(old_shape, new_shape)

        device = get_current_device()
        p_inv = self.pseudo_inverses[new_patch_size].to(device)

        def resample_patch_embed(patch_embed: torch.Tensor) -> torch.Tensor:
            resampled_kernel = p_inv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, '(h w) -> h w', h=new_patch_size, w=new_patch_size)

        patch_embed_map = torch.vmap(torch.vmap(resample_patch_embed, 0, 0), 1, 1)

        return patch_embed_map(patch_embed)
    

    def resize_and_project(self, x: torch.Tensor, new_patch_size: int) -> torch.Tensor:
        if new_patch_size == self.patch_size:
            weight = self.projection[1].weight
        else:
            weight = self._resize_embeddings(self.projection[1].weight, new_patch_size)

        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = torch.nn.functional.conv2d(x, weight=weight, bias=self.projection[1].bias, stride=new_patch_size)
        x = rearrange(x, 'bt e p1 p2 -> bt (p1 p2) e')

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


    def forward(self, x: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
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


class FlexiTSViT(torch.nn.Module):

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        self.num_classes = int(config['num_classes'])
        self.img_res = int(config['img_res'])
        self.patch_size = int(config['patch_size'])
        self.embed_size = int(config['embed_size'])
        self.temporal_depth = int(config['temporal_depth'])
        self.spatial_depth = int(config['spatial_depth'])
        self.embed_dropout_prob = float(config['embed_dropout_prob'])

        self.pseudo_inverses = {}

        self.num_patches = (self.img_res // self.patch_size)**2

        self.to_temporal_embedding = FlexiPatchEmbedding(**config)
        self.temporal_transformer = TransformerEncoder(depth=self.temporal_depth, **config)

        self.space_pos_embedding = torch.nn.Parameter(torch.randn(1, self.num_patches, self.embed_size))
        self.spatial_transformer = TransformerEncoder(depth=self.spatial_depth, **config)

        self.dropout = torch.nn.Dropout(self.embed_dropout_prob)

        self.mlp_head = torch.nn.Sequential(
            torch.nn.LayerNorm(self.embed_size),
            torch.nn.Linear(self.embed_size, self.patch_size**2)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 5
        _, _, _, H, W = x.shape

        new_patch_size = random.choice([2, 4, 6, 8, 12])
        new_num_patches = (self.img_res // new_patch_size)**2

        # Temporal encoder
        x = self.to_temporal_embedding(x, new_patch_size)
        x = self.temporal_transformer(x)

        # Spatial encoder
        x = x[:, :self.num_classes]

        x = rearrange(x, '(b p) n e -> (b n) p e', p=new_num_patches)
        x += self.resize_pos_embed(self.space_pos_embedding, new_patch_size)

        x = self.dropout(x)
        x = self.spatial_transformer(x)

        # Segmentation head
        x = rearrange(x, 'bn p e -> (bn p) e')
        x = self.resize_and_classify(x, new_patch_size)

        x = rearrange(x, '(b n p) hw -> b (p hw) n', n=self.num_classes, p=new_num_patches)
        x = rearrange(x, 'b (H W) n -> b n H W', H=H, W=W)

        return x
    

    def resize_pos_embed(self, pos_embed: torch.Tensor, new_size: int) -> torch.Tensor:
        if self.patch_size == new_size:
            return pos_embed

        pos_embed = rearrange(pos_embed, '1 (n1 n2) e -> 1 e n1 n2', n1=(self.img_res // self.patch_size))
        pos_embed = torch.nn.functional.interpolate(pos_embed, size=(self.img_res // new_size),
                                                    mode='bicubic', antialias=True)
        pos_embed = rearrange(pos_embed, '1 e n1 n2 -> 1 (n1 n2) e')

        return pos_embed


    def _resize(self, x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        x_resized = torch.nn.functional.interpolate(
            x[None, None, ...],
            shape,
            mode='bicubic',
            antialias=True,
        )

        return x_resized[0, 0, ...]
    

    def _calculate_pseudo_inverse(self, old_shape: tuple[int, int], new_shape: tuple[int, int]) -> torch.Tensor:
        matrix = []

        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0

            matrix.append(self._resize(basis_vec, new_shape).reshape(-1))

        resized_matrix = torch.stack(matrix)

        return torch.linalg.pinv(resized_matrix)
    

    def _resize_params(self, weight: torch.Tensor, bias: torch.Tensor, new_patch_size: int) -> torch.Tensor:
        if new_patch_size not in self.pseudo_inverses.keys():
            old_shape = (self.patch_size, self.patch_size)
            new_shape = (new_patch_size, new_patch_size)

            self.pseudo_inverses[new_patch_size] = self._calculate_pseudo_inverse(old_shape, new_shape)

        device = get_current_device()
        p_inv = self.pseudo_inverses[new_patch_size].to(device)

        def resample_embedding(embedding: torch.Tensor) -> torch.Tensor:
            return p_inv @ embedding
            
        resample_weight_map = torch.vmap(resample_embedding, 1, 1)
        weight = resample_weight_map(weight)
        bias = resample_embedding(bias)

        return weight, bias


    def resize_and_classify(self, x: torch.Tensor, new_patch_size: int) -> torch.Tensor:
        if new_patch_size == self.patch_size:
            weight = self.mlp_head[1].weight
            bias = self.mlp_head[1].bias
        else:
            weight, bias = self._resize_params(self.mlp_head[1].weight, self.mlp_head[1].bias,
                                               new_patch_size)

        x = torch.nn.functional.layer_norm(x, normalized_shape=[self.embed_size],
                                           weight=self.mlp_head[0].weight,
                                           bias=self.mlp_head[0].bias)
        x = torch.nn.functional.linear(x, weight=weight, bias=bias)

        return x


# if __name__ == '__main__':
#     config = {
#         'img_res': 24,
#         'num_channels': 11,
#         'num_features': 16,
#         'num_classes': 19,
#         'ignore_background': False,
#         'dropout_prob': 0.,
#         'patch_size': 6,
#         'embed_size': 128,
#         'temporal_depth': 4,
#         'spatial_depth': 4,
#         'num_heads': 4,
#         'pool': 'cls',
#         'dim_head': 32,
#         'embed_dropout_prob': 0.,
#         'forward_dropout_prob': 0.,
#         'forward_scale': 4
#     }

#     model = FlexiTSViT(config)
#     batch_size = 16

#     # summary(model, input_size=(batch_size, 60, 11, 24, 24))

#     x = torch.zeros(batch_size, 60, 11, 24, 24)
#     x = model(x)

#     print(x.shape)