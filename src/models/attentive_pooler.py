# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import torch
import torch.nn as nn

from src.models.utils.modules import (
    Block,
    CrossAttention,
    CrossAttentionBlock
)
from src.utils.tensors import trunc_normal_
from src.models.utils.pos_embs import get_3d_sincos_pos_embed

class AttentivePooler(nn.Module):
    """ Attentive Pooler """
    def __init__(
        self,
        num_queries=1,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        complete_block=True
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim)) # Learnable query tokens for the attentive pooler

        self.complete_block = complete_block # If True, use CrossAttentionBlock, else use CrossAttention (without MLP) with learnable query tokens
        if complete_block:
            self.cross_attention_block = CrossAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio, # MLP ratio will input dim -> dim * mlp_ratio -> dim (back to original dim)
                qkv_bias=qkv_bias,
                norm_layer=norm_layer) # output shape: [B, N, embed_dim]
        else:
            self.cross_attention_block = CrossAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias)

        # Depth 1 is already handled by cross_attention_block, from depth 2 onwards, use Block (self-attention + mlp)
        self.blocks = None
        if depth > 1:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=False,
                    norm_layer=norm_layer)
                for i in range(depth-1)])

        self.init_std = init_std
        trunc_normal_(self.query_tokens, std=self.init_std)
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        if self.complete_block:
            rescale(self.cross_attention_block.xattn.proj.weight.data, 1)
            rescale(self.cross_attention_block.mlp.fc2.weight.data, 1)
        else:
            rescale(self.cross_attention_block.proj.weight.data, 1)
        if self.blocks is not None:
            for layer_id, layer in enumerate(self.blocks, 1):
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        q = self.query_tokens.repeat(len(x), 1, 1) # Repeat query tokens for each sample in the batch_size [1, num_queries, embed_dim] -> [B, num_queries, embed_dim]
        q = self.cross_attention_block(q, x)
        if self.blocks is not None:
            for blk in self.blocks:
                q = blk(q)
        return q # [B, N, embed_dim]


class AttentiveClassifier(nn.Module):
    """ Attentive Classifier """
    def __init__(
        self,
        embed_dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        depth=1,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        num_classes=1000,
        complete_block=True,
    ):
        super().__init__()
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            complete_block=complete_block,
        ) # output [B, N=1, embed_dim]
        self.linear = nn.Linear(embed_dim, num_classes, bias=True) # output [B, num_classes]

    def forward(self, x):
        x = self.pooler(x).squeeze(1) # [B, N=1, embed_dim] -> [B, embed_dim]
        x = self.linear(x) # [B, embed_dim] -> [B, num_classes]
        return x

class AttentiveSegmentator(nn.Module):
    """ Attentive Segmentator 
    Unlike, AttentiveClassifier, which employs AttentivePooler to reduce the number of output tokens to 1 (or as specified), we keep the number of output tokens same as input tokens.
    Thus, we are using Self-Attention instead of Cross-Attention. And this is the same as the MAE and VideoMAE models.
    
    Why using AttentiveSegmentator? From feature visualization AND V-JEPA paper, the feature maps from the last layer of the model are not very interpretable. Trusting that they are useful, we need more complex decoders to extract the information from the feature maps. Hence, we use the same architecture as MAE and VideoMAE to decode the feature maps to the desired output rather than using a simple linear layer as in DINOv2 or RADIO.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        encoder_embed_dim=768,
        decoder_embed_dim=768,
        depth=1,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=False,
        num_classes=100,
    ):
        super().__init__()

        self.input_size = img_size
        self.patch_size = patch_size

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        
        assert num_frames > 1, 'AttentiveSegmentator requires num_frames > 1'


        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        self.uniform_power = uniform_power
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.num_classes = num_classes

        self.num_patches = (
            (num_frames // tubelet_size)
            * (img_size // patch_size)
            * (img_size // patch_size)
        )
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim),
            requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.GELU,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, tubelet_size*patch_size**2 * num_classes, bias=True) # decoder to patch

        # ------ initialize weights
        self._init_pos_embed(self.pos_embed.data)  # sincos pos-embed
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed):
        embed_dim = pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size
        sincos = get_3d_sincos_pos_embed(
            embed_dim,
            grid_size,
            grid_depth,
            cls_token=False,
            uniform_power=self.uniform_power
        )
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.decoder_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def forward(self, x):
        """
        :param x: input image/video
        """

        B, N, D = x.shape

        assert D == self.encoder_embed_dim, f'Input feature dim {D} does not match model encoder dim {self.encoder_embed_dim}'

        x = self.decoder_embed(x)
        x = torch.nn.functional.gelu(x)
        # Tokenize input
        pos_embed = self.pos_embed
        # if pos_embed is not None:
        #     pos_embed = self.interpolate_pos_encoding(x, pos_embed)

        if pos_embed is not None:
            x += pos_embed

        # Fwd prop
        outs = []
        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x)

        x = self.norm(x)
        x = self.decoder_pred(x)

        return x
