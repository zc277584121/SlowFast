#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import numpy
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from slowfast.models.common import DropPath, Mlp


def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None):
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = (
        tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    )

    tensor = pool(tensor)

    thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
    L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  #  tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape

# def vis_attn(attn: torch.Tensor, out_shape: list):
#     attn = attn.to('cpu').detach().numpy()
#     # print('1111', attn.shape)
#     # ([1, 1, 12545, 197])
#     # (B, head_num, Nq + 1, Nkv + 1)
#     attn = attn[0][0]
#     # Nq + 1, Nkv + 1
#     # 取了第一个通道，第一个头
#
#     C = attn.shape[0]
#     # Nkv + 1
#     print('attn.shape=', attn.shape)
#     input_tensor = attn[:, 1:]  # (3136, 384) (THW, C)
#     # Nq + 1, Nkv
#
#     thw = input_tensor.shape[-1]
#     hw = thw // 4
#     h = w = int(math.sqrt(hw))
#     print('input_tensor.shape=', input_tensor.shape)
#     input_tensor = input_tensor.reshape(C, 4, h, w)  # (4, 28, 28, 384) (T, H, W, C)
#     print('2 input_tensor.shape =', input_tensor.shape)
#     # Nq + 1, 4, H, W
#
#     # print('input_tensor =', input_tensor.shape)
#     # input_tensor = input_tensor[1]  # (28, 28, 384)
#     # # H, W, Nkv + 1
#     # # 取了第2个时间片
#     input_tensor = numpy.sum(input_tensor, axis=1)  # (28, 28, 384)
#     # Nq + 1, H, W,
#     # 时间片求和
#
#     # print('input_tensor =', input_tensor.shape)
#
#     # input_tensor = torch.sum(input_tensor, dim=-1)
#     input_tensor = input_tensor[0, ...]  # (28, 28)
#     # 取了Nkv 维度上的第0个位置，即cls_token
#     # H, W
#
#     input_tensor = normalization(input_tensor)
#     cm = plt.get_cmap("viridis")
#     heatmap = cm(input_tensor)
#     heatmap = heatmap[:, :, :3]
#     # Convert (H, W, C) to (C, H, W)
#     # heatmap = torch.Tensor(heatmap).permute(2, 0, 1)
#     plt.imshow(heatmap)
#     plt.show()
#     # print('input_tensor =', input_tensor.shape)



def vis_attn(attn: torch.Tensor, out_shape: list):
    attn = attn.to('cpu').detach().numpy()
    # out_shape = [s.numel() for s in out_shape]
    # print('1111', attn.shape)
    # ([1, 1, 12545, 197])
    # (B, head_num, Nq + 1, Nkv + 1)
    head_num = attn.shape[1]

    # 取了第一个通道，第一个头
    for head_idx in range(head_num):
        one_head_attn = attn[0][head_idx]
        # Nq + 1, Nkv + 1

        C = one_head_attn.shape[-1]
        # Nkv + 1
        print('attn.shape=', one_head_attn.shape)
        input_tensor = one_head_attn[1:]  # (3136, 384) (THW, C)
        # Nq, Nkv + 1


        print('input_tensor.shape=', input_tensor.shape)
        input_tensor = input_tensor.reshape(*out_shape, C)  # (4, 28, 28, 384) (T, H, W, C)
        print('2 input_tensor.shape =', input_tensor.shape)
        # 4, H, W, Nkv + 1

        # print('input_tensor =', input_tensor.shape)
        input_tensor = input_tensor[1]  # (28, 28, 384)
        # # H, W, Nkv + 1
        # # 取了第2个时间片
        # input_tensor = numpy.sum(input_tensor, axis=0)  # (28, 28, 384)
        # H, W, Nkv + 1
        # 取了第2个时间片

        # print('input_tensor =', input_tensor.shape)

        # input_tensor = torch.sum(input_tensor, dim=-1)
        input_tensor = input_tensor[..., 0]  # (28, 28)
        # 取了Nkv 维度上的第0个位置，即cls_token
        # H, W

        input_tensor = normalization(input_tensor)
        cm = plt.get_cmap("viridis")
        heatmap = cm(input_tensor)
        heatmap = heatmap[:, :, :3]
        # Convert (H, W, C) to (C, H, W)
        # heatmap = torch.Tensor(heatmap).permute(2, 0, 1)
        plt.imshow(heatmap)
        # plt.show()
        name = '/home/zhangchen/zhangchen_workspace/SlowFast/output_test_imgs_time_2/head_' + str(head_idx) + '_out_shape' + str(out_shape) + str(random.randint(100000, 200000)) + '.png'
        print('name = ', name)
        plt.savefig(name)
        # print('input_tensor =', input_tensor.shape)



# def vis_attn(attn: torch.Tensor, out_shape: list):
#     attn = attn.to('cpu').detach()#.numpy()
#     # print('1111', attn.shape)
#     # ([1, 1, 12545, 197])
#     # (B, head_num, Nq + 1, Nkv + 1)
#     B, head_num, Nq, Nkv = attn.shape
#     attn = attn[0] # get first example in a batch
#     # (head_num, Nq + 1, Nkv + 1)
#     for head_index in range(head_num):
#         one_head_attn = attn[head_index]
#         # (Nq + 1, Nkv + 1)
#         # print('aaa', one_head_attn.shape)
#         one_head_attn = one_head_attn[1:, 1:]
#         # print('bbb', one_head_attn.shape)
#         # (Nq, Nkv)
#         one_head_attn = one_head_attn[0]
#         # (Nkv)
#
#         Nkv = one_head_attn.shape[0] # t * h * w, t=4
#         hw = Nkv / 4
#         h = w = int(math.sqrt(hw))
#         one_head_attn = one_head_attn.reshape(4, h, w)
#         # 4, h, w
#         print('h = ', h)
#         print('w = ', w)
#         one_head_attn = one_head_attn[0]
#         one_head_attn = one_head_attn.numpy()
#         # Get the color map by name.
#         cm = plt.get_cmap("viridis")
#
#         one_head_attn = normalization(one_head_attn)
#         print('one_head_attn.shape=', one_head_attn.shape)
#         # print('one_head_attn.max()=', one_head_attn.max())
#         # print('one_head_attn.min()=', one_head_attn.min())
#         heatmap = cm(one_head_attn)
#         heatmap = heatmap[:, :, :3]
#         # Convert (H, W, C) to (C, H, W)
#         # heatmap = torch.Tensor(heatmap).permute(2, 0, 1)
#         plt.imshow(heatmap)
#         plt.show()

def normalization(data):
    _range = numpy.max(data) - numpy.min(data)
    return (data - numpy.min(data)) / _range


def vis_attn_by_qk(q, k, scale):
    q = q.to('cpu').detach()#.numpy()
    k = k.to('cpu').detach()#.numpy()
    # 1, 2, 12545, 96
    # (B, num_heads, Nq + 1, C//num_heads)
    q = q[0]  # get first example in a batch
    # (num_heads, Nq + 1, C // num_heads)
    k = k[0]  # get first example in a batch
    # (num_heads, Nkv + 1, C // num_heads)
    head_num = k.shape[0]
    # (head_num, Nq + 1, Nkv + 1)
    for head_index in range(head_num):
        one_head_q = q[head_index]
        one_head_k = k[head_index]

        one_head_q = one_head_q[1:]
        # Nq, C // num_heads
        one_head_k = one_head_k[1:]
        # Nkv, C // num_heads

        Nq = one_head_q.shape[0]
        one_head_q = one_head_q.reshape(4, Nq // 4, -1)
        # 4, Nq // 4, C // num_heads        Nq // 4 = HW
        Nkv = one_head_k.shape[0]
        one_head_k = one_head_k.reshape(4, Nkv // 4, -1)
        # 4, Nkv // 4, C // num_heads       Nkv // 4 = HW


        one_head_q = one_head_q[0]
        # HW, C // num_heads
        one_head_k = one_head_k[0]
        # HW, C // num_heads

        print('one_head_q.shape=', one_head_q.shape)
        print('one_head_k.shape=', one_head_k.shape)
        print('scale = ', scale)
        print('one_head_q @ one_head_k.transpose(-2, -1)).shape', (one_head_q @ one_head_k.transpose(-2, -1)).shape)
        one_head_attn = (one_head_q @ one_head_k.transpose(-2, -1)) * scale
        print('one_head_attn.shape=', one_head_attn.shape)
        # HW * HW
        # print('3 one_head_attn.shape =', one_head_attn.shape)
        one_head_attn = one_head_attn.softmax(dim=-1)

        # one_head_attn = one_head_attn[0] #todo
        # one_head_attn = one_head_attn[:, 0] #todo
        one_head_attn = torch.diagonal(one_head_attn) #todo
        # HW
        h = w = int(math.sqrt(one_head_attn.shape[0]))
        print('h = ', h)
        print('w = ', w)
        one_head_attn = one_head_attn.reshape(h, w)
        cm = plt.get_cmap("viridis")
        one_head_attn = normalization(one_head_attn.numpy())
        heatmap = cm(one_head_attn)
        heatmap = heatmap[:, :, :3]
        # Convert (H, W, C) to (C, H, W)
        # heatmap = torch.Tensor(heatmap).permute(2, 0, 1)
        plt.imshow(heatmap)
        plt.show()

class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.show_attn = nn.Softmax(dim=-1)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()

        if mode == "avg":
            self.pool_q = (
                nn.AvgPool3d(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                nn.AvgPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                nn.AvgPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "max":
            self.pool_q = (
                nn.MaxPool3d(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                nn.MaxPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                nn.MaxPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv":
            self.pool_q = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(head_dim) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(head_dim) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(head_dim) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

    def forward(self, x, thw_shape):
        B, N, C = x.shape

        qkv = (
            # (B, N, C)
            self.qkv(x)
            # (B, N, C*3)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            # (B, N, 3, num_heads, C//num_heads)
            .permute(2, 0, 3, 1, 4)
            # (3, B, num_heads, N, C//num_heads)
        )
        # print('self.num_heads =', self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # (B, num_heads, N, C//num_heads)
        # print('q.shape =', q.shape)
        # print('k.shape =', k.shape)
        # print('v.shape =', v.shape)
        # vis_attn_by_qk(q, k, self.scale)
        q, out_shape = attention_pool(
            q,
            self.pool_q,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, _ = attention_pool(
            k,
            self.pool_k,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, _ = attention_pool(
            v,
            self.pool_v,
            thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )
        
        # print('2 q.shape =', q.shape)
        # print('2 k.shape =', k.shape)
        # print('2 v.shape =', v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # print('3 attn.shape =', attn.shape)
        # attn = attn.softmax(dim=-1)
        attn = self.show_attn(attn)
        # print('4 attn.shape =', attn.shape)
        # print('attn.shape=', attn.shape)
        N = q.shape[2]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)
        # print('out_shape=', out_shape)
        # vis_attn(attn, out_shape) #todo

        return x, out_shape


class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        kernel_skip=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        stride_skip=(1, 1, 1),
        mode="conv",
        has_cls_embed=True,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=nn.LayerNorm,
            has_cls_embed=has_cls_embed,
            mode=mode,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool_skip = (
            nn.MaxPool3d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
            if len(kernel_skip) > 0
            else None
        )

    def forward(self, x, thw_shape):
        x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape)
        x_res, _ = attention_pool(
            x, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed
        )
        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new
