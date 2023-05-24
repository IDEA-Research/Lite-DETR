# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction
from ..functions import ms_deform_attn_core_pytorch
from ..functions import ms_deform_attn_core_pytorch_key_aware
import torch.utils.checkpoint as checkpoint


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, use_pytorch_version=False, value_proj_after=False, key_aware=True, add=True, proj_key=True, deformable_use_checkpoint=False, same_loc=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")
        self.use_pytorch_version = use_pytorch_version
        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.same_loc = same_loc

        if not same_loc:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        else:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        if not key_aware:
            self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.proj_key = proj_key
        if proj_key:
            self.key_proj = nn.Linear(d_model, d_model)
        else:
            self.key_proj = None
        # self.key_proj = None
        self.query_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self.key_aware = key_aware
        self.add = add
        self.deformable_use_checkpoint = deformable_use_checkpoint

        print("use_pytorch_version key_aware, add， same_loc", use_pytorch_version, key_aware, add, same_loc)
        self.value_proj_after = value_proj_after

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        if self.same_loc:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)
            for i in range(self.n_points):
                grid_init[:, i, :] *= i + 1
        else:
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1,
                                                                                                                  self.n_levels,
                                                                                                                  self.n_points,
                                                                                                                  1)
            for i in range(self.n_points):
                grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        if not self.key_aware:
            constant_(self.attention_weights.weight.data, 0.)
            constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)

        xavier_uniform_(self.query_proj.weight.data)
        constant_(self.query_proj.bias.data, 0.)
        if self.proj_key:
            xavier_uniform_(self.key_proj.weight.data)
            constant_(self.key_proj.bias.data, 0.)

        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        if not self.value_proj_after:
            value = self.value_proj(input_flatten)
        # if key is None:
        key = input_flatten
        if self.proj_key:
            key = self.key_proj(key)
        else:
            key = value
        # value = input_flatten
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
            key = key.masked_fill(input_padding_mask[..., None], float(0))
        key = key.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        if not self.same_loc:
            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        else:
            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_points, 2)
            sampling_offsets = sampling_offsets[:, :, :, None].repeat(1, 1, 1, self.n_levels, 1, 1)
        attention_weights = None
        if not self.key_aware:
            attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
            attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        if self.key_aware:
            if not self.deformable_use_checkpoint:
                output = ms_deform_attn_core_pytorch_key_aware(
                    query, value, key, input_padding_mask, input_spatial_shapes, sampling_locations, self.key_proj, self.value_proj, self.query_proj, attention_weights, self.add
                )
            else:
                output = checkpoint.checkpoint(ms_deform_attn_core_pytorch_key_aware, query, value, key, input_padding_mask,
                                               input_spatial_shapes, sampling_locations, self.key_proj, self.value_proj,
                                               self.query_proj, attention_weights, self.add)
        elif self.use_pytorch_version:
            output = ms_deform_attn_core_pytorch(
                value, input_spatial_shapes, sampling_locations, attention_weights,
            )
        else:
            output = MSDeformAttnFunction.apply(
                value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        if self.value_proj_after:
            output = self.value_proj(output)
        output = self.output_proj(output)
        return output
