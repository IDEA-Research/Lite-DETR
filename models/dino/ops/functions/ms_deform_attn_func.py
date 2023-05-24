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

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import math

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights,):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    # torch.stack().flatten(-2): N_*M_, D_, Lq_, P_ -> N_*M_, D_, Lq_, L_, P_ -> N_*M_, D_, Lq_, L_*P_
    # .sum(-1) L_*P_ 4*4=16 points add with attention weight
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()

def ms_deform_attn_core_pytorch_key_aware(query, value, key, input_padding_mask, value_spatial_shapes, sampling_locations, key_proj, value_proj, query_proj, attention_weights_linear, add):
    # for debug and test only,
    # need to use cuda version instead
    # N: batch szie; S_: total value num;   M_: head num 8; mD: 256/M (32)
    # Lq_: len q;  L_: num levels (4); P_: sample point per-level (4)
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    key_list = key.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    sampling_key_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        key_l_ = key_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)

        # N_*M_, D_, Lq_, P_
        sampling_key_l__ = F.grid_sample(key_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_key_list.append(sampling_key_l__)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    # import ipdb; ipdb.set_trace()
    key = torch.stack(sampling_key_list, dim=-2).flatten(-2)
    value = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    # N_*M_, D_, Lq_, P_ -> N_*M_, D_, Lq_, L_, P_ -> N_*M_, D_, Lq_, L_*P_

    # key = key.view(N_, M_, D_, Lq_, L_*P_).flatten(1, 2).permute(0, 2, 3, 1)
    # # N_, M_, D_, Lq_, L_*P_ -> N, Lq_, L_*P_, M_*D_
    #
    # key = key_proj(key)
    #
    # key = key.permute(0, 3, 1, 2).view(N_, M_, D_, Lq_, L_*P_).flatten(0, 1)
    # # N, Lq_, L_*P_, M_*D_ -> N_*M_, D_, Lq_, L_*P_
    #
    # key = key.permute(0, 2, 3, 1)  # N_*M_, D_, Lq_, L_*P_ -> N*M, Lq, L*P, D

    key = key.permute(0, 2, 3, 1).flatten(0, 1)   # N_*M_, D_, Lq_, L_*P_ -> N*M, Lq, L*P, D -> N*M*Lq, L*P, D

    N_, Lq, DD_ = query.shape
    query = query_proj(query)
    query = query.view(N_, Lq, M_, DD_ // M_)
    query = query.permute(0, 2, 1, 3).flatten(0, 2)  # N, Lq, M, D -> N, M, Lq, D -> N*M*Lq, D

    query = query.unsqueeze(-2)  # N*M*Lq, D-> N*M*Lq, 1, D
    dk = query.size()[-1]
    # import ipdb; ipdb.set_trace()
    # attention_weights = torch.matmul(key, query[..., None]).squeeze(-1)/ math.sqrt(dk)
    # attention_weights_linear = attention_weights_linear.transpose(1, 2).reshape(N_ * M_*Lq_, 1, L_ * P_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_)

    attention_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dk)
    # N*M*Lq, 1, D, x N*M*Lq, L*P, D   ==  N*M, Lq, 1,  L*P
    #
    # if add:
    #     attention_weights = attention_weights + attention_weights_linear
    # else:
    #     attention_weights = attention_weights * attention_weights_linear
    attention_weights = F.softmax(attention_weights, -1)

    # attention_weights = attention_weights.reshape(N_*M_, 1, Lq_, L_*P_)
    # attention_weights = attention_weights.transpose(1, 2)

    # value = value.view(N_, M_, D_, Lq_, L_ * P_).flatten(1, 2).permute(0, 2, 3, 1)
    # value = value_proj(value)
    # value = value.permute(0, 3, 1, 2).view(N_, M_, D_, Lq_, L_ * P_).flatten(0, 1)
    value = value.permute(0, 2, 3, 1).flatten(0, 1)  # N*M*Lq, L*P, D

    output = attention_weights.matmul(value)  # N*M, Lq, 1,  L*P x N*M*Lq, L*P, D -> N*M, Lq, 1,  D

    output = output.squeeze(-2).view(N_, M_, Lq_, D_).permute(0, 2, 1, 3)  # N*M, Lq, 1,  D -> N, Lq, M,  D

    output = output.flatten(2)

    # output = (value * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.contiguous()


def _reshape_to_batches(self, x):
    batch_size, seq_len, in_feature = x.size()
    sub_dim = in_feature // self.head_num
    return x.reshape(batch_size, seq_len, self.head_num, sub_dim) \
        .permute(0, 2, 1, 3) \
        .reshape(batch_size * self.head_num, seq_len, sub_dim)

def _reshape_from_batches(self, x):
    batch_size, seq_len, in_feature = x.size()
    batch_size //= self.head_num
    out_dim = in_feature * self.head_num
    return x.reshape(batch_size, self.head_num, seq_len, in_feature) \
        .permute(0, 2, 1, 3) \
        .reshape(batch_size, seq_len, out_dim)
