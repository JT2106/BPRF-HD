import math
from typing import Optional, Callable, Iterable, Tuple, Any, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
import torchhd.functional as functional


def binarize_hdrp(random_projection, device):
    shape = (random_projection.out_features, random_projection.in_features)
    total = shape[0] * shape[1]

    # get 0/1/2 and map to -1/0/+1
    probs = torch.tensor([0.33, 0.67, 0.33], device=device)
    idx = torch.multinomial(probs, num_samples=total, replacement=True) # [0,1,2]'s index

    sparse_param = idx.float() - 1.0  # 0→-1, 1→0, 2→+1
    sparse_param = sparse_param.view(shape)
    sparse_param = Parameter(sparse_param.to(device), requires_grad=False)
    random_projection.weight = sparse_param

    return random_projection

def HD_encoder(samples, random_projection):
        return random_projection(samples).sign()


class BinLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, init_weight: Tensor=None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BinLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        if init_weight is not None:
            self.weight = Parameter((init_weight*0.001).to(**factory_kwargs), requires_grad=True)
        else:
            self.weight = Parameter((torch.rand((out_features, in_features)) * 2 - 1) * 0.001, requires_grad=True)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        real_weights = self.weight
        scaling_factor = torch.mean(abs(real_weights))
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        clipped_weights = torch.clamp(real_weights, -1.0, 1.0)
        bin_weight = binary_weights_no_grad.detach() - clipped_weights.detach() + clipped_weights
        associate_memory = torch.sign(bin_weight).detach()

        return F.linear(input, bin_weight, bias=None), associate_memory


class LeHDC(nn.Module):
    def __init__(self, n_dimensions, n_classes, dropout=0.1):
        super(LeHDC, self).__init__()
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes

        self.bnn_classifier = BinLinear(n_dimensions, n_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor):
        query_vector = x
        x = self.dropout(x)
        x, associate_memory = self.bnn_classifier(x)

        return x, query_vector, associate_memory


class BipropLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, init_weight: Tensor=None, dim_reduction=0.2,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(BipropLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dim_reduction = dim_reduction

        if init_weight is not None:
            self.weight = Parameter(init_weight.to(**factory_kwargs), requires_grad=False)
        else:
            self.weight = Parameter((torch.rand((out_features, in_features)) * 2 - 1) * 0.001, requires_grad=False)

        self.score = Parameter((torch.rand((out_features, in_features)) * 2 - 1) * 0.001, requires_grad=True)

    def forward(self, sample):
        # scores: (d_out, d_in)
        d_out, d_in = self.score.shape

        # mean score for each column
        col_mean = self.score.mean(dim=0)

        # dimension reduction by col_mean
        _, idx = col_mean.sort()
        keep_cols = math.ceil((1 - self.dim_reduction) * d_in)
        drop_idx = idx[:d_in - keep_cols]  # the index of the collum to be set to 0

        # get mask
        mask_no_grad = torch.ones_like(self.score)
        mask_no_grad[:, drop_idx] = 0
        mask = mask_no_grad.detach() - self.score.detach() + self.score

        # get scaling factor
        abs_w = torch.abs(self.weight)
        q_w = abs_w * mask
        scaling_factor = q_w.sum() / mask.sum()  # L1 norm

        associate_memory = torch.sign(self.weight) * mask

        if self.weight.requires_grad:
            bin_weight_no_grad = scaling_factor * associate_memory
            clipped_weights = torch.clamp(self.weight, -1.0, 1.0)
            bin_weight = bin_weight_no_grad.detach() - clipped_weights.detach() + clipped_weights
        else:
            bin_weight = scaling_factor * associate_memory

        return F.linear(sample, bin_weight, bias=None), associate_memory, mask[0, :]


class BipropHD(nn.Module):
    def __init__(self, n_dimensions, n_classes, init_am, dim_reduction, dropout=0.0):
        super(BipropHD, self).__init__()
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes

        self.bp_classifier = BipropLinear(n_dimensions, n_classes, init_weight=init_am, dim_reduction=dim_reduction)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor):
        query_vector = x
        x = self.dropout(x)
        output, associate_memory, mask_dim = self.bp_classifier(x)
        query_vector = query_vector * mask_dim

        return output, query_vector, associate_memory, mask_dim


class BPRFHD(nn.Module):
    def __init__(self, hd_full_dim, n_classes, mask_dim, full_dim_am: Tensor=None, dropout=0.1):
        super(BPRFHD, self).__init__()
        self.hd_full_dim = hd_full_dim
        self.n_classes = n_classes

        valid_dim = mask_dim.sum()
        self.valid_dim = valid_dim

        valid_dim_mask = mask_dim.bool()
        self.valid_dim_mask = valid_dim_mask

        if full_dim_am is not None:
            valid_am = Parameter(full_dim_am[:, valid_dim_mask], requires_grad=True)
        else:
            valid_am = Parameter((torch.rand((n_classes, valid_dim)) * 2 - 1) * 0.001, requires_grad=True)

        self.refine_classifier = BinLinear(valid_dim, n_classes, init_weight=valid_am)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor):
        query_vector = x[:, self.valid_dim_mask]
        x = self.dropout(query_vector)
        x, associate_memory = self.refine_classifier(x)

        return x, query_vector, associate_memory


