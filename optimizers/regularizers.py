# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn
from utils.euclidean import givens_rotations, euc_sqdistance

class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class F2(Regularizer):
    def __init__(self, args):
        super(F2, self).__init__()
        self.weight = args.weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(f ** 2)/f.shape[0]
        return norm

class N3(Regularizer):
    def __init__(self, args):
        super(N3, self).__init__()
        self.weight = args.weight

    def forward(self, factors):
        """Regularized complex embeddings https://arxiv.org/pdf/1806.07297.pdf"""
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(
                torch.abs(f) ** 3
            )/f.shape[0]
        return norm


class N3_N3(Regularizer):
    def __init__(self, args):
        super(N3_N3, self).__init__()
        self.weight = args.weight
        self.weight2 = args.weight2

    def forward(self, factors):
        """Regularized complex embeddings https://arxiv.org/pdf/1806.07297.pdf"""
        norm = 0
        for f in factors[0]:
            norm += self.weight * torch.sum(
                torch.abs(f) ** 3
            )/f.shape[0]
        for f in factors[1]:
            norm += self.weight2 * torch.sum(
                torch.abs(f) ** 3
            )/f.shape[0]
        return norm


class F2_F2(Regularizer):
    def __init__(self, args):
        super(F2_F2, self).__init__()
        self.weight = args.weight
        self.weight2 = args.weight2

    def forward(self, factors):
        """Regularized complex embeddings https://arxiv.org/pdf/1806.07297.pdf"""
        norm = 0
        for f in factors[0]:
            norm += self.weight * torch.sum(
                torch.abs(f) ** 2
            )/f.shape[0]
        for f in factors[1]:
            norm += self.weight2 * torch.sum(
                torch.abs(f) ** 2
            )/f.shape[0]
        return norm

class NN_NN(Regularizer):
    def __init__(self, args):
        super(NN_NN, self).__init__()
        self.weight = args.weight
        self.weight2 = args.weight2
        self.n = args.n

    def forward(self, factors):
        """Regularized complex embeddings https://arxiv.org/pdf/1806.07297.pdf"""
        norm = 0
        for f in factors[0]:
            norm += self.weight * torch.sum(
                torch.abs(f) ** self.n
            )/f.shape[0]
        for f in factors[1]:
            norm += self.weight2 * torch.sum(
                torch.abs(f) ** self.n
            )/f.shape[0]
        return norm


class MeanVar(Regularizer):
    def __init__(self, args):
        super(MeanVar, self).__init__()
        self.weight = args.weight
        self.weight2 = args.weight2

    def forward(self, factors):
        """Regularized complex embeddings https://arxiv.org/pdf/1806.07297.pdf"""
        norm = 0
        for f in factors:
            norm -= self.weight * torch.sum(torch.mean(f**2,dim=-1)) /f.shape[0]
            norm += self.weight2 * torch.sum(torch.var(f**2,dim=-1)) /f.shape[0]
        return norm

class Var(Regularizer):
    def __init__(self, args):
        super(Var, self).__init__()
        self.weight = args.weight
        self.weight2 = args.weight2
    
    def forward(self, factors):
        norm = 0
        for f in factors[1]:
            norm += self.weight2 * torch.var(f)
        return norm
    
    
class RelReg(Regularizer):
    def __init__(self, args):
        super(RelReg, self).__init__()
        self.weight = args.weight

    def forward(self, factors):
        rel_head, rel_tail, rel_diag, sym_rel, h, rel, t = factors
        rel_head_rot = givens_rotations(rel_diag, rel_head)
        rel_tail_rot = givens_rotations(rel_diag, rel_tail)
        norm1 = torch.sum((rel_head_rot + rel - rel_tail) ** 2 * sym_rel, dim=-1) + torch.sum(
                          (rel_tail_rot + rel - rel_head) ** 2 * sym_rel, dim=-1)
        norm2 = self.weight[0]*torch.sum(t**2 + h**2 + rel**2) \
              + self.weight[1]*torch.sum((h * rel)**2  + (t * rel)**2)
        return torch.mean(torch.log(1. + torch.exp(norm1))) * self.weight[-1] + self.weight[2] * norm2 / h.shape[0]


class DURA(Regularizer):
    def __init__(self, args):
        super(DURA, self).__init__()
        self.weight = args.weight

    def forward(self, factors):
        h, r, t = factors
        norm = 0
        norm += self.weight[0]*torch.sum(t**2 + h**2)
        norm += self.weight[1]*torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight[2] * norm / h.shape[0]



