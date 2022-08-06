"""Euclidean Knowledge Graph embedding models where embeddings are in real space."""
import numpy as np
import torch
from torch import nn

from models.base import KGModel
from utils.euclidean import euc_sqdistance, givens_rotations, givens_reflection,euc_sqdistance_w,euc_sqdistance_r

EUC_MODELS = ["TransE", "CP", "MurE", "RotE", "RefE", "AttE"]
COMPLEX_MODELS = ["ComplEx", "RotatE"]


class BaseC(KGModel):
    """Complex Knowledge Graph Embedding models.

    Attributes:
        embeddings: complex embeddings for entities and relations
    """

    def __init__(self, args):
        """Initialize a Complex KGModel."""
        super(BaseC, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        assert self.rank % 2 == 0, "Complex models require even embedding dimension"
        self.rank = self.rank // 2

        self.entity.weight.data = self.init_size * self.entity.weight.to(self.data_type)
        self.rel.weight.data = self.init_size * self.rel.weight.to(self.data_type)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e = lhs_e[:, :self.rank], lhs_e[:, self.rank:]
        rhs_e = rhs_e[:, :self.rank], rhs_e[:, self.rank:]
        if eval_mode:
            return lhs_e[0] @ rhs_e[0].transpose(0, 1) + lhs_e[1] @ rhs_e[1].transpose(0, 1)
        else:
            return torch.sum(
                lhs_e[0] * rhs_e[0] + lhs_e[1] * rhs_e[1],
                1, keepdim=True
            )

    def get_complex_embeddings(self, queries):
        """Get complex embeddings of queries."""
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        rhs_e = self.entity(queries[:, 2])
        head_e = head_e[:, :self.rank], head_e[:, self.rank:]
        rel_e = rel_e[:, :self.rank], rel_e[:, self.rank:]
        rhs_e = rhs_e[:, :self.rank], rhs_e[:, self.rank:]
        return head_e, rel_e, rhs_e

    def get_factors(self, queries):
        """Compute factors for embeddings' regularization."""
        head_e, rel_e, rhs_e = self.get_complex_embeddings(queries)
        head_f = torch.sqrt(head_e[0] ** 2 + head_e[1] ** 2)
        rel_f = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        rhs_f = torch.sqrt(rhs_e[0] ** 2 + rhs_e[1] ** 2)
        return head_f, rel_f, rhs_f


class ComplEx(BaseC):
    """Simple complex model http://proceedings.mlr.press/v48/trouillon16.pdf"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        head_e, rel_e, _ = self.get_complex_embeddings(queries)
        lhs_e = torch.cat([
            head_e[0] * rel_e[0] - head_e[1] * rel_e[1],
            head_e[0] * rel_e[1] + head_e[1] * rel_e[0]
        ], 1)
        return lhs_e, self.bh(queries[:, 0])


class RotatE(BaseC):
    """Rotations in complex space https://openreview.net/pdf?id=HkgEQnRqYQ"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        head_e, rel_e, _ = self.get_complex_embeddings(queries)
        rel_norm = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        cos = rel_e[0] / rel_norm
        sin = rel_e[1] / rel_norm
        lhs_e = torch.cat([
            head_e[0] * cos - head_e[1] * sin,
            head_e[0] * sin + head_e[1] * cos
        ], 1)
        return lhs_e, self.bh(queries[:, 0])

class BaseE(KGModel):
    """Euclidean Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dist for distance and dot for dot product)
    """

    def __init__(self, args):
        super(BaseE, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        if self.sim == "dot":
            if eval_mode:
                score = lhs_e @ rhs_e.transpose(0, 1)
            else:
                score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
        else:
            score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
        return score


class TransE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.sim = "dist"

    def get_queries(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        lhs_e = head_e + rel_e
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class CP(BaseE):
    """Canonical tensor decomposition https://arxiv.org/pdf/1806.07297.pdf"""

    def __init__(self, args):
        super(CP, self).__init__(args)
        self.sim = "dot"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        return self.entity(queries[:, 0]) * self.rel(queries[:, 1]), self.bh(queries[:, 0])


class MurE(BaseE):
    """Diagonal scaling https://arxiv.org/pdf/1905.09791.pdf"""

    def __init__(self, args):
        super(MurE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = self.rel_diag(queries[:, 1]) * self.entity(queries[:, 0]) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class RotE(BaseE):
    """Euclidean 2x2 Givens rotations"""

    def __init__(self, args):
        super(RotE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = givens_rotations(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0])) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class RefE(BaseE):
    """Euclidean 2x2 Givens reflections"""

    def __init__(self, args):
        super(RefE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs = givens_reflection(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0]))
        rel = self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs + rel, lhs_biases


class AttE(BaseE):
    """Euclidean attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttE, self).__init__(args)
        self.sim = "dist"

        # reflection
        self.ref = nn.Embedding(self.sizes[1], self.rank)
        self.ref.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # rotation
        self.rot = nn.Embedding(self.sizes[1], self.rank)
        self.rot.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0

        # attention
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.act = nn.Softmax(dim=1)
        self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_reflection_queries(self, queries):
        lhs_ref_e = givens_reflection(
            self.ref(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_ref_e

    def get_rotation_queries(self, queries):
        lhs_rot_e = givens_rotations(
            self.rot(queries[:, 1]), self.entity(queries[:, 0])
        )
        return lhs_rot_e

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs_ref_e = self.get_reflection_queries(queries).view((-1, 1, self.rank))
        lhs_rot_e = self.get_rotation_queries(queries).view((-1, 1, self.rank))

        # self-attention mechanism
        cands = torch.cat([lhs_ref_e, lhs_rot_e], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        lhs_e = torch.sum(att_weights * cands, dim=1) + self.rel(queries[:, 1])
        return lhs_e, self.bh(queries[:, 0])







