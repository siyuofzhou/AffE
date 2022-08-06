from .batchnormse_models import *
from utils.euclidean import givens_rotations_v2, euc_sqdistance_t, norm_r, givens_rotations_no_norm


class AffE(BaseE):
    """Euclidean 2x2 Givens rotations"""

    def __init__(self, args):
        super(AffE, self).__init__(args)
        self.sim = "dist"
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_mul = nn.Embedding(self.sizes[1], self.rank)
        self.rel_mul.weight.data = torch.ones((self.sizes[1], self.rank), dtype=self.data_type)

        self.rel_sca = nn.Embedding(self.sizes[1], self.rank)
        self.rel_sca.weight.data = torch.ones((self.sizes[1], self.rank), dtype=self.data_type) * 2

        self.sca_down = nn.Parameter(torch.tensor([args.sca], dtype=self.data_type), requires_grad=False)
        self.regmul = nn.Parameter(torch.tensor([args.regmul], dtype=self.data_type), requires_grad=False)

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = givens_rotations(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0])) * self.rel_mul(queries[:, 1]) \
                + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        self.sca = 1.0 + self.sca_down * torch.tanh(self.rel_sca(queries[:, 1]))
        lhs_e = lhs_e * self.sca
        return lhs_e, lhs_biases

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            rhs_e = self.entity.weight
            bt = self.bt.weight
        else:
            rhs_e = self.entity(queries[:, 2]) * self.sca
            bt = self.bt(queries[:, 2])
        return rhs_e, bt

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        if eval_mode:
            score = - torch.sum(lhs_e ** 2, dim=-1, keepdim=True) + 2 * (self.sca * lhs_e) @ rhs_e.t() \
                    - (self.sca ** 2) @ (rhs_e ** 2).t()
        else:
            score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
        return score

    def get_factors(self, queries):
        return [self.rel_sca(queries[:, 1]), ], \
               [self.regmul - self.rel_mul(queries[:, 1]), ]
