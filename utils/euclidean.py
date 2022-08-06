"""Euclidean operations utils functions."""

import torch


def euc_sqdistance(x, y, eval_mode=False):
    """Compute euclidean squared distance between tensors.

    Args:
        x: torch.Tensor of shape (N1 x d)
        y: torch.Tensor of shape (N2 x d)
        eval_mode: boolean

    Returns:
        torch.Tensor of shape N1 x 1 with pairwise squared distances if eval_mode is false
        else torch.Tensor of shape N1 x N2 with all-pairs distances

    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    if eval_mode:
        y2 = y2.t()
        xy = x @ y.t()
    else:
        assert x.shape[0] == y.shape[0]
        xy = torch.sum(x * y, dim=-1, keepdim=True)
    return x2 + y2 - 2 * xy

def euc_sqdistance_t(a, b, t1, t2, r1, r2):
    x1 = torch.sum(a * a, dim=-1, keepdim=True) + torch.sum(b * b, dim=-1, keepdim=True).t()
    x2 = (r1 * r1) @ (t1 * t1).t() + (r2 * r2) @ (t2 * t2).t() - 2.0 * (r1 * r2) @ (t1 * t2).t()
    x3 = -2.0 * (a @ b.t())
    x4 = 2.0 * ((a * r1) @ t1.t()) - 2.0 * ((a * r2) @ t2.t()) - 2.0 * (r1 @ (b * t1).t()) + 2.0 * (r2 @ (b * t2).t())

    return x1+x2+x3+x4

def norm_r(x):
    givens = x.view((x.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    return givens.view((x.shape[0],-1))

def euc_sqdistance_r(x, y, r, eval_mode=False):
    """Compute euclidean squared distance between tensors.

    Args:
        x: torch.Tensor of shape (N1 x d)
        y: torch.Tensor of shape (N2 x d)
        eval_mode: boolean

    Returns:
        torch.Tensor of shape N1 x 1 with pairwise squared distances if eval_mode is false
        else torch.Tensor of shape N1 x N2 with all-pairs distances

    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = y * y #(N2 x d)
    r2 = r * r #(N1 x d)
    if eval_mode:
        y2 = r2 @ y2.t()
        xy = (x*r) @ y.t()
    else:
        assert x.shape[0] == y.shape[0]
        y2 = torch.sum(y2*r2,dim=-1,keepdim=True)
        xy = torch.sum(r * x * y, dim=-1, keepdim=True)
    return x2 + y2 - 2 * xy

def euc_sqdistance_w(x,y,w2,eval_mode=False):
    """Compute euclidean squared distance between tensors.

        Args:
            x: torch.Tensor of shape (N1 x d)
            y: torch.Tensor of shape (N2 x d)
            w2: torch.Tensor of shape (N1 x d)
            eval_mode: boolean

        Returns:
            torch.Tensor of shape N1 x 1 with pairwise squared distances if eval_mode is false
            else torch.Tensor of shape N1 x N2 with all-pairs distances

        """
    x2 = x * x
    y2 = y * y
    if eval_mode:
        x2 = torch.sum(x2 * w2,dim=-1,keepdim=True) #(N1,1)
        y2 = w2 @ y2.t() #(N1,N2)
        xy = (x*w2) @ y.t() #(N1,N2)
    else:
        assert x.shape[0] == y.shape[0]
        x2 = torch.sum(x2 * w2,dim=-1,keepdim=True)
        y2 = torch.sum(y2 * w2,dim=-1,keepdim=True)
        xy = torch.sum(x * y * w2, dim=-1, keepdim=True)
    return x2 + y2 - 2 * xy

def givens_rotations(r, x, r_mul=None):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    if r_mul is not None:
        givens = givens * r_mul.unsqueeze(-1)
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))


def givens_rotations_no_norm(r, x):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))


def givens_rotations_v2(r, x):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N1 x d), rotation parameters
        x: torch.Tensor of shape (N2 x d), points to rotate

    Returns:
        torch.Tensor os shape (N1 x N2 x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], 1, -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((1, x.shape[0], -1, 2))
    x_rot = givens[:, :, :, 0:1] * x + givens[:, :, :, 1:] * torch.cat((-x[:, :, :, 1:], x[:, :, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], x.shape[0], -1))


def givens_reflection(r, x):
    """Givens reflections.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to reflect

    Returns:
        torch.Tensor os shape (N x d) representing reflection of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True).clamp_min(1e-15)
    x = x.view((r.shape[0], -1, 2))
    x_ref = givens[:, :, 0:1] * torch.cat((x[:, :, 0:1], -x[:, :, 1:]), dim=-1) + givens[:, :, 1:] * torch.cat(
        (x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_ref.view((r.shape[0], -1))
