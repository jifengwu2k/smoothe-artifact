"""
# This file is mainly borrowed from SciPy Pade approximation for matrix exponential:
# - https://github.com/scipy/scipy/blob/main/scipy/sparse/linalg/_expm_multiply.py
# - https://github.com/scipy/scipy/blob/main/scipy/sparse/linalg/_matfuncs.py

"""

import scipy
import torch
import numpy as np
import scipy.sparse.linalg as splin
from torch_sparse import SparseTensor, t
import torch.sparse as sp


def density(A):
    return A._nnz() / (A.shape[0] * A.shape[1])


def sparse_coo_pytorch2np(A):
    A = A.coalesce()
    indices_np = A.indices().cpu().numpy()
    values_np = A.values().cpu().numpy()
    size_np = A.size()

    coo_matrix = scipy.sparse.coo_matrix((values_np, indices_np),
                                         shape=size_np)
    return coo_matrix.tocsr()


def sparse_np2pytorch(coo_matrix):
    # Extract the indices and values from the COO matrix
    if coo_matrix.format != "coo":
        coo_matrix = coo_matrix.tocoo()
    indices = np.vstack((coo_matrix.row, coo_matrix.col))
    values = coo_matrix.data

    # Convert indices and values to PyTorch tensors
    indices_tensor = torch.LongTensor(indices)
    values_tensor = torch.FloatTensor(values)

    # Create a PyTorch sparse tensor
    size_tensor = torch.Size(coo_matrix.shape)
    sparse_tensor = torch.sparse_coo_tensor(indices_tensor, values_tensor,
                                            size_tensor)

    return sparse_tensor


# sparse version of spsolve
# turns out to be inefficient bc P, Q are not sparse enough
# def spsolve(P, Q):
#     np_P = sparse_coo_pytorch2np(P).tocsc()
#     np_Q = sparse_coo_pytorch2np(Q).tocsc()
#     np_ret = scipy.sparse.linalg.spsolve(np_P, np_Q)
#     ret = sparse_np2pytorch(np_ret).to(P.device)
#     return ret


def spsolve(P, Q):
    dense_P = P.to_dense()
    dense_Q = Q.to_dense()
    ret = torch.linalg.solve(dense_P, dense_Q)
    sparsity = torch.count_nonzero(ret) / ret.numel()
    if sparsity < 0.1:
        ret = ret.to_sparse()
    else:
        return ret


def sparse_trace(A):
    A = A.coalesce()
    row, col = A.indices()
    val = A.values()
    mask = row == col
    masked_val = val[mask].double()
    return (masked_val[masked_val > 1] - 1).sum()


def _smart_matrix_product(A, B, alpha=None):
    if density(A) > 0.01 or density(B) > 0.01:
        A = A.to_dense()
        B = B.to_dense()
        out = A @ B
        out = out.to_sparse()
    else:
        out = sp.mm(A, B)
    if alpha is not None:
        out = out * alpha
    return out


def _onenorm(A):
    # compute the one norm of sparse matrix A
    if A.layout == torch.strided:
        return A.abs().sum(dim=0).max().item()
    else:
        if A._nnz() == 0:
            return 0
        return A.abs().sum(dim=0).values().max().item()


def _onenormest_matrix_power(A, p):
    # compute the one norm of sparse matrix A to the power of p
    if density(A) > 0.01:
        A = A.to_dense()
    A_p = torch.linalg.matrix_power(A, p)
    return _onenorm(A_p)


def _onenormest_product(A, B):
    # compute the one norm of product of A and B
    if density(A) > 0.01 or density(B) > 0.01:
        A = A.to_dense()
        B = B.to_dense()
        A_B = A @ B
    else:
        A_B = sp.mm(A, B)
    return _onenorm(A_B)


def _onenorm_matrix_power_nnm(A, p):
    """
    Compute the 1-norm of a non-negative integer power of a non-negative matrix.

    Parameters
    ----------
    A : a square ndarray or matrix or sparse matrix
        Input matrix with non-negative entries.
    p : non-negative integer
        The power to which the matrix is to be raised.

    Returns
    -------
    out : float
        The 1-norm of the matrix power p of A.

    """
    # Check input
    if int(p) != p or p < 0:
        raise ValueError('expected non-negative integer p')
    p = int(p)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')

    # Explicitly make a column vector so that this works when A is a
    # numpy matrix (in addition to ndarray and sparse matrix).
    v = torch.ones((A.shape[0], 1), dtype=torch.float, device=A.device)
    M = A.T
    for i in range(p):
        # v = M.dot(v)
        v = sp.mm(M, v)
    return torch.max(v)


def _ell(A, m):
    """
    A helper function for expm_2009.

    Parameters
    ----------
    A : linear operator
        A linear operator whose norm of power we care about.
    m : int
        The power of the linear operator

    Returns
    -------
    value : int
        A value related to a bound.

    """
    # The c_i are explained in (2.2) and (2.6) of the 2005 expm paper.
    # They are coefficients of terms of a generating function series expansion.
    c_i = {
        3: 100800.,
        5: 10059033600.,
        7: 4487938430976000.,
        9: 5914384781877411840000.,
        13: 113250775606021113483283660800000000.
    }
    abs_c_recip = c_i[m]

    # This is explained after Eq. (1.2) of the 2009 expm paper.
    # It is the "unit roundoff" of IEEE double precision arithmetic.
    u = 2**-53

    # Compute the one-norm of matrix power p of abs(A).
    A_abs_onenorm = _onenorm_matrix_power_nnm(abs(A), 2 * m + 1)

    alpha = A_abs_onenorm / (_onenorm(A) * abs_c_recip)
    alpha = alpha.item()

    # Treat zero norm as a special case.
    if not alpha:
        return 0

    log2_alpha_div_u = np.log2(alpha / u)
    value = int(np.ceil(log2_alpha_div_u / (2 * m)))
    return max(value, 0)


def _solve_P_Q(U, V):
    """
    A helper function for expm_2009.

    Parameters
    ----------
    U : ndarray
        Pade numerator.
    V : ndarray
        Pade denominator.
    """
    P = U + V
    Q = -U + V
    return spsolve(Q, P)


class _ExpmPadeHelper:

    def __init__(self, A):
        """
    From
    Initialize the object.

    Parameters
    ----------
    A : a dense or sparse square numpy matrix or ndarray
        The matrix to be exponentiated.
    """
        self.A = A
        self._A2 = None
        self._A4 = None
        self._A6 = None
        self._A8 = None
        self._A10 = None
        self._d4_exact = None
        self._d6_exact = None
        self._d8_exact = None
        self._d10_exact = None
        self._d4_approx = None
        self._d6_approx = None
        self._d8_approx = None
        self._d10_approx = None
        self.ident = sp.spdiags(torch.ones(A.shape[0]),
                                torch.zeros(1, dtype=torch.long),
                                A.shape).to(A.device)

    @property
    def A2(self):
        if self._A2 is None:
            self._A2 = _smart_matrix_product(self.A, self.A)
        return self._A2

    @property
    def A4(self):
        if self._A4 is None:
            self._A4 = _smart_matrix_product(self.A2, self.A2)
        return self._A4

    @property
    def A6(self):
        if self._A6 is None:
            self._A6 = _smart_matrix_product(self.A4, self.A2)
        return self._A6

    @property
    def A8(self):
        if self._A8 is None:
            self._A8 = _smart_matrix_product(self.A6, self.A2)
        return self._A8

    @property
    def A10(self):
        if self._A10 is None:
            self._A10 = _smart_matrix_product(self.A4, self.A6)
        return self._A10

    @property
    def d4_tight(self):
        if self._d4_exact is None:
            self._d4_exact = _onenorm(self.A4)**(1 / 4.)
        return self._d4_exact

    @property
    def d6_tight(self):
        if self._d6_exact is None:
            self._d6_exact = _onenorm(self.A6)**(1 / 6.)
        return self._d6_exact

    @property
    def d8_tight(self):
        if self._d8_exact is None:
            self._d8_exact = _onenorm(self.A8)**(1 / 8.)
        return self._d8_exact

    @property
    def d10_tight(self):
        if self._d10_exact is None:
            self._d10_exact = _onenorm(self.A10)**(1 / 10.)
        return self._d10_exact

    @property
    def d4_loose(self):
        if self._d4_exact is not None:
            return self._d4_exact
        else:
            if self._d4_approx is None:
                self._d4_approx = _onenormest_matrix_power(
                    self.A2,
                    2,
                )**(1 / 4.)
            return self._d4_approx

    @property
    def d6_loose(self):
        if self._d6_exact is not None:
            return self._d6_exact
        else:
            if self._d6_approx is None:
                self._d6_approx = _onenormest_matrix_power(self.A2,
                                                           3)**(1 / 6.)
            return self._d6_approx

    @property
    def d8_loose(self):
        if self._d8_exact is not None:
            return self._d8_exact
        else:
            if self._d8_approx is None:
                self._d8_approx = _onenormest_matrix_power(self.A4,
                                                           2)**(1 / 8.)
            return self._d8_approx

    @property
    def d10_loose(self):
        if self._d10_exact is not None:
            return self._d10_exact
        else:
            if self._d10_approx is None:
                self._d10_approx = _onenormest_product(self.A4,
                                                       self.A6)**(1 / 10.)
            return self._d10_approx

    def pade3(self):
        b = (120., 60., 12., 1.)
        U = _smart_matrix_product(self.A, b[3] * self.A2 + b[1] * self.ident)
        V = b[2] * self.A2 + b[0] * self.ident
        return U, V

    def pade5(self):
        b = (30240., 15120., 3360., 420., 30., 1.)
        U = _smart_matrix_product(
            self.A, b[5] * self.A4 + b[3] * self.A2 + b[1] * self.ident)
        V = b[4] * self.A4 + b[2] * self.A2 + b[0] * self.ident
        return U, V

    def pade7(self):
        b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
        U = _smart_matrix_product(
            self.A, b[7] * self.A6 + b[5] * self.A4 + b[3] * self.A2 +
            b[1] * self.ident)
        V = b[6] * self.A6 + b[4] * self.A4 + b[2] * self.A2 + b[0] * self.ident
        return U, V

    def pade9(self):
        b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
             2162160., 110880., 3960., 90., 1.)
        U = _smart_matrix_product(
            self.A, (b[9] * self.A8 + b[7] * self.A6 + b[5] * self.A4 +
                     b[3] * self.A2 + b[1] * self.ident))
        V = (b[8] * self.A8 + b[6] * self.A6 + b[4] * self.A4 +
             b[2] * self.A2 + b[0] * self.ident)
        return U, V

    def pade13_scaled(self, s):
        b = (64764752532480000., 32382376266240000., 7771770303897600.,
             1187353796428800., 129060195264000., 10559470521600.,
             670442572800., 33522128640., 1323241920., 40840800., 960960.,
             16380., 182., 1.)
        B = self.A * 2**-s
        B2 = self.A2 * 2**(-2 * s)
        B4 = self.A4 * 2**(-4 * s)
        B6 = self.A6 * 2**(-6 * s)
        U2 = _smart_matrix_product(B6, b[13] * B6 + b[11] * B4 + b[9] * B2)
        U = _smart_matrix_product(
            B, (U2 + b[7] * B6 + b[5] * B4 + b[3] * B2 + b[1] * self.ident))
        V2 = _smart_matrix_product(B6, b[12] * B6 + b[10] * B4 + b[8] * B2)
        V = V2 + b[6] * B6 + b[4] * B4 + b[2] * B2 + b[0] * self.ident

        return U, V


class _ExpmPadeHelper_uncached:

    def __init__(self, A):
        """
        Version without saving intermediate results to save memory.
        From
        Initialize the object.

        Parameters
        ----------
        A : a dense or sparse square numpy matrix or ndarray
            The matrix to be exponentiated.
        """
        self.A = A
        self.ident = sp.spdiags(torch.ones(A.shape[0]),
                                torch.zeros(1, dtype=torch.long),
                                A.shape).to(A.device)

    def A2(self):
        return _smart_matrix_product(self.A, self.A)

    def A4(self):
        return _smart_matrix_product(self.A2(), self.A2())

    def A6(self):
        return _smart_matrix_product(self.A4(), self.A2())

    def A8(self):
        return _smart_matrix_product(self.A6(), self.A2())

    def A10(self):
        return _smart_matrix_product(self.A4(), self.A6())

    def d4_tight(self):
        return _onenorm(self.A4())**(1 / 4.)

    def d6_tight(self):
        return _onenorm(self.A6())**(1 / 6.)

    def d8_tight(self):
        return _onenorm(self.A8())**(1 / 8.)

    def d10_tight(self):
        return _onenorm(self.A10())**(1 / 10.)

    def d4_loose(self):
        return _onenormest_matrix_power(
            self.A2(),
            2,
        )**(1 / 4.)

    def d6_loose(self):
        return _onenormest_matrix_power(self.A2(), 3)**(1 / 6.)

    def d8_loose(self):
        return _onenormest_matrix_power(self.A4(), 2)**(1 / 8.)

    def d10_loose(self):
        return _onenormest_product(self.A4(), self.A6())**(1 / 10.)

    def pade3(self):
        b = (120., 60., 12., 1.)
        U = _smart_matrix_product(self.A, b[3] * self.A2() + b[1] * self.ident)
        V = b[2] * self.A2() + b[0] * self.ident
        return U, V

    def pade5(self):
        b = (30240., 15120., 3360., 420., 30., 1.)
        U = _smart_matrix_product(
            self.A, b[5] * self.A4() + b[3] * self.A2() + b[1] * self.ident)
        V = b[4] * self.A4() + b[2] * self.A2() + b[0] * self.ident
        return U, V

    def pade7(self):
        b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
        U = _smart_matrix_product(
            self.A, b[7] * self.A6() + b[5] * self.A4() + b[3] * self.A2() +
            b[1] * self.ident)
        V = b[6] * self.A6() + b[4] * self.A4() + b[2] * self.A2(
        ) + b[0] * self.ident
        return U, V

    def pade9(self):
        b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
             2162160., 110880., 3960., 90., 1.)
        U = _smart_matrix_product(
            self.A, (b[9] * self.A8() + b[7] * self.A6() + b[5] * self.A4() +
                     b[3] * self.A2() + b[1] * self.ident))
        V = (b[8] * self.A8() + b[6] * self.A6() + b[4] * self.A4() +
             b[2] * self.A2() + b[0] * self.ident)
        return U, V

    def pade13_scaled(self, s):
        b = (64764752532480000., 32382376266240000., 7771770303897600.,
             1187353796428800., 129060195264000., 10559470521600.,
             670442572800., 33522128640., 1323241920., 40840800., 960960.,
             16380., 182., 1.)
        B = self.A * 2**-s
        B2 = self.A2() * 2**(-2 * s)
        B4 = self.A4() * 2**(-4 * s)
        B6 = self.A6() * 2**(-6 * s)
        U2 = _smart_matrix_product(B6, b[13] * B6 + b[11] * B4 + b[9] * B2)
        U = _smart_matrix_product(
            B, (U2 + b[7] * B6 + b[5] * B4 + b[3] * B2 + b[1] * self.ident))
        V2 = _smart_matrix_product(B6, b[12] * B6 + b[10] * B4 + b[8] * B2)
        V = V2 + b[6] * B6 + b[4] * B4 + b[2] * B2 + b[0] * self.ident

        return U, V


@torch.no_grad()
def sparse_expm_pade(A):
    assert A.layout == torch.sparse_coo
    assert A.shape[0] == A.shape[1]

    # Track functions of A to help compute the matrix exponential.
    h = _ExpmPadeHelper(A)

    # Try Pade order 3.
    eta_1 = max(h.d4_loose, h.d6_loose)
    if eta_1 < 1.495585217958292e-002 and _ell(h.A, 3) == 0:
        U, V = h.pade3()
        return _solve_P_Q(U, V)

    # Try Pade order 5.
    eta_2 = max(h.d4_tight, h.d6_loose)
    if eta_2 < 2.539398330063230e-001 and _ell(h.A, 5) == 0:
        U, V = h.pade5()
        return _solve_P_Q(U, V)

    # Try Pade orders 7 and 9.
    eta_3 = max(h.d6_tight, h.d8_loose)
    if eta_3 < 9.504178996162932e-001 and _ell(h.A, 7) == 0:
        U, V = h.pade7()
        return _solve_P_Q(U, V)
    if eta_3 < 2.097847961257068e+000 and _ell(h.A, 9) == 0:
        U, V = h.pade9()
        return _solve_P_Q(U, V)

    # Use Pade order 13.
    eta_4 = max(h.d8_loose, h.d10_loose)
    eta_5 = min(eta_3, eta_4)
    theta_13 = 4.25

    # Choose smallest s>=0 such that 2**(-s) eta_5 <= theta_13
    if eta_5 == 0:
        # Nilpotent special case
        s = 0
    else:
        s = max(int(np.ceil(np.log2(eta_5 / theta_13))), 0)
    s = s + _ell(2**-s * h.A, 13)
    U, V = h.pade13_scaled(s)
    X = _solve_P_Q(U, V)
    # X = r_13(A)^(2^s) by repeated squaring.
    for i in range(s):
        if X.layout != torch.strided:
            X = sp.mm(X, X)
        else:
            X = X @ X
    return X


class TraceSparseExpm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):

        # # detach so we can cast to NumPy
        # indices_np = x.indices().cpu().numpy()
        # values_np = x.values().cpu().numpy()
        # size_np = x.size()
        # coo_matrix = scipy.sparse.coo_matrix((values_np, (indices_np[0], indices_np[1])),
        #                            shape=size_np)
        # E = splin.expm(coo_matrix)
        # E = torch.sparse_coo_tensor(E.nonzero(),
        #                             E.data,
        #                             E.shape,
        #                             device=x.device)

        E = sparse_expm_pade(x)
        if E.layout != torch.strided:
            f = sparse_trace(E)
        else:
            f = (E.diag() - 1).sum()
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input


sparse_expm = TraceSparseExpm.apply


def dense_expm(A):
    """
    copied from: https://github.com/pytorch/pytorch/issues/105225#issuecomment-1737803127

    Workaround for poor matrix_exp parallelism for large batches
    See https://github.com/pytorch/pytorch/issues/107291 for details
    The result may be less precise than torch.matrix_exp"""

    A_shape = A.shape
    # add batch dimension
    A = A.unsqueeze(0)
    A = A.flatten(end_dim=-3)
    theta_thres = 1.461661507209034e+00
    norms = torch.linalg.matrix_norm(A, ord=1)
    s = (torch.floor(torch.log2(norms / theta_thres)) + 1).clamp(min=0)
    s_max = s.max().cpu().item()
    is_nan = s_max != s_max
    if is_nan:
        # As of PyTorch 2.0.1, matrix_exp on nan causes undefined behavior
        return torch.full_like(A, torch.nan)
    # rescale
    output_scaled = torch.matrix_exp(A * torch.pow(2.0, -s).view(-1, 1, 1))

    # sort
    sorted_s, sorted_s_inds = torch.sort(s, dim=0)
    split_counts = torch.unique_consecutive(sorted_s, return_counts=True)[1]
    split_edges = torch.cumsum(split_counts, dim=0) - 1
    split_adv = torch.diff(split_edges, prepend=split_edges.new_zeros([1]))
    unique_s = sorted_s[split_edges].long()
    diffs = torch.diff(unique_s, prepend=unique_s.new_zeros([1]))

    idxs = split_adv.tolist()
    ps = diffs.tolist()

    acc = output_scaled[sorted_s_inds]
    output_pieces = []
    for i, p in zip(idxs, ps):
        for _ in range(p):
            acc = acc @ acc
        output_pieces.append(acc[:i + 1])
        acc = acc[i + 1:]

    # Compose the result back
    output = torch.cat(output_pieces, dim=0)
    output = output[torch.argsort(sorted_s_inds)]
    output = torch.reshape(output, A_shape)
    return output


if __name__ == '__main__':
    dense = torch.randn(3,
                        3,
                        dtype=torch.float32,
                        requires_grad=True,
                        device='cuda')
    dense.retain_grad()

    sparse = dense.clone().detach().to_sparse()
    sparse.requires_grad = True
    sparse.retain_grad()

    de_out = torch.matrix_exp(dense)
    de_out.trace().backward()
    de_grad = dense.grad

    sp_out = sparse_expm(sparse)
    sp_out.sum().backward()
    sp_grad = sparse.grad.to_dense()

    print('dense output:', de_out)
    print('dense grad:', de_grad)

    print('sparse output:', sp_out)
    print('sparse grad:', sp_grad)
