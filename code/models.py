"""Recommenders: classical, graph, self-supervised graph, the density-weighted InfoMax baseline (HDGCN), and MIDGaP."""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sp2t(A):
    A = sp.coo_matrix(A, dtype=np.float32)
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    val = torch.tensor(A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, A.shape).coalesce().to(DEV)


def bipartite_norm(R):
    nu, ni = R.shape
    A = sp.bmat([[None, R], [R.T, None]], format='csr', dtype=np.float32)
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1.0
    D = sp.diags(np.power(d, -0.5))
    return sp.csr_matrix(D @ A @ D, dtype=np.float32)


def gcn_norm(A):
    A = sp.csr_matrix(A, dtype=np.float32).copy()
    A.setdiag(0); A.eliminate_zeros()
    A = A + sp.eye(A.shape[0], dtype=np.float32, format='csr')
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1.0
    D = sp.diags(np.power(d, -0.5))
    return sp.csr_matrix(D @ A @ D, dtype=np.float32)


class Base(nn.Module):
    needs_grad = True

    def embeddings(self):
        raise NotImplementedError

    def loss(self, u, i, j):
        raise NotImplementedError


class BPRMF(Base):
    def __init__(self, nu, ni, d=64, **kw):
        super().__init__()
        self.U = nn.Embedding(nu, d); self.I = nn.Embedding(ni, d)
        nn.init.normal_(self.U.weight, std=0.1); nn.init.normal_(self.I.weight, std=0.1)

    def embeddings(self):
        return self.U.weight, self.I.weight

    def loss(self, u, i, j):
        U, I = self.embeddings()
        pu, pi, pj = U[u], I[i], I[j]
        x = (pu * pi).sum(-1) - (pu * pj).sum(-1)
        reg = 1e-4 * (pu.pow(2).sum() + pi.pow(2).sum() + pj.pow(2).sum()) / len(u)
        return -F.logsigmoid(x).mean() + reg


class LightGCN(Base):
    def __init__(self, nu, ni, R, d=64, layers=3, **kw):
        super().__init__()
        self.nu, self.ni, self.L = nu, ni, layers
        self.E = nn.Embedding(nu + ni, d)
        nn.init.normal_(self.E.weight, std=0.1)
        self.A = sp2t(bipartite_norm(R))
        self.reg = kw.get('reg', 1e-4)

    def propagate(self, E=None, A=None):
        e = self.E.weight if E is None else E
        A = self.A if A is None else A
        out = [e]
        for _ in range(self.L):
            e = torch.sparse.mm(A, e)
            out.append(e)
        e = torch.stack(out, 1).mean(1)
        return e[:self.nu], e[self.nu:]

    def embeddings(self):
        return self.propagate()

    def bpr(self, U, I, u, i, j):
        x = (U[u] * I[i]).sum(-1) - (U[u] * I[j]).sum(-1)
        e0 = self.E.weight
        reg = self.reg * (e0[u].pow(2).sum() + e0[self.nu + i].pow(2).sum() + e0[self.nu + j].pow(2).sum()) / len(u)
        return -F.logsigmoid(x).mean() + reg

    def loss(self, u, i, j):
        U, I = self.embeddings()
        return self.bpr(U, I, u, i, j)


def info_nce(a, b, tau):
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
    pos = (a * b).sum(-1) / tau
    neg = torch.logsumexp(a @ b.t() / tau, dim=-1)
    return (neg - pos).mean()


class SGL(LightGCN):
    """Edge-dropout contrastive augmentation (Wu et al., SIGIR 2021)."""
    def __init__(self, nu, ni, R, d=64, layers=3, ssl=0.1, tau=0.2, drop=0.1, **kw):
        super().__init__(nu, ni, R, d, layers, **kw)
        self.ssl, self.tau, self.drop = ssl, tau, drop
        self._idx = self.A.indices(); self._val = self.A.values(); self._shape = self.A.shape

    def _view(self):
        m = torch.rand(self._val.numel(), device=DEV) > self.drop
        return torch.sparse_coo_tensor(self._idx[:, m], self._val[m], self._shape).coalesce()

    def loss(self, u, i, j):
        U, I = self.embeddings()
        L = self.bpr(U, I, u, i, j)
        U1, I1 = self.propagate(A=self._view())
        U2, I2 = self.propagate(A=self._view())
        uu = torch.unique(u); ii = torch.unique(i)
        L = L + self.ssl * (info_nce(U1[uu], U2[uu], self.tau) + info_nce(I1[ii], I2[ii], self.tau))
        return L


class SimGCL(LightGCN):
    """Augmentation-free noise perturbation (Yu et al., SIGIR 2022)."""
    def __init__(self, nu, ni, R, d=64, layers=3, ssl=0.1, tau=0.2, eps=0.1, **kw):
        super().__init__(nu, ni, R, d, layers, **kw)
        self.ssl, self.tau, self.eps = ssl, tau, eps

    def perturbed(self):
        e = self.E.weight
        out = []
        for _ in range(self.L):
            e = torch.sparse.mm(self.A, e)
            n = torch.rand_like(e)
            e = e + torch.sign(e) * F.normalize(n, dim=-1) * self.eps
            out.append(e)
        e = torch.stack(out, 1).mean(1)
        return e[:self.nu], e[self.nu:]

    def loss(self, u, i, j):
        U, I = self.embeddings()
        L = self.bpr(U, I, u, i, j)
        U1, I1 = self.perturbed(); U2, I2 = self.perturbed()
        uu = torch.unique(u); ii = torch.unique(i)
        return L + self.ssl * (info_nce(U1[uu], U2[uu], self.tau) + info_nce(I1[ii], I2[ii], self.tau))


class XSimGCL(LightGCN):
    """Cross-layer contrast with a single perturbed forward pass (Yu et al., TKDE 2023)."""
    def __init__(self, nu, ni, R, d=64, layers=3, ssl=0.1, tau=0.2, eps=0.2, star=1, **kw):
        super().__init__(nu, ni, R, d, layers, **kw)
        self.ssl, self.tau, self.eps, self.star = ssl, tau, eps, star

    def forward_both(self):
        e = self.E.weight
        clean, noisy, cl = [e], None, None
        en = e
        for l in range(self.L):
            e = torch.sparse.mm(self.A, e)
            clean.append(e)
            en = torch.sparse.mm(self.A, en)
            n = torch.rand_like(en)
            en = en + torch.sign(en) * F.normalize(n, dim=-1) * self.eps
            if l == self.star:
                cl = en
        c = torch.stack(clean, 1).mean(1)
        if cl is None:
            cl = en
        return (c[:self.nu], c[self.nu:]), (cl[:self.nu], cl[self.nu:])

    def loss(self, u, i, j):
        (U, I), (Uc, Ic) = self.forward_both()
        L = self.bpr(U, I, u, i, j)
        uu = torch.unique(u); ii = torch.unique(i)
        return L + self.ssl * (info_nce(U[uu], Uc[uu], self.tau) + info_nce(I[ii], Ic[ii], self.tau))


class HDGCN(Base):
    """Density-weighted InfoMax baseline (HDGCN): per-meta-path GCN, semantic attention,
    DGI-style corruption, and the subgraph-density weighting term."""
    needs_grad = True

    def __init__(self, nu, ni, R, mp_u, mp_i, feat_u, feat_i, rd_u, rd_i,
                 d=64, gamma=0.2, bpr=0.0, **kw):
        super().__init__()
        self.bpr_w = bpr
        self.nu, self.ni = nu, ni
        self.Xu = torch.tensor(feat_u, device=DEV); self.Xi = torch.tensor(feat_i, device=DEV)
        self.Au = [sp2t(gcn_norm(A)) for A in mp_u.values()]
        self.Ai = [sp2t(gcn_norm(A)) for A in mp_i.values()]
        fu, fi = feat_u.shape[1], feat_i.shape[1]
        self.Wu = nn.ModuleList([nn.Linear(fu, d) for _ in self.Au])
        self.Wi = nn.ModuleList([nn.Linear(fi, d) for _ in self.Ai])
        self.qu = nn.Linear(d, d); self.qi = nn.Linear(d, d)
        self.au = nn.Parameter(torch.randn(d) * 0.1); self.ai = nn.Parameter(torch.randn(d) * 0.1)
        self.Du = nn.Bilinear(d, d, 1); self.Di = nn.Bilinear(d, d, 1)
        self.wu = nn.Parameter(torch.zeros(len(self.Au))); self.wi = nn.Parameter(torch.zeros(len(self.Ai)))
        self.du = torch.tensor([rd_u[k] for k in mp_u], dtype=torch.float32, device=DEV)
        self.di = torch.tensor([rd_i[k] for k in mp_i], dtype=torch.float32, device=DEV)
        self.gamma = gamma

    def _branch(self, X, As, Ws, q, a, shuffle=False):
        Xs = X[torch.randperm(X.shape[0], device=DEV)] if shuffle else X
        H = [torch.relu(torch.sparse.mm(A, W(Xs))) for A, W in zip(As, Ws)]
        S = torch.stack(H, 0)
        e = torch.tanh(q(S)).mean(1) @ a
        beta = torch.softmax(e, 0)
        return (beta[:, None, None] * S).sum(0), S, beta

    def embeddings(self):
        Hu, _, _ = self._branch(self.Xu, self.Au, self.Wu, self.qu, self.au)
        Hi, _, _ = self._branch(self.Xi, self.Ai, self.Wi, self.qi, self.ai)
        return Hu, Hi

    def _dgi(self, X, As, Ws, q, a, D):
        H, S, beta = self._branch(X, As, Ws, q, a)
        Hn, Sn, _ = self._branch(X, As, Ws, q, a, shuffle=True)
        s = torch.sigmoid(S.mean(1))                       # graph summary per meta-path
        L = 0.0
        for m in range(S.shape[0]):
            sm = s[m].expand(S.shape[1], -1)
            pos = D(S[m], sm).squeeze(-1)
            neg = D(Sn[m], sm).squeeze(-1)
            L = L + F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) \
                  + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
        return L / S.shape[0], beta

    def loss(self, u, i, j):
        Lu, bu = self._dgi(self.Xu, self.Au, self.Wu, self.qu, self.au, self.Du)
        Li, bi = self._dgi(self.Xi, self.Ai, self.Wi, self.qi, self.ai, self.Di)
        # gamma * I(W_P, A_P): the density prior, as a cross-entropy to normalised density
        pu = torch.softmax(self.wu, 0); pi = torch.softmax(self.wi, 0)
        tu = self.du / self.du.sum(); ti = self.di / self.di.sum()
        Ld = (tu * (tu.clamp_min(1e-9).log() - pu.clamp_min(1e-9).log())).sum() \
           + (ti * (ti.clamp_min(1e-9).log() - pi.clamp_min(1e-9).log())).sum()
        L = Lu + Li + self.gamma * Ld
        if self.bpr_w > 0:
            Hu, Hi = self.embeddings()
            x = (Hu[u] * Hi[i]).sum(-1) - (Hu[u] * Hi[j]).sum(-1)
            L = L + self.bpr_w * (-F.logsigmoid(x).mean())
        return L


class MIDGaP(Base):
    """Density-corrected mutual-information gating with a variational bottleneck."""

    def __init__(self, nu, ni, R, mp_u, mp_i, feat_u, feat_i, prior_u, prior_i,
                 d=64, layers=2, beta=1e-3, gamma=1.0, tau=0.2, use_ib=True,
                 use_mp=True, use_content=True, hard_neg=True, **kw):
        super().__init__()
        self.nu, self.ni, self.L = nu, ni, layers
        self.beta, self.gamma, self.tau = beta, gamma, tau
        self.use_ib, self.use_mp, self.use_content, self.hard_neg = use_ib, use_mp, use_content, hard_neg
        self.E = nn.Embedding(nu + ni, d); nn.init.normal_(self.E.weight, std=0.1)
        self.A = sp2t(bipartite_norm(R))
        self.Cu = torch.tensor(feat_u, device=DEV); self.Ci = torch.tensor(feat_i, device=DEV)
        self.Pu = nn.Linear(feat_u.shape[1], d, bias=False)
        self.Pi = nn.Linear(feat_i.shape[1], d, bias=False)
        self.Au = [sp2t(gcn_norm(A)) for A in mp_u.values()]
        self.Ai = [sp2t(gcn_norm(A)) for A in mp_i.values()]
        self.gu = nn.Parameter(torch.zeros(len(self.Au)))
        self.gi = nn.Parameter(torch.zeros(len(self.Ai)))
        self.pu = torch.tensor([prior_u[k] for k in mp_u], dtype=torch.float32, device=DEV)
        self.pi = torch.tensor([prior_i[k] for k in mp_i], dtype=torch.float32, device=DEV)
        self.mu = nn.Linear(d, d); self.lv = nn.Linear(d, d)
        nn.init.zeros_(self.lv.weight); nn.init.constant_(self.lv.bias, -4.0)
        self.reg = kw.get('reg', 1e-4)

    def base(self):
        e = self.E.weight
        if self.use_content:
            c = torch.cat([self.Pu(self.Cu), self.Pi(self.Ci)], 0)
            e = e + c
        return e

    def encode(self):
        e = self.base()
        out = [e]
        for _ in range(self.L):
            e = torch.sparse.mm(self.A, e)
            out.append(e)
        h = torch.stack(out, 1).mean(1)
        hu, hi = h[:self.nu], h[self.nu:]
        if self.use_mp:
            b = self.base()
            bu, bi = b[:self.nu], b[self.nu:]
            gu = torch.softmax(self.gu, 0); gi = torch.softmax(self.gi, 0)
            hu = hu + sum(gu[m] * torch.sparse.mm(A, bu) for m, A in enumerate(self.Au))
            hi = hi + sum(gi[m] * torch.sparse.mm(A, bi) for m, A in enumerate(self.Ai))
        return hu, hi

    def posterior(self):
        hu, hi = self.encode()
        h = torch.cat([hu, hi], 0)
        mu = self.mu(h)
        lv = self.lv(h).clamp(-10, 2) if self.use_ib else torch.full_like(mu, -20.0)
        return mu, lv

    def embeddings(self):
        mu, _ = self.posterior()
        return mu[:self.nu], mu[self.nu:]

    def sample(self, n=1):
        mu, lv = self.posterior()
        s = torch.exp(0.5 * lv)
        return [mu + s * torch.randn_like(s) for _ in range(n)], mu, lv

    def gate_kl(self):
        gu = torch.softmax(self.gu, 0); gi = torch.softmax(self.gi, 0)
        ku = (gu * (gu.clamp_min(1e-9).log() - self.pu.clamp_min(1e-9).log())).sum()
        ki = (gi * (gi.clamp_min(1e-9).log() - self.pi.clamp_min(1e-9).log())).sum()
        return ku + ki

    def loss(self, u, i, j):
        (z,), mu, lv = self.sample(1)
        Z_u, Z_i = z[:self.nu], z[self.nu:]
        x = (Z_u[u] * Z_i[i]).sum(-1) - (Z_u[u] * Z_i[j]).sum(-1)
        L = -F.logsigmoid(x).mean()
        e0 = self.E.weight
        L = L + self.reg * (e0[u].pow(2).sum() + e0[self.nu + i].pow(2).sum() + e0[self.nu + j].pow(2).sum()) / len(u)
        if self.use_ib:
            idx = torch.cat([u, self.nu + i, self.nu + j]).unique()
            kl = 0.5 * (mu[idx].pow(2) + lv[idx].exp() - 1.0 - lv[idx]).sum(-1).mean()
            L = L + self.beta * kl
        if self.use_mp:
            L = L + self.gamma * self.gate_kl()
        return L
