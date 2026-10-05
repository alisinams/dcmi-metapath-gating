"""Density-corrected mutual information (DCMI) for meta-path informativeness.

For every meta-path P the estimator answers one question: how much does knowing a
node's P-neighbourhood reduce uncertainty about interactions the model has not
seen?  The raw plug-in mutual information is confounded by the density of the
meta-path subgraph, so it is referenced against a degree-preserving null
ensemble simulated by Monte Carlo.
"""
import numpy as np
import scipy.sparse as sp


def rownorm(A):
    A = sp.csr_matrix(A, dtype=np.float32)
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1.0
    return sp.diags(1.0 / d) @ A


def density(A):
    A = sp.csr_matrix(A)
    n = A.shape[0]
    m = (A.nnz - (A.diagonal() > 0).sum()) / 2.0
    return 2.0 * m / (n * (n - 1))


def null_adj(A, rng):
    """Configuration-model surrogate: same edge count, same expected degrees."""
    A = sp.csr_matrix(A)
    n = A.shape[0]
    deg = np.asarray((A > 0).sum(1)).ravel().astype(np.float64)
    m = int(round(A.nnz / 2.0))
    if m == 0 or deg.sum() == 0:
        return sp.csr_matrix(A.shape, dtype=np.float32)
    p = deg / deg.sum()
    a = rng.choice(n, size=2 * m, p=p)
    b = rng.choice(n, size=2 * m, p=p)
    keep = a != b
    a, b = a[keep][:m], b[keep][:m]
    M = sp.csr_matrix((np.ones(len(a), np.float32), (a, b)), shape=(n, n))
    M = M.maximum(M.T)
    M.data[:] = 1.0
    return M


def _mi_discrete(x, y, nx):
    """Plug-in mutual information (nats) between a discrete code x and binary y."""
    n = len(x)
    c = np.zeros((nx, 2), np.float64)
    np.add.at(c, (x, y), 1.0)
    p = c / n
    px = p.sum(1, keepdims=True); py = p.sum(0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = p * np.log(p / (px * py))
    return float(np.nansum(t))


def _bins(v, nb):
    """Quantile codes; zero-inflated scores get their own bin."""
    z = v <= 0
    code = np.zeros(len(v), np.int64)
    pos = ~z
    if pos.sum() > nb:
        q = np.quantile(v[pos], np.linspace(0, 1, nb + 1)[1:-1])
        q = np.unique(q)
        code[pos] = np.searchsorted(q, v[pos], side='right') + 1
    elif pos.sum() > 0:
        code[pos] = 1
    return code, int(code.max()) + 1


def _rows(A, R, side, rows):
    """Propagated scores for a subset of user rows, densified batch-wise."""
    if side == 'u':
        return np.asarray((rownorm(A)[rows] @ R).todense(), dtype=np.float32)
    return np.asarray((R[rows] @ rownorm(A).T).todense(), dtype=np.float32)


def path_mi(A, R, side, probe, neg_per_pos, K, rng, batch=1024, nbins=12):
    """MI between the (binned) meta-path-propagated score of a pair and whether the
    pair is a held-out interaction."""
    n_users, n_items = R.shape
    pu, pi = probe[:, 0], probe[:, 1]
    nu = np.repeat(pu, neg_per_pos)
    ni = rng.integers(0, n_items, size=len(nu))
    qu = np.concatenate([pu, nu]); qi = np.concatenate([pi, ni])
    y = np.concatenate([np.ones(len(pu), np.int64), np.zeros(len(nu), np.int64)])
    rows, inv = np.unique(qu, return_inverse=True)
    v = np.zeros(len(qu), np.float32)
    for s0 in range(0, len(rows), batch):
        idx = rows[s0:s0 + batch]
        S = _rows(A, R, side, idx)
        S -= 1e6 * np.asarray(R[idx].todense(), dtype=np.float32)
        sel = (inv >= s0) & (inv < s0 + len(idx))
        v[sel] = S[inv[sel] - s0, qi[sel]]
    v[v < 0] = 0.0                                    # training items carry no probe signal
    code, nb = _bins(v, nbins)
    return _mi_discrete(code, y, nb)


def dcmi(mp, R, side, probe, n_null=40, neg_per_pos=20, K=20, seed=0, max_probe=4000):
    """Return raw MI, null mean/sd, density and the density-corrected score per meta-path."""
    rng = np.random.default_rng(seed)
    if len(probe) > max_probe:
        probe = probe[rng.choice(len(probe), max_probe, replace=False)]
    out = {}
    for name, A in mp.items():
        raw = path_mi(A, R, side, probe, neg_per_pos, K, rng)
        nulls = np.array([path_mi(null_adj(A, rng), R, side, probe, neg_per_pos, K, rng)
                          for _ in range(n_null)])
        mu, sd = float(nulls.mean()), float(nulls.std() + 1e-12)
        out[name] = dict(mi=raw, null_mu=mu, null_sd=sd, density=float(density(A)),
                         excess=raw - mu, z=(raw - mu) / sd)
    return out


def gate_prior(stats, kind='dcmi', temp=0.25):
    """Turn meta-path statistics into a prior over gates.

    All variants share the transform: scores are range-normalised to [-1, 0]
    relative to the best path and passed through a tempered softmax, so the
    comparison between criteria isolates the scoring function itself.

    'dcmi' standardises the excess by the null spread, which measures how
    certain the excess is; 'excess' uses the raw excess, which measures how
    large it is. The two disagree whenever null variances differ sharply.
    """
    names = list(stats)
    if kind == 'dcmi':
        v = np.array([stats[n]['z'] for n in names], float)
    elif kind in ('excess', 'excess_scaled'):
        v = np.array([stats[n]['excess'] for n in names], float)
    elif kind == 'dcmi_scaled':
        v = np.array([stats[n]['z'] for n in names], float)
    elif kind == 'mi':
        v = np.array([stats[n]['mi'] for n in names], float)
    elif kind == 'density':
        v = np.array([stats[n]['density'] for n in names], float)
    elif kind == 'uniform':
        v = np.zeros(len(names))
    else:
        raise ValueError(kind)
    if kind.endswith('_scaled'):
        # scale by the mean magnitude, so a prior sharpens only when the paths
        # really differ; range normalisation cannot express 'these are tied'
        sc = float(np.mean(np.abs(v)))
        z = np.zeros_like(v) if sc <= 0 else (v - v.max()) / sc
    else:
        rng_ = v.max() - v.min()
        z = np.zeros_like(v) if rng_ <= 0 else (v - v.max()) / rng_
    e = np.exp(z / temp)
    return dict(zip(names, e / e.sum()))
