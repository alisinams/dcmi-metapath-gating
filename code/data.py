"""Dataset construction: implicit feedback, k-core, temporal splits, meta-path adjacencies."""
import os, gzip, json, pickle, re
import numpy as np, pandas as pd
import scipy.sparse as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA = os.path.join(ROOT, 'data')
CACHE = os.path.join(ROOT, 'cache')
os.makedirs(CACHE, exist_ok=True)


def kcore(df, ku, ki):
    while True:
        uc = df.u.value_counts(); ic = df.i.value_counts()
        bad_u = uc[uc < ku].index; bad_i = ic[ic < ki].index
        if len(bad_u) == 0 and len(bad_i) == 0:
            return df
        df = df[~df.u.isin(bad_u) & ~df.i.isin(bad_i)]
        if len(df) == 0:
            return df


def reindex(df):
    us = {u: k for k, u in enumerate(sorted(df.u.unique()))}
    its = {i: k for k, i in enumerate(sorted(df.i.unique()))}
    df = df.copy()
    df['u'] = df.u.map(us); df['i'] = df.i.map(its)
    return df, us, its


def temporal_split(df, f_tr=0.8, f_va=0.1):
    t1 = df.t.quantile(f_tr); t2 = df.t.quantile(f_tr + f_va)
    tr = df[df.t <= t1]; va = df[(df.t > t1) & (df.t <= t2)]; te = df[df.t > t2]
    # keep only users/items seen in training
    us = set(tr.u); its = set(tr.i)
    va = va[va.u.isin(us) & va.i.isin(its)]
    te = te[te.u.isin(us) & te.i.isin(its)]
    return tr, va, te


def random_split(df, rng, f_tr=0.8, f_va=0.1):
    tr, va, te = [], [], []
    for u, g in df.groupby('u'):
        idx = rng.permutation(len(g)); g = g.iloc[idx]
        n = len(g); n1 = max(1, int(round(n * f_tr))); n2 = max(n1 + 1, int(round(n * (f_tr + f_va))))
        tr.append(g.iloc[:n1]); va.append(g.iloc[n1:n2]); te.append(g.iloc[n2:])
    tr = pd.concat(tr); va = pd.concat(va); te = pd.concat(te)
    us = set(tr.u); its = set(tr.i)
    va = va[va.u.isin(us) & va.i.isin(its)]; te = te[te.u.isin(us) & te.i.isin(its)]
    return tr, va, te


# ---------------- loaders ----------------

def load_ml100k():
    p = os.path.join(DATA, 'ml-100k', 'ml-100k')
    r = pd.read_csv(os.path.join(p, 'u.data'), sep='\t', names=['u', 'i', 'r', 't'])
    users = pd.read_csv(os.path.join(p, 'u.user'), sep='|', names=['u', 'age', 'gender', 'occ', 'zip'], encoding='latin-1')
    gcols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
             'Fantasy', 'FilmNoir', 'Horror', 'Musical', 'Mystery', 'Romance', 'SciFi', 'Thriller', 'War', 'Western']
    items = pd.read_csv(os.path.join(p, 'u.item'), sep='|', encoding='latin-1', header=None,
                        names=['i', 'title', 'date', 'vdate', 'url'] + gcols)
    r = r[r.r >= 4][['u', 'i', 't']]
    return r, users, items, gcols


def load_ml1m():
    p = os.path.join(DATA, 'ml-1m', 'ml-1m')
    r = pd.read_csv(os.path.join(p, 'ratings.dat'), sep='::', engine='python', names=['u', 'i', 'r', 't'], encoding='latin-1')
    users = pd.read_csv(os.path.join(p, 'users.dat'), sep='::', engine='python', names=['u', 'gender', 'age', 'occ', 'zip'], encoding='latin-1')
    items = pd.read_csv(os.path.join(p, 'movies.dat'), sep='::', engine='python', names=['i', 'title', 'genres'], encoding='latin-1')
    r = r[r.r >= 4][['u', 'i', 't']]
    return r, users, items


def load_lastfm():
    p = os.path.join(DATA, 'lastfm')
    ua = pd.read_csv(os.path.join(p, 'user_artists.dat'), sep='\t')
    ua.columns = ['u', 'i', 'w']
    ua['t'] = 0
    fr = pd.read_csv(os.path.join(p, 'user_friends.dat'), sep='\t')
    tg = pd.read_csv(os.path.join(p, 'user_taggedartists.dat'), sep='\t')
    return ua[['u', 'i', 't', 'w']], fr, tg


def load_amazon(min_year=2015):
    p = os.path.join(DATA, 'amazon')
    r = pd.read_csv(os.path.join(p, 'Video_Games.csv.gz'))
    r.columns = ['u', 'i', 'r', 't']
    r['t'] = r.t.astype('int64') // 1000
    r = r[pd.to_datetime(r.t, unit='s').dt.year >= min_year]
    return r[r.r >= 4.0][['u', 'i', 't']]


# ---------------- meta-path adjacency helpers ----------------

def norm_adj(A):
    A = sp.csr_matrix(A, dtype=np.float32)
    A.setdiag(0); A.eliminate_zeros()
    A = A + sp.eye(A.shape[0], dtype=np.float32, format='csr')
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1.0
    Dm = sp.diags(np.power(d, -0.5))
    return sp.csr_matrix(Dm @ A @ Dm, dtype=np.float32)


def topk_sparsify(A, k):
    """Keep top-k entries per row of a weighted adjacency (symmetrised)."""
    A = sp.csr_matrix(A)
    rows, cols, vals = [], [], []
    for r in range(A.shape[0]):
        s, e = A.indptr[r], A.indptr[r + 1]
        if e - s == 0:
            continue
        idx = A.indices[s:e]; v = A.data[s:e]
        m = idx != r
        idx, v = idx[m], v[m]
        if len(v) > k:
            sel = np.argpartition(-v, k)[:k]
            idx, v = idx[sel], v[sel]
        rows.append(np.full(len(idx), r)); cols.append(idx); vals.append(v)
    if not rows:
        return sp.csr_matrix(A.shape, dtype=np.float32)
    R = np.concatenate(rows); C = np.concatenate(cols); V = np.concatenate(vals)
    M = sp.csr_matrix((V, (R, C)), shape=A.shape, dtype=np.float32)
    M = M.maximum(M.T)
    return M


def attr_adj(labels, n, topk=None):
    """A^{X-attr-X}: two nodes linked when they share an attribute value."""
    lab = np.asarray(labels)
    ok = lab >= 0
    idx = np.where(ok)[0]
    B = sp.csr_matrix((np.ones(len(idx), np.float32), (idx, lab[ok])), shape=(n, lab.max() + 1))
    cnt = np.asarray(B.sum(0)).ravel(); cnt[cnt == 0] = 1
    Bw = B @ sp.diags(1.0 / np.sqrt(cnt))
    A = Bw @ Bw.T
    return topk_sparsify(A, topk) if topk else sp.csr_matrix(A)


def multi_attr_adj(B, n, topk=None):
    """B: (n x n_attr) binary membership matrix (multi-label attributes)."""
    B = sp.csr_matrix(B, dtype=np.float32)
    cnt = np.asarray(B.sum(0)).ravel(); cnt[cnt == 0] = 1
    Bw = B @ sp.diags(1.0 / np.sqrt(cnt))
    A = Bw @ Bw.T
    return topk_sparsify(A, topk) if topk else sp.csr_matrix(A)


def cooccur_adj(R, topk):
    """A^{X-Y-X} from bipartite R (rows=X). Cosine-normalised co-occurrence."""
    R = sp.csr_matrix(R, dtype=np.float32)
    d = np.sqrt(np.asarray(R.sum(1)).ravel()); d[d == 0] = 1
    Rn = sp.diags(1.0 / d) @ R
    A = Rn @ Rn.T
    return topk_sparsify(A, topk)


def loo_split(df):
    """Leave-one-out by timestamp: last interaction -> test, second last -> valid."""
    df = df.sort_values(['u', 't'], kind='mergesort')
    g = df.groupby('u').cumcount(ascending=False)
    te = df[g == 0]; va = df[g == 1]; tr = df[g >= 2]
    us = set(tr.u); its = set(tr.i)
    va = va[va.u.isin(us) & va.i.isin(its)]; te = te[te.u.isin(us) & te.i.isin(its)]
    return tr, va, te


# ---------- raw (unsparsified) meta-path densities, as used by the HDGCN density prior ----------

def dens_attr(labels, n):
    lab = np.asarray(labels); lab = lab[lab >= 0]
    _, c = np.unique(lab, return_counts=True)
    return float((c * (c - 1)).sum() / (n * (n - 1)))


def dens_bipartite(B, seed=0, sample=1500):
    """P(two nodes share at least one neighbour) estimated on a random row sample."""
    B = sp.csr_matrix(B, dtype=np.float32)
    n = B.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    tot = 0; hit = 0
    for s0 in range(0, len(idx), 200):
        blk = idx[s0:s0 + 200]
        C = (B[blk] @ B.T)
        C = sp.csr_matrix(C)
        h = C.nnz - int((C.diagonal() > 0).sum())
        hit += h; tot += len(blk) * (n - 1)
    return float(hit / tot)
