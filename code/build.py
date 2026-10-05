"""Build the four benchmark datasets with meta-path adjacencies and content features."""
import os, sys, gzip, json, pickle
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from data import *

TOPK = 20          # neighbours kept per row in every semantic meta-path graph
CDIM = 64          # content feature dimension


def content_svd(texts, dim=CDIM, seed=0):
    tf = TfidfVectorizer(max_features=20000, stop_words='english', min_df=2)
    X = tf.fit_transform(texts)
    d = min(dim, min(X.shape) - 1)
    Z = TruncatedSVD(n_components=d, random_state=seed).fit_transform(X)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    if d < dim:
        Z = np.hstack([Z, np.zeros((Z.shape[0], dim - d), np.float32)])
    return Z.astype(np.float32)


def pack(name, tr, va, te, nu, ni, mp_u, mp_i, cu, ci, split_kind, rd_u=None, rd_i=None):
    R = sp.csr_matrix((np.ones(len(tr), np.float32), (tr.u.values, tr.i.values)), shape=(nu, ni))
    R.data[:] = 1.0
    out = dict(name=name, n_users=nu, n_items=ni, split=split_kind,
               train=tr[['u', 'i']].values.astype(np.int64),
               valid=va[['u', 'i']].values.astype(np.int64),
               test=te[['u', 'i']].values.astype(np.int64),
               R=R, mp_u=mp_u, mp_i=mp_i, feat_u=cu, feat_i=ci, rd_u=rd_u or {}, rd_i=rd_i or {})
    with open(os.path.join(CACHE, name + '.pkl'), 'wb') as f:
        pickle.dump(out, f, protocol=4)
    dens = len(tr) / (nu * ni)
    print('%-10s U=%6d I=%6d train=%8d val=%7d test=%7d dens=%.3f%% mp_u=%s mp_i=%s'
          % (name, nu, ni, len(tr), len(va), len(te), dens * 100, list(mp_u), list(mp_i)))
    return out


def build_ml100k():
    r, users, items, gcols = load_ml100k()
    r = kcore(r, 10, 10)
    r, umap, imap = reindex(r)
    tr, va, te = loo_split(r)
    nu, ni = r.u.max() + 1, r.i.max() + 1
    users = users[users.u.isin(umap)].copy(); users['u'] = users.u.map(umap); users = users.sort_values('u')
    items = items[items.i.isin(imap)].copy(); items['i'] = items.i.map(imap); items = items.sort_values('i')
    Rtr = sp.csr_matrix((np.ones(len(tr), np.float32), (tr.u.values, tr.i.values)), shape=(nu, ni))
    occ = pd.factorize(users.occ)[0]
    agb = np.digitize(users.age.values, [18, 25, 35, 45, 56])
    gen = pd.factorize(users.gender)[0]
    mp_u = {'U-I-U': cooccur_adj(Rtr, TOPK), 'U-O-U': attr_adj(occ, nu, TOPK),
            'U-A-U': attr_adj(agb, nu, TOPK), 'U-G-U': attr_adj(gen, nu, TOPK)}
    rd_u = {'U-I-U': dens_bipartite(Rtr), 'U-O-U': dens_attr(occ, nu),
            'U-A-U': dens_attr(agb, nu), 'U-G-U': dens_attr(gen, nu)}
    G = items[gcols].values.astype(np.float32)
    yr = pd.to_datetime(items.date, format='%d-%b-%Y', errors='coerce').dt.year
    dec = ((yr.fillna(yr.median()) // 5) * 5).astype(int)
    dlab = pd.factorize(dec)[0]
    mp_i = {'I-U-I': cooccur_adj(Rtr.T.tocsr(), TOPK), 'I-G-I': multi_attr_adj(G, ni, TOPK),
            'I-Y-I': attr_adj(dlab, ni, TOPK)}
    rd_i = {'I-U-I': dens_bipartite(Rtr.T.tocsr()), 'I-G-I': dens_bipartite(sp.csr_matrix(G)),
            'I-Y-I': dens_attr(dlab, ni)}
    gnames = np.array(gcols)
    itxt = [str(t) + ' ' + ' '.join(gnames[G[k] > 0]) for k, t in enumerate(items.title.values)]
    ci = content_svd(itxt)
    utxt = ['occ_%s age_%s gen_%s' % (o, a, g) for o, a, g in zip(users.occ, agb, users.gender)]
    cu = content_svd(utxt)
    return pack('ml100k', tr, va, te, nu, ni, mp_u, mp_i, cu, ci, 'leave-one-out', rd_u, rd_i)


def build_ml1m():
    r, users, items = load_ml1m()
    r = kcore(r, 10, 10)
    r, umap, imap = reindex(r)
    tr, va, te = temporal_split(r)
    nu, ni = r.u.max() + 1, r.i.max() + 1
    users = users[users.u.isin(umap)].copy(); users['u'] = users.u.map(umap); users = users.sort_values('u')
    items = items[items.i.isin(imap)].copy(); items['i'] = items.i.map(imap); items = items.sort_values('i')
    Rtr = sp.csr_matrix((np.ones(len(tr), np.float32), (tr.u.values, tr.i.values)), shape=(nu, ni))
    occ = pd.factorize(users.occ)[0]; agb = pd.factorize(users.age)[0]; gen = pd.factorize(users.gender)[0]
    mp_u = {'U-I-U': cooccur_adj(Rtr, TOPK), 'U-O-U': attr_adj(occ, nu, TOPK),
            'U-A-U': attr_adj(agb, nu, TOPK), 'U-G-U': attr_adj(gen, nu, TOPK)}
    rd_u = {'U-I-U': dens_bipartite(Rtr), 'U-O-U': dens_attr(occ, nu),
            'U-A-U': dens_attr(agb, nu), 'U-G-U': dens_attr(gen, nu)}
    gset = sorted({g for s in items.genres for g in s.split('|')})
    gidx = {g: k for k, g in enumerate(gset)}
    rows, cols = [], []
    for k, s in enumerate(items.genres.values):
        for g in s.split('|'):
            rows.append(k); cols.append(gidx[g])
    G = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(ni, len(gset)))
    yr = items.title.str.extract(r'\((\d{4})\)')[0].astype(float)
    dec = ((yr.fillna(yr.median()) // 5) * 5).astype(int)
    dlab = pd.factorize(dec)[0]
    mp_i = {'I-U-I': cooccur_adj(Rtr.T.tocsr(), TOPK), 'I-G-I': multi_attr_adj(G, ni, TOPK),
            'I-Y-I': attr_adj(dlab, ni, TOPK)}
    rd_i = {'I-U-I': dens_bipartite(Rtr.T.tocsr()), 'I-G-I': dens_bipartite(G),
            'I-Y-I': dens_attr(dlab, ni)}
    itxt = [str(t) + ' ' + s.replace('|', ' ') for t, s in zip(items.title, items.genres)]
    ci = content_svd(itxt)
    cu = content_svd(['occ_%s age_%s gen_%s' % (o, a, g) for o, a, g in zip(users.occ, users.age, users.gender)])
    return pack('ml1m', tr, va, te, nu, ni, mp_u, mp_i, cu, ci, 'temporal', rd_u, rd_i)


def build_lastfm(seed=0):
    ua, fr, tg = load_lastfm()
    ua = ua.sort_values(['u', 'w'], ascending=[True, False])
    ua = ua.groupby('u').head(50)                       # cap at the 50 most-played artists per user
    d = kcore(ua[['u', 'i', 't']], 5, 5)
    d, umap, imap = reindex(d)
    rng = np.random.default_rng(seed)
    tr, va, te = random_split(d, rng)
    nu, ni = d.u.max() + 1, d.i.max() + 1
    Rtr = sp.csr_matrix((np.ones(len(tr), np.float32), (tr.u.values, tr.i.values)), shape=(nu, ni))
    fr = fr[fr.userID.isin(umap) & fr.friendID.isin(umap)]
    F = sp.csr_matrix((np.ones(len(fr), np.float32), (fr.userID.map(umap), fr.friendID.map(umap))), shape=(nu, nu))
    F = F.maximum(F.T)
    tg2 = tg[tg.userID.isin(umap)]
    tl = pd.factorize(tg2.tagID)[0]
    UT = sp.csr_matrix((np.ones(len(tg2), np.float32), (tg2.userID.map(umap).values, tl)), shape=(nu, tl.max() + 1))
    UT.data[:] = 1.0
    mp_u = {'U-A-U': cooccur_adj(Rtr, TOPK), 'U-F-U': topk_sparsify(F, TOPK), 'U-T-U': cooccur_adj(UT, TOPK)}
    rd_u = {'U-A-U': dens_bipartite(Rtr), 'U-F-U': float(F.nnz / (nu * (nu - 1))), 'U-T-U': dens_bipartite(UT)}
    tg3 = tg[tg.artistID.isin(imap)]
    tl3 = pd.factorize(tg3.tagID)[0]
    AT = sp.csr_matrix((np.ones(len(tg3), np.float32), (tg3.artistID.map(imap).values, tl3)), shape=(ni, tl3.max() + 1))
    AT.data[:] = 1.0
    mp_i = {'A-U-A': cooccur_adj(Rtr.T.tocsr(), TOPK), 'A-T-A': cooccur_adj(AT, TOPK)}
    rd_i = {'A-U-A': dens_bipartite(Rtr.T.tocsr()), 'A-T-A': dens_bipartite(AT)}
    tagnames = pd.read_csv(os.path.join(DATA, 'lastfm', 'tags.dat'), sep='\t', encoding='latin-1')
    tmap = dict(zip(tagnames.tagID, tagnames.tagValue.astype(str)))
    itxt = [''] * ni
    for aid, g in tg3.groupby('artistID'):
        itxt[imap[aid]] = ' '.join(str(tmap.get(t, '')) for t in g.tagID.values[:60])
    itxt = [x if x.strip() else 'unknown' for x in itxt]
    ci = content_svd(itxt)
    utxt = [''] * nu
    for uid, g in tg2.groupby('userID'):
        utxt[umap[uid]] = ' '.join(str(tmap.get(t, '')) for t in g.tagID.values[:60])
    utxt = [x if x.strip() else 'unknown' for x in utxt]
    cu = content_svd(utxt)
    return pack('lastfm', tr, va, te, nu, ni, mp_u, mp_i, cu, ci, 'random', rd_u, rd_i)


def build_amazon():
    r = load_amazon()
    r = kcore(r, 5, 5)
    print('amazon after 5-core:', len(r), r.u.nunique(), r.i.nunique())
    r, umap, imap = reindex(r)
    tr, va, te = temporal_split(r)
    nu, ni = r.u.max() + 1, r.i.max() + 1
    Rtr = sp.csr_matrix((np.ones(len(tr), np.float32), (tr.u.values, tr.i.values)), shape=(nu, ni))
    meta = {}
    with gzip.open(os.path.join(DATA, 'amazon', 'meta_Video_Games.jsonl.gz'), 'rt', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            a = d.get('parent_asin')
            if a in imap:
                meta[a] = d
    cats, stores, txt = np.full(ni, -1), np.full(ni, -1), [''] * ni
    cset, sset = {}, {}
    for a, k in imap.items():
        d = meta.get(a, {})
        c = (d.get('categories') or ['NA'])[-1]
        s = d.get('store') or 'NA'
        cats[k] = cset.setdefault(str(c), len(cset)); stores[k] = sset.setdefault(str(s), len(sset))
        txt[k] = '%s %s %s' % (d.get('title', ''), ' '.join(map(str, d.get('categories') or [])), s)
    txt = [x if x.strip() else 'unknown' for x in txt]
    mp_u = {'U-I-U': cooccur_adj(Rtr, TOPK)}
    rd_u = {'U-I-U': dens_bipartite(Rtr)}
    mp_i = {'I-U-I': cooccur_adj(Rtr.T.tocsr(), TOPK), 'I-C-I': attr_adj(cats, ni, TOPK),
            'I-B-I': attr_adj(stores, ni, TOPK)}
    rd_i = {'I-U-I': dens_bipartite(Rtr.T.tocsr()), 'I-C-I': dens_attr(cats, ni),
            'I-B-I': dens_attr(stores, ni)}
    ci = content_svd(txt)
    P = sp.diags(1.0 / np.maximum(np.asarray(Rtr.sum(1)).ravel(), 1)) @ Rtr
    cu = np.asarray(P @ ci, dtype=np.float32)
    return pack('amazon', tr, va, te, nu, ni, mp_u, mp_i, cu, ci, 'temporal', rd_u, rd_i)


if __name__ == '__main__':
    which = sys.argv[1:] or ['ml100k', 'ml1m', 'lastfm', 'amazon']
    for w in which:
        globals()['build_' + w]()
