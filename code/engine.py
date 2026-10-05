"""Training loop, full-catalogue evaluation, beyond-accuracy metrics, uncertainty."""
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from models import DEV

KS = (10, 20)


# ------------------------- evaluation -------------------------

def _dcg(hits, K):
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    return (hits * disc).sum(1)


def rank_metrics(scores, gt_lists, K):
    """scores: (B, n_items) already masked. gt_lists: list of positive-item arrays."""
    top = torch.topk(scores, K, dim=1).indices.cpu().numpy()
    rec = np.zeros(len(top)); ndcg = np.zeros(len(top))
    hits = np.zeros((len(top), K))
    for b, g in enumerate(gt_lists):
        s = set(g.tolist())
        h = np.array([1.0 if t in s else 0.0 for t in top[b]])
        hits[b] = h
        rec[b] = h.sum() / max(len(s), 1)
        idcg = (1.0 / np.log2(np.arange(2, min(len(s), K) + 2))).sum()
        ndcg[b] = (h / np.log2(np.arange(2, K + 2))).sum() / max(idcg, 1e-9)
    return rec, ndcg, top


def emb_scorer(U, I):
    def f(ub):
        return U[ub] @ I.t()
    return f


@torch.no_grad()
def evaluate(scorer, n_items, train_csr, mask_csr, gt, ks=KS, batch=512, pop=None, want_lists=False):
    users = sorted(gt)
    res = {k: {'recall': [], 'ndcg': []} for k in ks}
    Kmax = max(ks)
    all_top = []
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        sc = scorer(ub).float()
        for r, u in enumerate(ub):
            sc[r, train_csr.indices[train_csr.indptr[u]:train_csr.indptr[u + 1]]] = -1e9
            if mask_csr is not None:
                sc[r, mask_csr.indices[mask_csr.indptr[u]:mask_csr.indptr[u + 1]]] = -1e9
        gts = [gt[u] for u in ub]
        for k in ks:
            rc, nd, top = rank_metrics(sc, gts, k)
            res[k]['recall'].append(rc); res[k]['ndcg'].append(nd)
            if k == Kmax:
                all_top.append(top)
    out = {}
    for k in ks:
        out['recall@%d' % k] = np.concatenate(res[k]['recall'])
        out['ndcg@%d' % k] = np.concatenate(res[k]['ndcg'])
    T = np.concatenate(all_top, 0)
    cnt = np.bincount(T.ravel(), minlength=n_items).astype(np.float64)
    out['coverage'] = float((cnt > 0).sum() / n_items)
    p = np.sort(cnt / cnt.sum())
    n = len(p)
    out['gini'] = float((2 * np.arange(1, n + 1) - n - 1).dot(p) / (n - 1))
    if pop is not None:
        out['arp'] = float(pop[T].mean())
    if want_lists:
        out['top'] = T; out['users'] = np.array(users)
    return out


def summarise(m):
    return {k: (float(np.mean(v)) if isinstance(v, np.ndarray) else v)
            for k, v in m.items() if k not in ('top', 'users')}


# ------------------------- training -------------------------

def sample_batch(pairs, n_items, bs, rng, R_csr, model=None, hard=0):
    idx = rng.integers(0, len(pairs), bs)
    u = pairs[idx, 0]; i = pairs[idx, 1]
    if hard and model is not None:
        cand = rng.integers(0, n_items, (bs, hard))
        with torch.no_grad():
            U, I = model.embeddings()
            ut = torch.as_tensor(u, device=DEV, dtype=torch.long)
            ct = torch.as_tensor(cand, device=DEV, dtype=torch.long)
            s = (U[ut].unsqueeze(1) * I[ct]).sum(-1)
            pick = s.argmax(1).cpu().numpy()
        j = cand[np.arange(bs), pick]
    else:
        j = rng.integers(0, n_items, bs)
    return (torch.as_tensor(u, device=DEV, dtype=torch.long),
            torch.as_tensor(i, device=DEV, dtype=torch.long),
            torch.as_tensor(j, device=DEV, dtype=torch.long))


def train(model, data, epochs=200, bs=2048, lr=1e-3, patience=10, seed=0,
          hard=0, eval_every=5, verbose=False, min_epochs=0):
    """Fit a model, selecting on validation Recall@20.

    When that metric never rises above zero, which happens for the two-stage
    InfoMax model on the sparsest benchmark, early stopping would end training
    almost immediately. `min_epochs` holds the run open and the checkpoint is
    then chosen by lowest training loss, so the baseline is not under-trained.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pairs = data['train']; ni = data['n_items']
    R = data['R'].tocsr()
    gt_va = _gt(data['valid'])
    if len(gt_va) > 2000:
        keys = sorted(gt_va)
        sel = np.random.default_rng(12345).choice(len(keys), 2000, replace=False)
        gt_va = {keys[k]: gt_va[keys[k]] for k in sel}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_batch = max(1, len(pairs) // bs)
    best, best_state, bad, hist = -1, None, 0, []
    best_loss, loss_state = float('inf'), None
    t0 = time.time()
    ep_used = 0
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for _ in range(n_batch):
            u, i, j = sample_batch(pairs, ni, bs, rng, R, model, hard)
            loss = model.loss(u, i, j)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        ep_used = ep
        if ep % eval_every == 0:
            model.eval()
            with torch.no_grad():
                U, I = model.embeddings()
                m = evaluate(emb_scorer(U, I), ni, R, None, gt_va, ks=(20,))
            v = float(np.mean(m['recall@20']))
            hist.append((ep, tot / n_batch, v))
            if verbose:
                print('   ep%3d loss=%.4f val_recall@20=%.4f' % (ep, tot / n_batch, v))
            mloss = tot / n_batch
            if mloss < best_loss:
                best_loss = mloss
                loss_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            if v > best + 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience and ep >= min_epochs:
                    break
    if best <= 0 and loss_state is not None:
        model.load_state_dict(loss_state)          # degenerate validation signal
    elif best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return dict(val=best, epochs=ep_used, seconds=time.time() - t0, hist=hist)


def _gt(pairs):
    d = {}
    for u, i in pairs:
        d.setdefault(int(u), []).append(int(i))
    return {u: np.array(v) for u, v in d.items()}


# ------------------------- non-neural baselines -------------------------

class Frozen:
    """Wraps precomputed score matrices in the (U, I) interface."""
    def __init__(self, U, I):
        self.U = U; self.I = I

    def embeddings(self):
        return self.U, self.I


def most_pop(R):
    pop = torch.tensor(np.asarray(R.sum(0)).ravel(), dtype=torch.float32, device=DEV)
    def scorer(ub):
        return pop.unsqueeze(0).expand(len(ub), -1).clone()
    return scorer


def item_knn(R, k=100):
    R = sp.csr_matrix(R, dtype=np.float32)
    n = np.sqrt(np.asarray(R.power(2).sum(0)).ravel()); n[n == 0] = 1
    Rn = R @ sp.diags(1.0 / n)
    S = (Rn.T @ Rn).tolil()
    S.setdiag(0)
    S = sp.csr_matrix(S)
    rows, cols, vals = [], [], []
    for r in range(S.shape[0]):
        a, b = S.indptr[r], S.indptr[r + 1]
        idx, v = S.indices[a:b], S.data[a:b]
        if len(v) > k:
            sel = np.argpartition(-v, k)[:k]; idx, v = idx[sel], v[sel]
        rows.append(np.full(len(idx), r)); cols.append(idx); vals.append(v)
    Sk = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=S.shape)
    Sk = sp.csr_matrix(Sk)
    def scorer(ub):
        P = np.asarray((R[ub] @ Sk.T).todense(), dtype=np.float32)
        return torch.tensor(P, device=DEV)
    return scorer
