"""Protocol study, uncertainty analysis, sensitivity sweep and MI-compression traces."""
import os, sys, json, argparse, time
import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import DEV, LightGCN, MIDGaP
import engine as E
import run as RUN
from run import load, get_priors, build_model, RES, HP


# --------------------- 1. sampled versus full-catalogue protocol ---------------------

@torch.no_grad()
def sampled_eval(U, I, train_csr, gt, n_neg=99, ks=(10, 20), seed=0):
    rng = np.random.default_rng(seed)
    n_items = I.shape[0]
    out = {k: [] for k in ks}
    for u, pos in gt.items():
        seen = set(train_csr.indices[train_csr.indptr[u]:train_csr.indptr[u + 1]].tolist()) | set(pos.tolist())
        neg = []
        while len(neg) < n_neg:
            c = rng.integers(0, n_items, n_neg)
            neg.extend([int(x) for x in c if int(x) not in seen][:n_neg - len(neg)])
        for p in pos:
            cand = np.array([p] + neg)
            s = (U[u] @ I[cand].t()).cpu().numpy()
            r = int((s > s[0]).sum())
            for k in ks:
                out[k].append((1.0, 1.0 / np.log2(r + 2)) if r < k else (0.0, 0.0))
    return {('hr@%d' % k): float(np.mean([a for a, _ in v])) for k, v in out.items()} | \
           {('ndcg@%d' % k): float(np.mean([b for _, b in v])) for k, v in out.items()}


def protocol_study(dname='ml100k', models=('BPR-MF', 'LightGCN', 'HDGCN', 'MIDGaP'), seeds=3):
    RUN.load_cfg()
    d = load(dname)
    R = d['R'].tocsr(); gt = E._gt(d['test'])
    Rva = sp.csr_matrix((np.ones(len(d['valid']), np.float32),
                         (d['valid'][:, 0], d['valid'][:, 1])), shape=R.shape)
    rows = []
    for mn in models:
        st = get_priors(d)[0] if mn.startswith('MIDGaP') else None
        for s in range(seeds):
            cfg = dict(RUN.CFG.get(dname, {}).get(mn, {}))
            m, hard = build_model(mn, d, st, s, **cfg)
            m = m.to(DEV)
            E.train(m, d, epochs=HP[dname]['epochs'], bs=HP[dname]['bs'],
                    patience=HP[dname]['patience'], eval_every=5, seed=s, hard=hard,
                    min_epochs=100 if mn.startswith('HDGCN') else 0)
            with torch.no_grad():
                U, I = m.embeddings()
            full = E.summarise(E.evaluate(E.emb_scorer(U, I), d['n_items'], R, Rva, gt, ks=(10, 20)))
            samp = sampled_eval(U, I, R, gt, seed=s)
            rows.append(dict(model=mn, seed=s,
                             full_hr10=full['recall@10'], full_hr20=full['recall@20'],
                             full_ndcg10=full['ndcg@10'], full_ndcg20=full['ndcg@20'],
                             samp_hr10=samp['hr@10'], samp_hr20=samp['hr@20'],
                             samp_ndcg10=samp['ndcg@10'], samp_ndcg20=samp['ndcg@20']))
            print(rows[-1], flush=True)
    json.dump(rows, open(os.path.join(RES, 'protocol_%s.json' % dname), 'w'), indent=1)


# --------------------- 2. uncertainty, calibration, selective recommendation ---------------------

@torch.no_grad()
def mc_scores(model, users, n_items, S=16):
    """Monte-Carlo predictive mean and variance over the variational posterior."""
    acc = torch.zeros(len(users), n_items, device=DEV)
    acc2 = torch.zeros_like(acc)
    for _ in range(S):
        (z,), _, _ = model.sample(1)
        Zu, Zi = z[:model.nu], z[model.nu:]
        sc = Zu[users] @ Zi.t()
        acc += sc; acc2 += sc * sc
    mean = acc / S
    var = (acc2 / S - mean * mean).clamp_min(0)
    return mean, var


def uncertainty_study(dname, seeds=3, S=16, topm=20):
    RUN.load_cfg()
    d = load(dname); st = get_priors(d)[0]
    R = d['R'].tocsr(); gt = E._gt(d['test'])
    Rva = sp.csr_matrix((np.ones(len(d['valid']), np.float32),
                         (d['valid'][:, 0], d['valid'][:, 1])), shape=R.shape)
    deg = np.asarray(R.sum(1)).ravel()
    out = []
    for s in range(seeds):
        cfg = dict(RUN.CFG.get(dname, {}).get('MIDGaP', {}))
        m, hard = build_model('MIDGaP', d, st, s, **cfg)
        m = m.to(DEV)
        E.train(m, d, epochs=HP[dname]['epochs'], bs=HP[dname]['bs'],
                patience=HP[dname]['patience'], eval_every=5, seed=s, hard=hard)
        users = sorted(gt)
        unc, nd, rec = [], [], []
        for a in range(0, len(users), 512):
            ub = users[a:a + 512]
            mean, var = mc_scores(m, ub, d['n_items'], S)
            for r, u in enumerate(ub):
                mean[r, R.indices[R.indptr[u]:R.indptr[u + 1]]] = -1e9
                mean[r, Rva.indices[Rva.indptr[u]:Rva.indptr[u + 1]]] = -1e9
            top = torch.topk(mean, topm, dim=1).indices
            unc.append(torch.gather(var, 1, top).mean(1).cpu().numpy())
            rc, nn, _ = E.rank_metrics(mean, [gt[u] for u in ub], topm)
            nd.append(nn); rec.append(rc)
        out.append(dict(seed=s, users=np.array(users).tolist(),
                        unc=np.concatenate(unc).tolist(),
                        ndcg=np.concatenate(nd).tolist(),
                        recall=np.concatenate(rec).tolist(),
                        deg=deg[np.array(users)].tolist()))
        print('uncertainty seed', s, 'done', flush=True)
    json.dump(out, open(os.path.join(RES, 'uncertainty_%s.json' % dname), 'w'))


# --------------------- 3. beta sensitivity + compression trace ---------------------

def sensitivity(dname, betas=(0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1), seeds=3):
    RUN.load_cfg()
    d = load(dname); st = get_priors(d)[0]
    R = d['R'].tocsr(); gt = E._gt(d['test'])
    Rva = sp.csr_matrix((np.ones(len(d['valid']), np.float32),
                         (d['valid'][:, 0], d['valid'][:, 1])), shape=R.shape)
    deg = np.asarray(R.sum(1)).ravel()
    rows = []
    for b in betas:
        for s in range(seeds):
            cfg = dict(RUN.CFG.get(dname, {}).get('MIDGaP', {})); cfg['beta'] = b
            cfg['use_ib'] = b > 0
            m, hard = build_model('MIDGaP', d, st, s, **cfg)
            m = m.to(DEV)
            E.train(m, d, epochs=HP[dname]['epochs'], bs=HP[dname]['bs'],
                    patience=HP[dname]['patience'], eval_every=5, seed=s, hard=hard)
            with torch.no_grad():
                U, I = m.embeddings()
                mu, lv = m.posterior()
                kl = float((0.5 * (mu.pow(2) + lv.exp() - 1 - lv).sum(-1)).mean())
            mm = E.evaluate(E.emb_scorer(U, I), d['n_items'], R, Rva, gt, want_lists=True)
            cold = np.array([deg[u] for u in sorted(gt)])
            q = np.quantile(cold, 0.2)
            rows.append(dict(beta=b, seed=s, recall20=float(np.mean(mm['recall@20'])),
                             ndcg20=float(np.mean(mm['ndcg@20'])),
                             cold_recall20=float(np.mean(mm['recall@20'][cold <= q])),
                             kl=kl, coverage=mm['coverage']))
            print(rows[-1], flush=True)
    json.dump(rows, open(os.path.join(RES, 'sensitivity_%s.json' % dname), 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['protocol', 'uncertainty', 'sensitivity'])
    ap.add_argument('--dataset', default='ml1m')
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    if a.what == 'protocol':
        protocol_study(a.dataset, seeds=a.seeds)
    elif a.what == 'uncertainty':
        uncertainty_study(a.dataset, seeds=a.seeds)
    else:
        sensitivity(a.dataset, seeds=a.seeds)
