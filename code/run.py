"""Experiment driver: trains every model on every dataset over multiple seeds."""
import os, sys, json, pickle, argparse, time
import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import DEV, BPRMF, LightGCN, SGL, SimGCL, XSimGCL, HDGCN, MIDGaP
import engine as E
from dcmi import dcmi, gate_prior

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE = os.path.join(ROOT, 'cache'); RES = os.path.join(ROOT, 'results')
os.makedirs(RES, exist_ok=True)

HP = {  # optimisation budget per dataset
    'ml100k': dict(epochs=200, bs=1024, eval_every=5, patience=4),
    'ml1m':   dict(epochs=200, bs=4096, eval_every=5, patience=4),
    'lastfm': dict(epochs=200, bs=1024, eval_every=5, patience=4),
    'amazon': dict(epochs=200, bs=8192, eval_every=5, patience=4),
}


def load(name):
    with open(os.path.join(CACHE, name + '.pkl'), 'rb') as f:
        return pickle.load(f)


def get_priors(d, kind='dcmi', n_null=40, seed=0):
    f = os.path.join(CACHE, d['name'] + '_dcmi.json')
    if os.path.exists(f):
        st = json.load(open(f))
    else:
        st = {'u': dcmi(d['mp_u'], d['R'].tocsr(), 'u', d['valid'], n_null=n_null, seed=seed),
              'i': dcmi(d['mp_i'], d['R'].tocsr(), 'i', d['valid'], n_null=n_null, seed=seed)}
        for side, rd in (('u', d['rd_u']), ('i', d['rd_i'])):
            for k in st[side]:
                st[side][k]['density'] = rd[k]          # HDGCN definition: raw meta-path density
        json.dump(st, open(f, 'w'), indent=1)
    return st, gate_prior(st['u'], kind), gate_prior(st['i'], kind)


def build_model(name, d, st, seed, **over):
    nu, ni, R = d['n_users'], d['n_items'], d['R'].tocsr()
    torch.manual_seed(seed); np.random.seed(seed)
    if name == 'BPR-MF':
        return BPRMF(nu, ni, **over), 0
    if name == 'LightGCN':
        return LightGCN(nu, ni, R, **over), 0
    if name == 'SGL':
        return SGL(nu, ni, R, **over), 0
    if name == 'SimGCL':
        return SimGCL(nu, ni, R, **over), 0
    if name == 'XSimGCL':
        return XSimGCL(nu, ni, R, **over), 0
    if name.startswith('HDGCN'):
        if name.endswith('+BPR'):
            over.setdefault('bpr', 1.0)
        return HDGCN(nu, ni, R, d['mp_u'], d['mp_i'], d['feat_u'], d['feat_i'],
                     d['rd_u'], d['rd_i'], **over), 0
    if name.startswith('MIDGaP'):
        kind = over.pop('prior', 'dcmi')
        pu = gate_prior(st['u'], kind); pi = gate_prior(st['i'], kind)
        hard = over.pop('hard', 8)
        m = MIDGaP(nu, ni, R, d['mp_u'], d['mp_i'], d['feat_u'], d['feat_i'], pu, pi, **over)
        return m, hard
    raise ValueError(name)


CFG = {}


def load_cfg():
    """Best validated configuration per (dataset, model), read straight from the tuning logs."""
    global CFG
    import glob
    best = {}
    for f in glob.glob(os.path.join(RES, 'tune_*.jsonl')):
        for l in open(f):
            try:
                r = json.loads(l)
            except ValueError:
                continue
            k = (r['dataset'], r['model'])
            if k not in best or r['val'] > best[k]['val']:
                best[k] = r
    CFG = {}
    for (d, m), r in best.items():
        CFG.setdefault(d, {})[m] = r['cfg']
    if 'ml1m' in CFG:                       # ML-100K inherits the ML-1M configuration
        for m, c in CFG['ml1m'].items():
            CFG.setdefault('ml100k', {}).setdefault(m, c)
    json.dump(CFG, open(os.path.join(RES, 'best_cfg.json'), 'w'), indent=1)


def run_one(dname, mname, seed, over=None, save_pref=None):
    if over is None:
        over = dict(CFG.get(dname, {}).get(mname, {}))
    d = load(dname)
    st = get_priors(d)[0] if mname.startswith('MIDGaP') else None
    hp = HP[dname]
    R = d['R'].tocsr()
    Rva = sp.csr_matrix((np.ones(len(d['valid']), np.float32),
                         (d['valid'][:, 0], d['valid'][:, 1])), shape=R.shape)
    gt = E._gt(d['test'])
    pop = np.asarray(R.sum(0)).ravel(); pop = pop / pop.max()
    t0 = time.time()
    if mname == 'MostPop':
        sc = E.most_pop(R); info = dict(epochs=0, seconds=0.0, val=0.0, params=0)
    elif mname == 'ItemKNN':
        sc = E.item_knn(R); info = dict(epochs=0, seconds=time.time() - t0, val=0.0, params=0)
    else:
        model, hard = build_model(mname, d, st, seed, **(over or {}))
        model = model.to(DEV)
        # the two-stage InfoMax model is held open for a minimum budget because its
        # validation ranking metric can be degenerate; see engine.train
        min_ep = 100 if mname.startswith('HDGCN') else 0
        info = E.train(model, d, epochs=hp['epochs'], bs=hp['bs'], patience=hp['patience'],
                       eval_every=hp['eval_every'], seed=seed, hard=hard, min_epochs=min_ep)
        info['params'] = int(sum(p.numel() for p in model.parameters()))
        with torch.no_grad():
            U, I = model.embeddings()
            if hasattr(model, 'gu'):          # record where the learned gate settled
                info['gate_u'] = dict(zip(d['mp_u'], torch.softmax(model.gu, 0).tolist()))
                info['gate_i'] = dict(zip(d['mp_i'], torch.softmax(model.gi, 0).tolist()))
            elif hasattr(model, 'wu'):
                info['gate_u'] = dict(zip(d['mp_u'], torch.softmax(model.wu, 0).tolist()))
                info['gate_i'] = dict(zip(d['mp_i'], torch.softmax(model.wi, 0).tolist()))
        sc = E.emb_scorer(U, I)
    m = E.evaluate(sc, d['n_items'], R, Rva, gt, pop=pop, want_lists=True)
    out = dict(dataset=dname, model=mname, seed=seed, **E.summarise(m),
               **{k: info[k] for k in ('epochs', 'seconds', 'val', 'params')})
    for k in ('gate_u', 'gate_i'):
        if k in info:
            out[k] = info[k]
    if save_pref:
        os.makedirs(os.path.join(RES, 'per_user_npz'), exist_ok=True)
        np.savez_compressed(os.path.join(RES, 'per_user_npz', save_pref + '.npz'),
                            users=m['users'], recall20=m['recall@20'], ndcg20=m['ndcg@20'],
                            recall10=m['recall@10'], ndcg10=m['ndcg@10'], top=m['top'])
    return out


MODELS = ['MostPop', 'ItemKNN', 'BPR-MF', 'LightGCN', 'SGL', 'SimGCL', 'XSimGCL', 'HDGCN', 'HDGCN+BPR', 'MIDGaP']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['ml1m', 'lastfm', 'amazon', 'ml100k'])
    ap.add_argument('--models', nargs='+', default=MODELS)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--out', default='main.jsonl')
    args = ap.parse_args()
    load_cfg()
    path = os.path.join(RES, args.out)
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done.add((r['dataset'], r['model'], r['seed']))
    with open(path, 'a') as f:
        for dn in args.datasets:
            for mn in args.models:
                seeds = [0] if mn in ('MostPop', 'ItemKNN') else list(range(args.seeds))
                for s in seeds:
                    if (dn, mn, s) in done:
                        continue
                    r = run_one(dn, mn, s, save_pref='%s_%s_s%d' % (dn, mn.replace('-', ''), s))
                    f.write(json.dumps(r) + '\n'); f.flush()
                    print('%-8s %-9s s%d  R@20=%.4f N@20=%.4f  cov=%.3f  %.0fs'
                          % (dn, mn, s, r['recall@20'], r['ndcg@20'], r['coverage'], r['seconds']), flush=True)


if __name__ == '__main__':
    main()
