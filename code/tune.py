"""Validation-set grid search. Every method receives the same search budget."""
import os, sys, json, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import run_one, RES

GRID = {
    'LightGCN': [dict(layers=1), dict(layers=2), dict(layers=3)],
    'SGL':      [dict(ssl=0.05, drop=0.1), dict(ssl=0.5, drop=0.2)],
    'SimGCL':   [dict(ssl=0.05, eps=0.1), dict(ssl=0.2, eps=0.2)],
    'XSimGCL':  [dict(ssl=0.05, eps=0.1), dict(ssl=0.2, eps=0.2)],
    'BPR-MF':   [dict()],
    'HDGCN':    [dict(gamma=0.05), dict(gamma=0.2)],
    'HDGCN+BPR': [dict(gamma=0.2)],
    'MIDGaP':   [dict(beta=1e-4, gamma=0.1), dict(beta=1e-3, gamma=1.0), dict(beta=1e-2, gamma=1.0)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--models', nargs='+', default=list(GRID))
    ap.add_argument('--out', default='tune.jsonl')
    a = ap.parse_args()
    path = os.path.join(RES, a.out)
    done = {}
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done[(r['dataset'], r['model'], json.dumps(r['cfg'], sort_keys=True))] = r
    with open(path, 'a') as f:
        for dn in a.datasets:
            for mn in a.models:
                for cfg in GRID[mn]:
                    key = (dn, mn, json.dumps(cfg, sort_keys=True))
                    if key in done:
                        continue
                    r = run_one(dn, mn, 0, over=dict(cfg))
                    r['cfg'] = cfg
                    f.write(json.dumps(r) + '\n'); f.flush()
                    print('%-8s %-10s %-34s val=%.4f  test R@20=%.4f  %.0fs'
                          % (dn, mn, cfg, r['val'], r['recall@20'], r['seconds']), flush=True)


if __name__ == '__main__':
    main()
