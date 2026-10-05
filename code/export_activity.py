"""Training-window activity (interactions per user) of every benchmark, for the figures.

Writes results/train_activity.json so that the figures can be redrawn from the result
files alone; the splits themselves are rebuilt locally with build.py.
"""
import os, json, pickle
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE = os.path.join(ROOT, 'cache'); RES = os.path.join(ROOT, 'results')

if __name__ == '__main__':
    out = {}
    for d in ('ml100k', 'ml1m', 'lastfm', 'amazon'):
        p = os.path.join(CACHE, d + '.pkl')
        if os.path.exists(p):
            R = pickle.load(open(p, 'rb'))['R'].tocsr()
            out[d] = np.asarray(R.sum(1)).ravel().astype(int).tolist()
    json.dump(out, open(os.path.join(RES, 'train_activity.json'), 'w', encoding='utf-8'), separators=(',', ':'))
    print({d: len(v) for d, v in out.items()})
