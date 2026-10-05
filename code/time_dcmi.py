"""Measure the offline cost of the DCMI estimator on the largest benchmark."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import load
from dcmi import dcmi

out = {}
for n in ['ml1m', 'lastfm', 'amazon']:
    d = load(n)
    t0 = time.time()
    dcmi(d['mp_u'], d['R'].tocsr(), 'u', d['valid'], n_null=40, seed=0)
    dcmi(d['mp_i'], d['R'].tocsr(), 'i', d['valid'], n_null=40, seed=0)
    out[n] = round(time.time() - t0, 1)
    print(n, out[n], 's', flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), '..', 'results', 'dcmi_timing.json'), 'w'), indent=1)
