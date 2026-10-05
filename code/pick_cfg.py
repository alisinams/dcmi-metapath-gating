"""Select the best validated configuration per (dataset, model) and freeze it."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import RES

best = {}
for f in glob.glob(os.path.join(RES, 'tune_*.jsonl')):
    for l in open(f):
        r = json.loads(l)
        k = (r['dataset'], r['model'])
        if k not in best or r['val'] > best[k]['val']:
            best[k] = r

cfg = {}
for (d, m), r in best.items():
    cfg.setdefault(d, {})[m] = r['cfg']

# ML-100K is used only for the protocol study; it inherits the ML-1M configuration
if 'ml1m' in cfg:
    cfg.setdefault('ml100k', {})
    for m, c in cfg['ml1m'].items():
        cfg['ml100k'].setdefault(m, c)

json.dump(cfg, open(os.path.join(RES, 'best_cfg.json'), 'w'), indent=1)
for d in sorted(cfg):
    for m in sorted(cfg[d]):
        v = best.get((d, m))
        print('%-8s %-10s %-32s val=%s' % (d, m, cfg[d][m], ('%.4f' % v['val']) if v else 'inherited'))
