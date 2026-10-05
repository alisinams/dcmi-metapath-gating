"""Ablations: gate-criterion study and component study."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import run_one, RES, load_cfg, CFG
import run as RUN

GATE = {
    'DCMI (full)':   dict(prior='dcmi'),
    'DCMI, effect size': dict(prior='excess'),
    'DCMI, scale-aware': dict(prior='excess_scaled'),
    'raw MI':        dict(prior='mi'),
    'density prior': dict(prior='density'),
    'uniform':       dict(prior='uniform'),
    'free gate':     dict(prior='dcmi', gamma=0.0),
}

COMP = {
    'full':               dict(),
    'w/o bottleneck':     dict(use_ib=False),
    'w/o meta-paths':     dict(use_mp=False),
    'w/o content':        dict(use_content=False),
    'w/o hard negatives': dict(hard=0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', choices=['gate', 'comp'], required=True)
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    load_cfg()
    grid = GATE if a.study == 'gate' else COMP
    path = os.path.join(RES, 'ablation_%s_%s.jsonl' % (a.study, '_'.join(a.datasets)))
    done = set()
    if os.path.exists(path):
        for l in open(path):
            r = json.loads(l); done.add((r['dataset'], r['variant'], r['seed']))
    with open(path, 'a') as f:
        for dn in a.datasets:
            base = dict(RUN.CFG.get(dn, {}).get('MIDGaP', {}))
            for vn, over in grid.items():
                for s in range(a.seeds):
                    if (dn, vn, s) in done:
                        continue
                    cfg = dict(base); cfg.update(over)
                    r = run_one(dn, 'MIDGaP', s, over=cfg,
                                save_pref='abl_%s_%s_s%d' % (dn, vn.replace(' ', '').replace('/', ''), s))
                    r['variant'] = vn; r['cfg'] = {k: v for k, v in cfg.items()}
                    f.write(json.dumps(r) + '\n'); f.flush()
                    print('%-8s %-20s s%d  R@20=%.4f N@20=%.4f  %.0fs'
                          % (dn, vn, s, r['recall@20'], r['ndcg@20'], r['seconds']), flush=True)


if __name__ == '__main__':
    main()
