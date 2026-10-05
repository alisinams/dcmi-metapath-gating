"""Re-tune and re-run the two-stage InfoMax baseline under the minimum-epoch budget.

Rows produced before that fairness fix are discarded so no stale result survives.
Run only when no other experiment process is writing to the result files.
"""
import os, sys, glob, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, '..', 'results'))
DS = ['ml1m', 'lastfm', 'amazon']


def drop(path, pred):
    if not os.path.exists(path):
        return 0
    keep, n = [], 0
    for l in open(path, encoding='utf-8'):
        if not l.strip():
            continue
        r = json.loads(l)
        if pred(r):
            n += 1
        else:
            keep.append(l)
    open(path, 'w', encoding='utf-8').writelines(keep)
    return n


def sh(*a):
    print('>>', ' '.join(a), flush=True)
    subprocess.run([sys.executable, '-W', 'ignore'] + list(a), cwd=HERE, check=False)


if __name__ == '__main__':
    dropped = 0
    for f in glob.glob(os.path.join(RES, 'tune_*.jsonl')) + glob.glob(os.path.join(RES, 'main_*.jsonl')):
        dropped += drop(f, lambda r: r['model'].startswith('HDGCN'))
    for f in glob.glob(os.path.join(RES, 'per_user_npz', '*_HDGCN*_s*.npz')):
        os.remove(f); dropped += 1
    print('discarded %d stale HDGCN records' % dropped, flush=True)
    tag = {'ml1m': 'a', 'lastfm': 'b', 'amazon': 'c'}
    for d in DS:
        sh('tune.py', '--datasets', d, '--models', 'HDGCN', 'HDGCN+BPR', '--out', 'tune_%s.jsonl' % tag[d])
    for d in DS:
        sh('run.py', '--datasets', d, '--models', 'HDGCN', 'HDGCN+BPR',
           '--seeds', '5', '--out', 'main_%s.jsonl' % d)
    print('HDGCN refresh complete', flush=True)
