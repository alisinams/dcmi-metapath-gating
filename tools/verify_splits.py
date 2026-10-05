"""Check that locally rebuilt splits are the ones behind every reported number.

code/build.py regenerates cache/<dataset>.pkl deterministically from the public raw data.
The splits are not redistributed (MovieLens terms forbid it), so this script compares the
rebuilt files with cache/split_fingerprints.json:
  * integer arrays (train / validation / test pairs) and the sparsity pattern of every
    meta-path adjacency must match exactly (SHA-256);
  * floating-point values (adjacency weights, content features, densities) are compared
    after rounding and only reported, because the last bits can change across library versions.

    python tools/verify_splits.py          # check cache/*.pkl
    python tools/verify_splits.py --write  # rewrite the fingerprint file (maintainers only)
"""
import hashlib, json, os, pickle, sys
import numpy as np
import scipy.sparse as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CACHE = os.path.join(ROOT, 'cache')
FP = os.path.join(CACHE, 'split_fingerprints.json')
DATASETS = ('ml100k', 'ml1m', 'lastfm', 'amazon')


def h(b):
    return hashlib.sha256(b).hexdigest()


def walk(obj, path=''):
    """Yield (path, kind, fingerprint); kind is 'exact' or 'float'."""
    if isinstance(obj, dict):
        for k in sorted(obj):
            yield from walk(obj[k], '%s/%s' % (path, k))
    elif sp.issparse(obj):
        m = sp.csr_matrix(obj)
        m.sum_duplicates(); m.sort_indices()
        yield path + ':shape', 'exact', str(m.shape)
        yield path + ':pattern', 'exact', h(m.indptr.astype(np.int64).tobytes() + m.indices.astype(np.int64).tobytes())
        yield path + ':values', 'float', h(np.round(m.data.astype(np.float64), 5).tobytes())
    elif isinstance(obj, np.ndarray):
        if obj.dtype.kind in 'iub':
            yield path, 'exact', h(np.ascontiguousarray(obj.astype(np.int64)).tobytes()) + ' ' + str(obj.shape)
        else:
            yield path, 'float', h(np.round(obj.astype(np.float64), 4).tobytes()) + ' ' + str(obj.shape)
    elif isinstance(obj, (float, np.floating)):
        yield path, 'float', '%.6g' % float(obj)
    else:
        yield path, 'exact', str(obj.item() if hasattr(obj, 'item') else obj)


def fingerprints(name):
    data = pickle.load(open(os.path.join(CACHE, name + '.pkl'), 'rb'))
    return {p: [k, v] for p, k, v in walk(data)}


def main():
    if '--write' in sys.argv:
        out = {d: fingerprints(d) for d in DATASETS if os.path.exists(os.path.join(CACHE, d + '.pkl'))}
        json.dump(out, open(FP, 'w', encoding='utf-8'), indent=1, sort_keys=True)
        print('wrote', FP, {d: len(v) for d, v in out.items()})
        return 0
    ref = json.load(open(FP, encoding='utf-8'))
    failed = False
    for d in DATASETS:
        p = os.path.join(CACHE, d + '.pkl')
        if not os.path.exists(p):
            print('%-7s missing (run code/build.py)' % d); failed = True; continue
        got = fingerprints(d)
        exact_bad = [k for k, (kind, v) in ref[d].items() if kind == 'exact' and got.get(k, [None, None])[1] != v]
        float_bad = [k for k, (kind, v) in ref[d].items() if kind == 'float' and got.get(k, [None, None])[1] != v]
        status = 'IDENTICAL' if not exact_bad and not float_bad else ('SPLITS IDENTICAL, floats differ' if not exact_bad else 'MISMATCH')
        print('%-7s %s' % (d, status))
        for k in exact_bad[:10]:
            print('   exact field differs:', k)
        for k in float_bad[:5]:
            print('   float field differs:', k)
        failed |= bool(exact_bad)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
