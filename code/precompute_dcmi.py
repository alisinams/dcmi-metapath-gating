import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import load, get_priors
for n in sys.argv[1:]:
    d = load(n)
    st, pu, pi = get_priors(d)
    print(n, 'user-side prior', {k: round(v, 3) for k, v in pu.items()},
             'item-side prior', {k: round(v, 3) for k, v in pi.items()}, flush=True)
