"""Aggregate every result file into the manuscript tables and the numbers the text quotes."""
import os, sys, json, glob, pickle
import numpy as np
import scipy.sparse as sp
from scipy.stats import wilcoxon, spearmanr

WORK = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(WORK, 'results'); CACHE = os.path.join(WORK, 'cache')
DS = ['ml1m', 'lastfm', 'amazon']
NICE = {'ml100k': 'ML-100K', 'ml1m': 'ML-1M', 'lastfm': 'Last.FM', 'amazon': 'Amazon-VG'}
ORDER = ['MostPop', 'ItemKNN', 'BPR-MF', 'LightGCN', 'SGL', 'SimGCL', 'XSimGCL',
         'HDGCN', 'HDGCN+BPR', 'MIDGaP']
LABEL = {'MIDGaP': 'MIDGaP (ours)', 'HDGCN': 'HDGCN', 'HDGCN+BPR': 'HDGCN + ranking loss'}


def jl(p):
    return [json.loads(l) for l in open(p, encoding='utf-8')] if os.path.exists(p) else []


def jload(p):
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None


def fmt(m, s, nd=4, bold=False):
    t = ('%.' + str(nd) + 'f ± %.' + str(nd) + 'f') % (m, s)
    return '**%s**' % t if bold else t


def per_user(d, model, metric='ndcg20'):
    """Metric averaged over seeds, aligned per user."""
    fs = sorted(glob.glob(os.path.join(RES, 'per_user_npz', '%s_%s_s*.npz' % (d, model.replace('-', '')))))
    if not fs:
        return None, None
    acc = None
    for f in fs:
        z = np.load(f)
        u, v = z['users'], z[metric]
        o = np.argsort(u); u, v = u[o], v[o]
        acc = v if acc is None else acc + v
    return u, acc / len(fs)


def stats_table():
    rows = []
    for d in ['ml1m', 'lastfm', 'amazon', 'ml100k']:
        p = pickle.load(open(os.path.join(CACHE, d + '.pkl'), 'rb'))
        R = p['R'].tocsr()
        n = len(p['train'])
        deg = np.asarray(R.sum(1)).ravel()
        rows.append([NICE[d], '{:,}'.format(p['n_users']), '{:,}'.format(p['n_items']),
                     '{:,}'.format(n), '%.3f' % (100 * n / (p['n_users'] * p['n_items'])),
                     '%d / %d' % (len(p['mp_u']), len(p['mp_i'])),
                     '{:,}'.format(len(p['valid'])), '{:,}'.format(len(p['test'])),
                     p['split'], '%.0f' % np.median(deg)])
    return dict(n='2', caption='Benchmark statistics after filtering. The split column names the '
                'protocol; median activity is the median number of training interactions per user.',
                header=['Dataset', 'Users', 'Items', 'Train', 'Density (%)', 'Meta-paths (U/I)',
                        'Valid', 'Test', 'Split', 'Median activity'],
                rows=rows,
                note='Density is the fraction of user-item cells that carry a training interaction. '
                     'The meta-path column counts the user-side and item-side relations available '
                     'to the gate.')


def main_table(rows_main):
    body = []
    best = {}
    for d in DS:
        for k in ('recall@20', 'ndcg@20'):
            vals = {m: np.mean([r[k] for r in rows_main if r['dataset'] == d and r['model'] == m])
                    for m in ORDER if any(r['dataset'] == d and r['model'] == m for r in rows_main)}
            if vals:
                best[(d, k)] = max(vals, key=vals.get)
    sig = {}
    for d in DS:
        u0, v0 = per_user(d, 'MIDGaP')
        ps = {}
        for m in ORDER:
            if m == 'MIDGaP':
                continue
            u1, v1 = per_user(d, m)
            if v1 is None or v0 is None or len(v1) != len(v0):
                continue
            try:
                ps[m] = wilcoxon(v0, v1, zero_method='zsplit').pvalue
            except ValueError:
                ps[m] = 1.0
        # Holm correction within dataset
        for rank, (m, p) in enumerate(sorted(ps.items(), key=lambda x: x[1])):
            sig[(d, m)] = p * (len(ps) - rank) < 0.05
    for m in ORDER:
        row = [LABEL.get(m, m)]
        for d in DS:
            for k in ('recall@20', 'ndcg@20'):
                v = [r[k] for r in rows_main if r['dataset'] == d and r['model'] == m]
                if not v:
                    row.append('n/a'); continue
                mark = '*' if (m != 'MIDGaP' and sig.get((d, m), False)) else ''
                row.append(fmt(np.mean(v), np.std(v), 4, bold=(best.get((d, k)) == m)) + mark)
        body.append(row)
    hdr = ['Method']
    for d in DS:
        hdr += ['%s R@20' % NICE[d], '%s NDCG@20' % NICE[d]]
    return dict(n='3', caption='Full-catalogue ranking accuracy, mean ± standard deviation over five '
                'seeds. Best value per column in bold. An asterisk marks a difference from MIDGaP that '
                'is significant under a two-sided Wilcoxon signed-rank test on per-user NDCG@20 with '
                'Holm correction within each dataset (α = 0.05).',
                header=hdr, rows=body,
                note='Deterministic methods (MostPop, ItemKNN) are run once and reported with zero '
                     'dispersion.')


def protocol_table():
    rows_p = jload(os.path.join(RES, 'protocol_ml100k.json')) or []
    body = []
    for m in ['BPR-MF', 'LightGCN', 'HDGCN', 'MIDGaP']:
        r = [x for x in rows_p if x['model'] == m]
        if not r:
            continue
        body.append([LABEL.get(m, m),
                     fmt(np.mean([x['samp_hr10'] for x in r]), np.std([x['samp_hr10'] for x in r]), 4),
                     fmt(np.mean([x['samp_hr20'] for x in r]), np.std([x['samp_hr20'] for x in r]), 4),
                     fmt(np.mean([x['full_hr10'] for x in r]), np.std([x['full_hr10'] for x in r]), 4),
                     fmt(np.mean([x['full_hr20'] for x in r]), np.std([x['full_hr20'] for x in r]), 4),
                     '%.1f' % (np.mean([x['samp_hr20'] for x in r]) / max(np.mean([x['full_hr20'] for x in r]), 1e-9))])
    return dict(n='4', caption='The same models on ML-100K under the sampled protocol used by the '
                'earlier heterogeneous-graph literature (one positive against 99 sampled negatives) '
                'and under full-catalogue ranking. The last column is the inflation factor at rank 20.',
                header=['Method', 'HR@10 (sampled)', 'HR@20 (sampled)', 'HR@10 (full)',
                        'HR@20 (full)', 'Inflation'],
                rows=body, note='Three seeds; leave-one-out split by timestamp.')


def metapath_table():
    body = []
    for d in ['ml1m', 'lastfm', 'amazon']:
        st = jload(os.path.join(CACHE, '%s_dcmi.json' % d))
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dcmi import gate_prior
        for side in ('u', 'i'):
            pd_ = gate_prior(st[side], 'density'); pz = gate_prior(st[side], 'dcmi')
            for k, v in st[side].items():
                body.append([NICE[d], k, '%.3f' % v['density'], '%.4f' % v['mi'],
                             '%.4f ± %.4f' % (v['null_mu'], v['null_sd']),
                             '%+.1f' % v['z'], '%.3f' % pd_[k], '%.3f' % pz[k]])
    return dict(n='5', caption='Meta-path statistics. Raw mutual information is compared against a '
                'degree-preserving null ensemble of 40 configuration-model surrogates; the z-score is '
                'the excess in units of the null standard deviation. The last two columns are the gate '
                'priors the density criterion and DCMI produce.',
                header=['Dataset', 'Meta-path', 'Density', 'MI (nats)', 'Null MI', 'z',
                        'π density', 'π DCMI'],
                rows=body,
                note='Density is the edge density of the meta-path subgraph, rescaled to the unit '
                     'interval within each node side. A negative z marks a relation that predicts '
                     'held-out interactions less well than a randomly rewired graph of the same '
                     'density. The two prior columns sum to one within each node side.')


def ablation_table():
    g = sum((jl(f) for f in sorted(glob.glob(os.path.join(RES, 'ablation_gate*.jsonl')))), [])
    c = sum((jl(f) for f in sorted(glob.glob(os.path.join(RES, 'ablation_comp*.jsonl')))), [])
    body = []
    for tag, rows_, order in (('Gate criterion', g, ['DCMI (full)', 'raw MI', 'density prior', 'uniform', 'free gate']),
                              ('Component', c, ['full', 'w/o bottleneck', 'w/o meta-paths', 'w/o content', 'w/o hard negatives'])):
        for v in order:
            row = ['%s: %s' % (tag, v)]
            for d in DS:
                x = [r['ndcg@20'] for r in rows_ if r['dataset'] == d and r['variant'] == v]
                b = [r['ndcg@20'] for r in rows_ if r['dataset'] == d and r['variant'] == order[0]]
                if not x:
                    row += ['n/r', 'n/r']; continue
                row.append('%.4f ± %.4f' % (np.mean(x), np.std(x)))
                row.append('%+.1f' % (100 * (np.mean(x) / np.mean(b) - 1)) if b else 'n/r')
            body.append(row)
    hdr = ['Variant']
    for d in DS:
        hdr += ['%s NDCG@20' % NICE[d], '%s Δ (%%)' % NICE[d]]
    return dict(n='6', caption='Ablations over three seeds. The upper block replaces the criterion that '
                'anchors the meta-path gate; the lower block removes one component at a time. Δ is the '
                'relative change against the first row of each block.',
                header=hdr, rows=body,
                note='The first row of each block is the same configuration, run once in each '
                     'study, so the gap between them measures run-to-run variability rather '
                     'than any effect.')


def beyond_table(rows_main):
    body = []
    for m in ORDER:
        row = [LABEL.get(m, m)]
        for d in DS:
            for k in ('coverage', 'arp', 'gini'):
                v = [r[k] for r in rows_main if r['dataset'] == d and r['model'] == m and k in r]
                row.append('%.3f' % np.mean(v) if v else 'n/r')
        e = [r for r in rows_main if r['dataset'] == 'ml1m' and r['model'] == m]
        row.append('%.0f' % np.mean([x['seconds'] for x in e]) if e else 'n/r')
        row.append('{:,}'.format(int(np.mean([x['params'] for x in e]))) if e else 'n/r')
        body.append(row)
    hdr = ['Method']
    for d in DS:
        hdr += ['%s cov.' % NICE[d], '%s ARP' % NICE[d], '%s Gini' % NICE[d]]
    hdr += ['ML-1M train (s)', 'Params']
    return dict(n='7', caption='Catalogue coverage, average recommendation popularity and the Gini '
                'index of the exposure distribution over the top-20 lists, together with the training '
                'cost on ML-1M. Lower ARP and Gini indicate less concentration on head items.',
                header=hdr, rows=body,
                note='Training time is wall-clock on a single NVIDIA RTX 3090 including early '
                     'stopping; parameter counts exclude frozen content features.')


def numbers(rows_main):
    N = {}

    def mean(d, m, k):
        v = [r[k] for r in rows_main if r['dataset'] == d and r['model'] == m]
        return float(np.mean(v)) if v else float('nan')

    for d in DS + ['ml100k']:
        for m in ORDER:
            for k, tag in (('recall@20', 'R20'), ('ndcg@20', 'N20'), ('recall@10', 'R10'), ('ndcg@10', 'N10')):
                N['%s_%s_%s' % (tag, m.replace('-', '').replace('+', 'p'), d)] = round(mean(d, m, k), 4)
    for d in DS + ['ml100k']:
        for m in ORDER:
            v = [r['coverage'] for r in rows_main if r['dataset'] == d and r['model'] == m]
            if v:
                N['cov_%s_%s' % (m.replace('-', '').replace('+', 'p'), d)] = round(float(np.mean(v)), 3)
    # how close popularity ranking comes to the best learned method
    for d in DS:
        learned = [(m, mean(d, m, 'ndcg@20')) for m in ORDER if m not in ('MostPop', 'ItemKNN')]
        learned = [x for x in learned if not np.isnan(x[1])]
        pop = mean(d, 'MostPop', 'ndcg@20')
        if learned and not np.isnan(pop):
            b = max(learned, key=lambda x: x[1])[1]
            N['popgap_%s' % d] = round(100 * (pop / b - 1), 1)
    # relative gain of MIDGaP over the strongest baseline per dataset
    for d in DS:
        base = [(m, mean(d, m, 'ndcg@20')) for m in ORDER if m != 'MIDGaP']
        base = [b for b in base if not np.isnan(b[1])]
        if base:
            bm, bv = max(base, key=lambda x: x[1])
            N['best_base_%s' % d] = bm
            N['gain_%s' % d] = round(100 * (mean(d, 'MIDGaP', 'ndcg@20') / bv - 1), 1)
    # cold-start quintile gain
    for d in DS:
        p = pickle.load(open(os.path.join(CACHE, d + '.pkl'), 'rb'))
        deg = np.asarray(p['R'].tocsr().sum(1)).ravel()
        u0, v0 = per_user(d, 'MIDGaP')
        if v0 is None:
            continue
        q = np.quantile(deg[u0], 0.2)
        cold = deg[u0] <= q
        N['cold_share_%s' % d] = round(float(cold.mean()), 3)
        for m in ('LightGCN', 'XSimGCL', 'HDGCN'):
            u1, v1 = per_user(d, m)
            if v1 is None:
                continue
            N['coldgain_%s_%s' % (m, d)] = round(100 * (v0[cold].mean() / max(v1[cold].mean(), 1e-9) - 1), 1)
    tm = jload(os.path.join(RES, 'dcmi_timing.json'))
    if tm:
        N['dcmi_seconds'] = int(round(max(tm.values())))
    # protocol inflation
    pr = jload(os.path.join(RES, 'protocol_ml100k.json')) or []
    for m in ('HDGCN', 'MIDGaP', 'LightGCN', 'BPR-MF'):
        r = [x for x in pr if x['model'] == m]
        if r:
            k = m.replace('-', '')
            s20 = float(np.mean([x['samp_hr20'] for x in r]))
            f20 = float(np.mean([x['full_hr20'] for x in r]))
            N['samp_hr20_%s' % k] = round(s20, 4)
            N['full_hr20_%s' % k] = round(f20, 4)
            N['infl_%s' % k] = round(s20 / max(f20, 1e-9), 1)
    # uncertainty
    un = jload(os.path.join(RES, 'uncertainty_ml1m.json'))
    if un:
        rs, cs = [], []
        for r in un:
            rs.append(spearmanr(r['unc'], r['deg']).statistic)
            u = np.array(r['unc']); nd = np.array(r['ndcg']); o = np.argsort(u)
            cs.append(nd[o[:int(0.5 * len(o))]].mean() / nd.mean())
        N['unc_spearman'] = round(float(np.mean(rs)), 3)
        N['unc_spearman_sd'] = round(float(np.std(rs)), 3)
        N['selective_delta50'] = round(100 * (float(np.mean(cs)) - 1), 1)
        # does variance order users by how well they are served, and does it survive activity control?
        rq, rw = [], []
        for r in un:
            u = np.array(r['unc']); nd = np.array(r['ndcg']); dg = np.array(r['deg'])
            rq.append(spearmanr(u, nd).statistic)
            cut = np.quantile(dg, np.linspace(.1, .9, 9))
            dec = np.digitize(dg, cut)
            per = [spearmanr(u[dec == d], nd[dec == d]).statistic for d in range(10) if (dec == d).sum() > 30]
            per = [x for x in per if x == x]
            if per:
                rw.append(float(np.mean(per)))
        N['unc_ndcg_rho'] = round(float(np.mean(rq)), 3)
        N['unc_ndcg_rho_within'] = round(float(np.mean(rw)), 3)
        users = np.array(un[0]['users']); unc = np.mean([r['unc'] for r in un], 0)
        ndm = np.mean([r['ndcg'] for r in un], 0)
        ub, vb = per_user('ml1m', 'LightGCN')
        if vb is not None:
            sel = np.isin(ub, users)
            N['routing_corr'] = round(float(spearmanr(unc, ndm - vb[sel]).statistic), 3)
    # where each anchor drives the learned gate on the collaborative item-side route
    from dcmi import gate_prior
    g_all = sum((jl(f) for f in sorted(glob.glob(os.path.join(RES, 'ablation_gate*.jsonl')))), [])
    tag = {'DCMI (full)': 'dcmi', 'raw MI': 'mi', 'density prior': 'dens',
           'uniform': 'unif', 'free gate': 'free'}
    for d in DS:
        fr = [r['gate_i'] for r in g_all
              if r['dataset'] == d and r['variant'] == 'free gate' and 'gate_i' in r]
        if not fr:
            continue
        mean = {k: np.mean([x[k] for x in fr]) for k in fr[0]}
        key = max(mean, key=mean.get)
        N['task_route_%s' % d] = key
        for v, t in tag.items():
            w = [r['gate_i'][key] for r in g_all
                 if r['dataset'] == d and r['variant'] == v and 'gate_i' in r]
            if w:
                N['gate_%s_%s' % (t, d)] = round(float(np.mean(w)), 3)
    # gate ablation: every variant relative to the DCMI anchor
    g = sum((jl(f) for f in sorted(glob.glob(os.path.join(RES, 'ablation_gate*.jsonl')))), [])
    vtag = {'DCMI (full)': 'dcmi', 'DCMI, effect size': 'eff', 'DCMI, scale-aware': 'scaled',
            'raw MI': 'mi', 'density prior': 'dens', 'uniform': 'unif', 'free gate': 'free'}
    for d in DS:
        a = [r['ndcg@20'] for r in g if r['dataset'] == d and r['variant'] == 'DCMI (full)']
        if not a:
            continue
        N['gateabs_dcmi_%s' % d] = round(float(np.mean(a)), 4)
        for v, t in vtag.items():
            b = [r['ndcg@20'] for r in g if r['dataset'] == d and r['variant'] == v]
            if b:
                N['gatedelta_%s_%s' % (t, d)] = round(100 * (np.mean(b) / np.mean(a) - 1), 1)
        N['dens_drop_%s' % d] = -N.get('gatedelta_dens_%s' % d, 0.0)
    # component ablation relative to the full model
    c = sum((jl(f) for f in sorted(glob.glob(os.path.join(RES, 'ablation_comp*.jsonl')))), [])
    ctag = {'w/o bottleneck': 'ib', 'w/o meta-paths': 'mp', 'w/o content': 'cnt',
            'w/o hard negatives': 'hn'}
    for d in DS:
        a = [r['ndcg@20'] for r in c if r['dataset'] == d and r['variant'] == 'full']
        if not a:
            continue
        for v, t in ctag.items():
            b = [r['ndcg@20'] for r in c if r['dataset'] == d and r['variant'] == v]
            if b:
                N['compdrop_%s_%s' % (t, d)] = round(100 * (1 - np.mean(b) / np.mean(a)), 1)
    # activity-quintile comparison against LightGCN
    for d in DS:
        p = pickle.load(open(os.path.join(CACHE, d + '.pkl'), 'rb'))
        deg = np.asarray(p['R'].tocsr().sum(1)).ravel()
        u0, v0 = per_user(d, 'MIDGaP'); u1, v1 = per_user(d, 'LightGCN')
        if v0 is None or v1 is None:
            continue
        q = np.quantile(deg[u0], [0, .2, .4, .6, .8, 1.0]); q[-1] += 1
        idx = np.clip(np.digitize(deg[u0], q[1:-1]), 0, 4)
        for gq in (0, 4):
            m0, m1 = v0[idx == gq].mean(), v1[idx == gq].mean()
            N['q%d_gain_%s' % (gq + 1, d)] = round(100 * (m0 / max(m1, 1e-12) - 1), 1)
    # The first row of each ablation block is the same configuration, run once in the
    # gate study and once in the component study.  The gap between the two is a direct
    # measure of run-to-run variability, which early stopping makes large on ML-1M.
    g = sum((jl(f) for f in sorted(glob.glob(os.path.join(RES, 'ablation_gate*.jsonl')))), [])
    c = sum((jl(f) for f in sorted(glob.glob(os.path.join(RES, 'ablation_comp*.jsonl')))), [])
    eps = []
    for d in DS:
        a = [r for r in g if r['dataset'] == d and r['variant'] == 'DCMI (full)']
        b = [r for r in c if r['dataset'] == d and r['variant'] == 'full']
        if not a or not b:
            continue
        ma, mb = np.mean([r['ndcg@20'] for r in a]), np.mean([r['ndcg@20'] for r in b])
        N['rerun_gap_' + d] = round(abs(100 * (ma / mb - 1)), 1)
        if d == 'ml1m':
            eps = [r['epochs'] for r in a + b]
    if eps:
        N['rerun_ep_lo'] = int(min(eps)); N['rerun_ep_hi'] = int(max(eps))
    # magnitudes, so the prose can choose its own direction word
    for k in [k for k in N if isinstance(N[k], float)]:
        N['abs_' + k] = round(abs(N[k]), 1)
    return N


def main():
    rows_main = []
    for f in sorted(glob.glob(os.path.join(RES, 'main*.jsonl'))):
        rows_main += jl(f)
    T = {}
    T['tab2'] = stats_table()
    if rows_main:
        T['tab3'] = main_table(rows_main)
        T['tab7'] = beyond_table(rows_main)
    T['tab4'] = protocol_table()
    T['tab5'] = metapath_table()
    T['tab6'] = ablation_table()
    T['tab1'] = json.load(open(os.path.join(RES, 'tab1_taxonomy.json'), encoding='utf-8'))
    json.dump(T, open(os.path.join(RES, 'tables.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    N = numbers(rows_main) if rows_main else {}
    json.dump(N, open(os.path.join(RES, 'numbers.json'), 'w', encoding='utf-8'), indent=1)
    print('tables:', sorted(T), '  numbers:', len(N))


if __name__ == '__main__':
    main()
