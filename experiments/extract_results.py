import json, os, glob
import numpy as np

results_dir = r'c:\Cursor\Mocho\.cursor\skills\article-generator-v3\paper\experiments\results'

def bootstrap_ci(values, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=float)
    point = float(np.mean(arr))
    boots = [float(np.mean(rng.choice(arr, size=len(arr), replace=True))) for _ in range(n_boot)]
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return point, lo, hi

all_data = []

for f in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
    name = os.path.basename(f).replace('.json','')
    if name == 'summary':
        continue
    try:
        with open(f) as fh:
            obj = json.load(fh)
        if not isinstance(obj, dict) or 'results' not in obj:
            continue
        bench = obj.get('benchmark', '')
        cond = obj.get('condition', name)
        rows = obj['results']
        n = len(rows)
        if n < 10:
            continue

        if bench == 'mosaic':
            scores = [r['scc'] for r in rows]
            pt, lo, hi = bootstrap_ci(scores)
            all_data.append({'bench': bench, 'cond': cond, 'n': n, 'metric': 'SCC', 'mean': pt, 'lo': lo, 'hi': hi})
        elif bench == 'sysprompt':
            scores = [r['scc'] for r in rows]
            pt, lo, hi = bootstrap_ci(scores)
            all_data.append({'bench': bench, 'cond': cond, 'n': n, 'metric': 'SCC', 'mean': pt, 'lo': lo, 'hi': hi})
        elif bench == 'ifbench':
            scores = [r['instruction_accuracy'] for r in rows]
            pt, lo, hi = bootstrap_ci(scores)
            all_data.append({'bench': bench, 'cond': cond, 'n': n, 'metric': 'instr_acc', 'mean': pt, 'lo': lo, 'hi': hi})
        elif bench == 'ifeval':
            scores = [r.get('instruction_accuracy', 0) for r in rows]
            ps = [1.0 if r.get('follow_all', False) else 0.0 for r in rows]
            pt, lo, hi = bootstrap_ci(scores)
            pt2, lo2, hi2 = bootstrap_ci(ps)
            all_data.append({'bench': bench, 'cond': cond, 'n': n, 'metric': 'instr_acc', 'mean': pt, 'lo': lo, 'hi': hi, 'ps_mean': pt2, 'ps_lo': lo2, 'ps_hi': hi2})
        elif bench == 'toolsel':
            scores = [r.get('tool_accuracy', r.get('accuracy', 0)) for r in rows]
            pt, lo, hi = bootstrap_ci(scores)
            all_data.append({'bench': bench, 'cond': cond, 'n': n, 'metric': 'tool_acc', 'mean': pt, 'lo': lo, 'hi': hi})
        elif bench == 'followbench':
            scores = [r.get('score', r.get('scc', 0)) for r in rows]
            pt, lo, hi = bootstrap_ci(scores)
            all_data.append({'bench': bench, 'cond': cond, 'n': n, 'metric': 'score', 'mean': pt, 'lo': lo, 'hi': hi})
    except Exception as e:
        print(f"ERROR {name}: {e}")

for bench in ['mosaic', 'sysprompt', 'ifbench', 'ifeval', 'toolsel', 'followbench']:
    items = [d for d in all_data if d['bench'] == bench]
    if not items:
        continue
    items.sort(key=lambda x: x['mean'], reverse=True)
    print(f"\n{'='*70}")
    print(f" {bench.upper()} (metric: {items[0]['metric']})")
    print(f"{'='*70}")
    print(f" {'Condition':<20s} {'N':>5s}  {'Mean':>7s}  {'95% CI':>17s}  {'vs Base':>8s}")
    print(f" {'-'*20} {'-'*5}  {'-'*7}  {'-'*17}  {'-'*8}")
    base_val = None
    for d in items:
        if d['cond'] == 'baseline':
            base_val = d['mean']
    for d in items:
        ci = f"[{d['lo']:.3f}, {d['hi']:.3f}]"
        delta = f"+{(d['mean']-base_val)*100:.1f}pp" if base_val is not None and d['cond'] != 'baseline' else '-'
        extra = ''
        if 'ps_mean' in d:
            extra = f"  ps={d['ps_mean']:.3f} [{d['ps_lo']:.3f},{d['ps_hi']:.3f}]"
        print(f" {d['cond']:<20s} {d['n']:>5d}  {d['mean']:>7.4f}  {ci:>17s}  {delta:>8s}{extra}")
