"""Backtest: exact RULES.txt log score + rolling backtest of main.py's model vs persistence.

Usage: python3 tools/backtest.py [--data reads.json] [--sims N] [--min-hist K] [--gap-h H]
reads.json: list of {at, rows: [[rank, universe_id, playing, name], ...]}.
"""
import argparse
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import main as M  # noqa: E402

FLOOR = 1e-4
DEFAULT_DATA = '/workspace/make-fun/roblox-data/reads.json'


def score_read(fc, chart, places=10):
    """fc: {place: {uid: p}}; chart: list of universe ids in rank order. Sum of 10 floored logs."""
    total = 0.0
    for k in range(1, places + 1):
        if k > len(chart):
            break
        rows = fc.get(k, {})
        g = chart[k - 1]
        if g in rows:
            p = rows[g]
        else:
            unnamed = sum(1 for u in chart if u not in rows)
            p = (1 - sum(rows.values())) / unnamed if unnamed else 0.0
        total += math.log(max(p, FLOOR))
    return total


def baseline(chart, places=10):
    """House persistence forecast (starter/main.ts)."""
    fc = {}
    for k in range(1, places + 1):
        d = {chart[k - 1]: 0.5}
        if k == 1:
            d[chart[1]] = 0.3
        else:
            d[chart[k - 2]] = 0.15
            if k < len(chart):
                d[chart[k]] = 0.15
        fc[k] = d
    return fc


def to_fc(out):
    fc = {}
    for r in out['probabilities']:
        fc.setdefault(r['place'], {})[r['universe_id']] = r['p']
    return fc


def load_reads(path):
    import json
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    return [{'t': M.ts(r['at']), 'ids': [x[1] for x in sorted(r['rows'])],
             'play': {x[1]: x[2] for x in r['rows']}} for r in raw]


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')


def task_for(T, hours=72):
    return {'window': {'from': iso(T), 'until': iso(T + hours * 3600)}, 'places': 10}


def mean(xs):
    return sum(xs) / len(xs)


def practice(reads, sims):
    lo, hi = M.ts('2026-11-07T23:00:00Z'), M.ts('2026-11-10T23:00:00Z')
    hist = [r for r in reads if r['t'] < lo]
    ev = [r for r in reads if lo <= r['t'] < hi]
    base = mean([score_read(baseline(hist[-1]['ids']), r['ids']) for r in ev])
    fc = to_fc(M.forecast(task_for(lo), hist, sims=sims))
    mod = mean([score_read(fc, r['ids']) for r in ev])
    return len(hist), len(ev), base, mod


def rolling(reads, sims, min_hist, gap_h):
    res = []
    for i in range(min_hist - 1, len(reads)):
        T = reads[i]['t'] + gap_h * 3600
        ev = [r for r in reads if T <= r['t'] < T + 72 * 3600]
        if len(ev) < 3:
            continue
        hist = reads[:i + 1]
        fc = to_fc(M.forecast(task_for(T), hist, sims=sims))
        m = mean([score_read(fc, r['ids']) for r in ev])
        b = mean([score_read(baseline(hist[-1]['ids']), r['ids']) for r in ev])
        res.append((iso(T), len(ev), m, b))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DEFAULT_DATA)
    ap.add_argument('--sims', type=int, default=4000)
    ap.add_argument('--min-hist', type=int, default=8)
    ap.add_argument('--gap-h', type=float, default=2.1)
    a = ap.parse_args()
    reads = load_reads(a.data)
    nh, ne, base, mod = practice(reads, a.sims)
    print(f'practice: {nh} history reads, {ne} scored reads')
    print(f'  baseline {base:.6f}   model {mod:.6f}  (sims={a.sims})')
    res = rolling(reads, a.sims, a.min_hist, a.gap_h)
    for T, n, m, b in res:
        print(f'  window {T} reads={n:2d} model {m:9.3f} baseline {b:9.3f}')
    if res:
        print(f'rolling over {len(res)} windows: model {mean([r[2] for r in res]):.3f}'
              f'  baseline {mean([r[3] for r in res]):.3f}')


if __name__ == '__main__':
    main()
