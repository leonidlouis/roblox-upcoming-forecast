"""Roblox Up-and-Coming top-10 forecaster: Monte Carlo rank diffusion.

Standard library only, deterministic (seeded random.Random, no clock).
Reads $MAKEFUN_INPUT_DIR (default /input), writes $MAKEFUN_OUTPUT_DIR (default /output).
"""
import json
import math
import os
import random
from datetime import datetime, timezone

PARAMS = dict(c0=0.4, b_tr=0.1263, b_gr=-0.44, b_age=0.05, A0=5.5, b_pl=-0.15, b_x=-0.18,
              s0=0.35, gam=0.6, nu=4.0, lam=1.2, mnew=2.0, snew=0.55, vnew=-0.1,
              floor=0.0325, coff=2.75, kvol=0.5, lam_ad=0.5)
SIMS = 40000
SEED = 20261113
TREND_H = 24.0      # lookback for rank trend / player growth
VOL_H = 48.0        # lookback for own rank volatility
MAX_NEW = 16        # cap on simulated newcomers per run
MAX_ROWS = 100      # rows per place allowed by the rules
DEFAULT_PLACES = 10


def ts(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def parse_read(o):
    """One history line -> {'t','ids','play'} or None if unusable."""
    if not isinstance(o, dict) or o.get('ok') is False:
        return None
    rows = o.get('up_and_coming')
    if not isinstance(rows, list) or not rows:
        return None
    try:
        t = ts(o['at'])
    except (KeyError, TypeError, ValueError):
        return None
    good = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        u, rk = _int(r.get('universe_id')), _num(r.get('rank'))
        if u is None or rk is None:
            continue
        good.append((rk, u, _num(r.get('playing'))))
    good.sort(key=lambda z: (z[0], z[1]))
    ids, play, seen = [], {}, set()
    for _, u, p in good:
        if u in seen:
            continue
        seen.add(u)
        ids.append(u)
        play[u] = p
    return {'t': t, 'ids': ids, 'play': play} if ids else None


def normalize_task(task):
    """Validate task.json; fall back to history_until + 72 h when the window is missing."""
    if not isinstance(task, dict):
        raise ValueError('task.json must be an object')
    w = task.get('window') if isinstance(task.get('window'), dict) else {}
    start = w.get('from') or task.get('history_until')
    if not start:
        raise ValueError('task.json has no window.from or history_until')
    t0 = ts(start)
    t1 = ts(w['until']) if w.get('until') else t0 + 72 * 3600
    if t1 <= t0:
        t1 = t0 + 72 * 3600
    places = _int(task.get('places')) or DEFAULT_PLACES
    out = dict(task)
    out['window'] = dict(w, **{'from': start, 'until': datetime.fromtimestamp(t1, timezone.utc).isoformat()})
    out['places'] = max(1, min(places, 10))
    out.setdefault('output', 'forecast.json')
    return out


def load(input_dir):
    with open(os.path.join(input_dir, 'task.json'), encoding='utf-8') as f:
        task = normalize_task(json.load(f))
    reads = []
    with open(os.path.join(input_dir, 'history.ndjson'), encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = parse_read(json.loads(line))
            except ValueError:
                r = None
            if r:
                reads.append(r)
    reads.sort(key=lambda r: r['t'])
    return task, reads


def ref_read(hist, hours):
    """Latest read at least `hours` before the last one, else the oldest read."""
    cut = hist[-1]['t'] - hours * 3600
    ref = None
    for r in hist:
        if r['t'] <= cut:
            ref = r
    return ref if ref is not None else hist[0]


def features(hist, T, P):
    """Per game on the last chart: (id, log-rank, trend, growth, age_days, log_players_rel, volatility)."""
    L = lambda r: math.log(r + P['coff'])
    last = hist[-1]
    t0 = hist[0]['t']
    n = len(last['ids'])
    first = {}
    for r in hist:
        for u in r['ids']:
            first.setdefault(u, r['t'])
    ref = ref_read(hist, TREND_H)
    dt = max((last['t'] - ref['t']) / 86400, 0.25)
    rref = {u: i + 1 for i, u in enumerate(ref['ids'])}
    past = [r for r in hist if r['t'] >= last['t'] - VOL_H * 3600]
    ppos = [{u: i + 1 for i, u in enumerate(r['ids'])} for r in past]
    plays = sorted(max(last['play'][u] or 1, 1) for u in last['ids'])
    m = len(plays)
    med = plays[m // 2] if m % 2 else 0.5 * (plays[m // 2 - 1] + plays[m // 2])
    out = []
    for i, u in enumerate(last['ids']):
        r = i + 1
        p = max(last['play'][u] or 1, 1)
        pp = ref['play'].get(u)
        # trend: rank movement per day; a game absent from the ref read counts as ranked past the chart
        tr = (L(r) - L(rref.get(u, n + 20))) / dt
        gr = math.log(p / pp) / dt if pp and pp > 0 else 1.0
        # games already on the chart at the first read have unknown age: assume ~4 days old
        age = (T - first[u]) / 86400 + (4.0 if first[u] == t0 else 0.0)
        seq = [(past[q]['t'], L(ppos[q].get(u, n + 15))) for q in range(len(past))]
        inc = [(seq[q + 1][1] - seq[q][1]) ** 2 / max((seq[q + 1][0] - seq[q][0]) / 86400, 0.05)
               for q in range(len(seq) - 1)]
        vol = math.sqrt(sum(inc) / len(inc)) if inc else 0.45
        out.append((u, L(r), max(min(tr, 2), -2), max(min(gr, 2), -2), age,
                    math.log(p / med), max(vol, 0.05)))
    return out


def newcomer_rate(hist, P):
    """Newcomers/day: blend of the fitted prior and the recent rate of games first seen in the top 25."""
    last = hist[-1]
    first, best = {}, {}
    for r in hist:
        for i, u in enumerate(r['ids']):
            first.setdefault(u, r['t'])
            best[u] = min(best.get(u, 999), i + 1)
    span = min(72 * 3600, last['t'] - hist[0]['t'])
    nnew = sum(1 for u, t in first.items()
               if t > last['t'] - span and t > hist[0]['t'] and best[u] <= 25)
    rate = nnew / max(span / 86400, 0.5)
    a = min(max(P['lam_ad'], 0), 1)
    return abs(P['lam']) * (1 - a) + a * rate * abs(P['lam']) / 1.2


def poisson(rng, lam):
    k, p, lim = 0, 1.0, math.exp(-lam)
    while True:
        p *= rng.random()
        if p <= lim:
            return k
        k += 1


def simulate(task, hist, P, sims, seed):
    """Return (ids, counts[place][game]) of place occupancy over `sims` runs."""
    T = ts(task['window']['from'])
    span_h = max((ts(task['window']['until']) - T) / 3600, 1.0)
    places = task['places']
    F = features(hist, T, P)
    G = len(F)
    ids = [f[0] for f in F]
    x = [f[1] for f in F]
    v = [P['c0'] + P['b_tr'] * f[2] + P['b_gr'] * f[3] + P['b_age'] * max(f[4] - P['A0'], 0)
         + P['b_pl'] * f[5] + P['b_x'] * f[1] for f in F]
    volm = [(f[6] / 0.45) ** min(max(P['kvol'], 0), 1.5) for f in F]
    lamv = newcomer_rate(hist, P)
    nu = min(max(P['nu'], 1.5), 60)
    gam = min(max(P['gam'], 0.2), 1.2)
    gap_h = max((T - hist[-1]['t']) / 3600, 0.0)
    rng = random.Random(seed)
    counts = [[0] * G for _ in range(places)]
    for _ in range(sims):
        h = (gap_h + rng.random() * span_h) / 24.0
        sc = abs(P['s0']) * h ** gam
        pos = []
        for g in range(G):
            z = rng.gauss(0, 1) / math.sqrt(rng.gammavariate(nu / 2, 2) / nu)
            pos.append((x[g] + v[g] * h + sc * volm[g] * z, g))
        for _k in range(min(poisson(rng, lamv * h), MAX_NEW)):
            pos.append((rng.gauss(P['mnew'] + P['vnew'] * h, abs(P['snew'])), -1))
        pos.sort()
        for k in range(min(places, len(pos))):
            g = pos[k][1]
            if g >= 0:
                counts[k][g] += 1
    return ids, counts


def forecast(task, hist, P=PARAMS, sims=SIMS, seed=SEED):
    places = normalize_task(task)['places']
    if not hist:
        return {'probabilities': [], 'top10': []}
    task = normalize_task(task)
    ids, counts = simulate(task, hist, P, sims, seed)
    G = len(ids)
    fl = min(max(P['floor'], 1e-4), 0.2)
    probs = []
    for k in range(places):
        row = [((1 - fl) * counts[k][g] / sims + fl / G, g) for g in range(G)]
        row.sort(key=lambda z: (-z[0], ids[z[1]]))
        rows = [(math.floor(p * 1e6) / 1e6, g) for p, g in row[:MAX_ROWS]]  # floor: sums never exceed 1
        for p, g in rows:
            probs.append({'universe_id': ids[g], 'place': k + 1, 'p': p})
    # top10: greedy assignment by place occupancy, each game used once
    taken, top = set(), []
    for k in range(places):
        cand = [g for g in range(G) if g not in taken]
        if not cand:
            break
        best = max(cand, key=lambda g: (counts[k][g], -g))
        taken.add(best)
        top.append(ids[best])
    # pad with the last chart's order if the chart is shorter than `places`
    for u in hist[-1]['ids']:
        if len(top) >= places:
            break
        if u not in top:
            top.append(u)
    return {'probabilities': probs, 'top10': top}


def main():
    inp = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    outd = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    task, hist = load(inp)
    if not hist:
        raise SystemExit('history.ndjson holds no usable read')
    fc = forecast(task, hist)
    os.makedirs(outd, exist_ok=True)
    with open(os.path.join(outd, task.get('output', 'forecast.json')), 'w', encoding='utf-8') as f:
        f.write(json.dumps(fc, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
