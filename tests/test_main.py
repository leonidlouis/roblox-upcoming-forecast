import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import main as M  # noqa: E402
import backtest as B  # noqa: E402

TASK = {"history_until": "2026-11-13T06:00:00.000Z",
        "window": {"from": "2026-11-13T06:00:00.000Z", "until": "2026-11-16T06:00:00.000Z",
                   "slot_minutes": 60, "slots": 72},
        "places": 10, "output": "forecast.json"}


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f)


def make_read(at, n=30, shuffle=0, playing=True, extra=True, source='makefun/roblox-chart-recorder'):
    ids = list(range(1000, 1000 + n))
    if shuffle:
        ids = ids[shuffle:] + ids[:shuffle]
    rows = [{"rank": i + 1, "universe_id": u, "root_place_id": u, "name": "g%d" % u,
             "playing": (500 + 10 * (n - i)) if playing else None,
             "up_votes": None, "down_votes": None, "genre": None} for i, u in enumerate(ids)]
    o = {"at": at, "source": source, "filters": None, "ok": True, "up_and_coming": rows}
    if extra:
        o.update(top_trending=None, top_playing_now=None, details=None, responses=[])
    return o


def run_dir(reads, task=TASK, sims=None):
    with tempfile.TemporaryDirectory() as d:
        inp, out = os.path.join(d, 'in'), os.path.join(d, 'out')
        os.makedirs(inp)
        write_json(os.path.join(inp, 'task.json'), task)
        with open(os.path.join(inp, 'history.ndjson'), 'w') as f:
            for r in reads:
                f.write(json.dumps(r) + '\n')
        t, h = M.load(inp)
        return M.forecast(t, h, sims=sims or 300)


def check_schema(tc, fc, task=TASK):
    tc.assertEqual(set(fc), {'probabilities', 'top10'})
    tc.assertEqual(len(fc['top10']), 10)
    tc.assertEqual(len(set(fc['top10'])), 10)
    sums, counts, seen = {}, {}, set()
    for r in fc['probabilities']:
        tc.assertEqual(set(r), {'universe_id', 'place', 'p'})
        tc.assertIsInstance(r['universe_id'], int)
        tc.assertTrue(1 <= r['place'] <= task['places'])
        tc.assertTrue(0 <= r['p'] <= 1)
        tc.assertNotIn((r['universe_id'], r['place']), seen)
        seen.add((r['universe_id'], r['place']))
        sums[r['place']] = sums.get(r['place'], 0) + r['p']
        counts[r['place']] = counts.get(r['place'], 0) + 1
    for k in sums:
        tc.assertLessEqual(sums[k], 1.0)
        tc.assertLessEqual(counts[k], 100)
    tc.assertLess(len(json.dumps(fc)), 1048576)


class TestForecast(unittest.TestCase):
    def test_multi_read(self):
        reads = [make_read("2026-11-12T%02d:00:00.000Z" % h, shuffle=h % 3) for h in range(0, 24, 2)]
        check_schema(self, run_dir(reads))

    def test_single_read(self):
        check_schema(self, run_dir([make_read("2026-11-12T10:00:00.000Z")]))

    def test_two_reads(self):
        check_schema(self, run_dir([make_read("2026-11-12T00:00:00.000Z"),
                                    make_read("2026-11-12T08:00:00.000Z", shuffle=2)]))

    def test_null_playing_and_missing_extras(self):
        reads = [make_read("2026-11-12T00:00:00.000Z", playing=False, extra=False),
                 make_read("2026-11-12T01:00:00.000Z", playing=False)]
        check_schema(self, run_dir(reads))

    def test_game_missing_from_reference_read(self):
        a = make_read("2026-11-11T00:00:00.000Z", n=30)
        b = make_read("2026-11-12T00:00:00.000Z", n=40)
        check_schema(self, run_dir([a, b]))

    def test_determinism(self):
        reads = [make_read("2026-11-12T%02d:00:00.000Z" % h, shuffle=h % 4) for h in range(6)]
        self.assertEqual(run_dir(reads), run_dir(reads))

    def test_short_chart_pads_top10(self):
        check_schema(self, run_dir([make_read("2026-11-12T00:00:00.000Z", n=12)]))


class TestEndToEnd(unittest.TestCase):
    def test_cli_writes_file(self):
        with tempfile.TemporaryDirectory() as d:
            inp, out = os.path.join(d, 'in'), os.path.join(d, 'out')
            os.makedirs(inp)
            write_json(os.path.join(inp, 'task.json'), TASK)
            with open(os.path.join(inp, 'history.ndjson'), 'w') as f:
                f.write(json.dumps(make_read("2026-11-12T00:00:00.000Z")) + '\n')
            env = dict(os.environ, MAKEFUN_INPUT_DIR=inp, MAKEFUN_OUTPUT_DIR=out)
            outs = []
            for _ in range(2):
                subprocess.run([sys.executable, os.path.join(ROOT, 'main.py')], env=env, check=True)
                with open(os.path.join(out, 'forecast.json'), 'rb') as f:
                    outs.append(f.read())
            self.assertEqual(outs[0], outs[1])
            check_schema(self, json.loads(outs[0]))


class TestScorer(unittest.TestCase):
    def test_baseline_practice_value(self):
        if not os.path.exists(B.DEFAULT_DATA):
            self.skipTest('backtest data not available')
        reads = B.load_reads(B.DEFAULT_DATA)
        _, _, base, _ = B.practice(reads, 50)
        self.assertAlmostEqual(base, -43.512039, places=6)

    def test_scorer_floor_and_unnamed_share(self):
        chart = list(range(1, 21))
        self.assertAlmostEqual(B.score_read({k: {k: 1.0} for k in range(1, 11)}, chart), 0.0)
        # nothing named: each place spreads 1/20 over the chart
        import math
        self.assertAlmostEqual(B.score_read({}, chart), 10 * math.log(1 / 20))
        # named wrong game with p=1: right game gets 0 -> floor
        self.assertAlmostEqual(B.score_read({1: {99: 1.0}}, chart),
                               math.log(1e-4) + 9 * math.log(1 / 20))


if __name__ == '__main__':
    unittest.main()
