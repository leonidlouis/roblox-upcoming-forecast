# Roblox Up-and-Coming top-10 forecaster

Entry for make.fun `roblox-up-and-coming-forecast-01`. Python 3.13, standard library only, no network,
no API keys, no model calls. `main.py` reads the chart history and writes `forecast.json`.

## Method

A Monte Carlo rank-diffusion model, fully deterministic (`random.Random(20261113)`, no clock):

1. Take every game on the last chart, with `x = log(rank + 2.75)`.
2. Daily drift `v = c0 + b_tr*rank_trend_24h + b_gr*player_growth_24h + b_age*max(age_days - 5.5, 0)
   + b_pl*log(players/median) + b_x*x` (fitted constants are in `PARAMS`). Games with `playing = null` count as 1 player; games missing from the 24 h reference read count as ranked past the chart end.
3. Student-t noise (nu = 4) scaled by `s0 * h^0.6 * (own 48 h rank volatility / 0.45)^0.5`, where `h` is the horizon in days, drawn uniformly over the scored window plus the gap since the last read. The window comes from `task.json`.
4. Poisson newcomers (prior rate blended with the history's recent newcomer rate), placed at log-rank
   `N(2.0 - 0.1*h, 0.55)`.
5. 40000 simulations: sort each, count how often each game holds each place. A small floor (3.25 %) is spread
   over all known games; the remainder is left unnamed for newcomers. Rows are rounded *down* so each place sums to at most 1; at most 100 rows per place.
6. `top10` is a greedy pick of the most frequent game per place, each game used once.

Inputs are validated: unusable reads are skipped, rows are de-duplicated, a missing window falls back to
`history_until` + 72 h, and very short histories (1-2 reads) work. Runtime is about 4 s.

## Run

    mkdir -p out && MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=out python3 main.py

Without the env vars it uses `/input` and `/output`.

## Tests and backtest

    python3 -m unittest discover -s tests -v
    python3 tools/backtest.py [--data reads.json] [--sims 4000]

`tools/backtest.py` re-implements the RULES.txt log score exactly (unnamed share split over the read's
un-named games, floor 0.0001) and reproduces the house baseline's practice score of -43.512039. It also runs
a rolling backtest: each archive read as the history end (window starts 2.1 h later, 72 h long, at least 8
reads of history) against persistence. The data file is a list of `{at, rows: [[rank, universe_id, playing, name], ...]}`.

Measured (archive of 48 reads):

| | model | persistence baseline |
|---|---|---|
| practice window (10 reads) | -32.72 (40000 sims; -32.81 at 4000) | -43.512039 |
| rolling, 38 windows | -34.77 (4000 sims) | -47.35 |

The model's parameters were fitted on this same archive, so these numbers are in-sample and optimistic.
