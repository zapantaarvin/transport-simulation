# Site scenarios (2026-11-14, synthetic; pass rule = every file on time)

50,000 files per site over Oct 1–30, stored as Parquet in `data/scenarios/`. Rates in MB/s. Sizes 0.1 MB – 15 GB, log-uniform for A and B
(every size range equally likely, so the median is about 40 MB and 1% of files are over 13 GB).
Make + run: `python src/scenarios.py make && python src/scenarios.py run` (about 25s total).

Each site folder has `0_prerun.png` (input only), `settings_grid.png` (threads × per-thread rate → % of
files on time, ✓ = every file on time), `1_load_vs_max.png`, and `2_latency_spread.png` + `4_queue_busiest_day.png`
+ `5_sla_heatmap.png` at the chosen 5-min setting. Full write-up with charts: `REPORT.html`.

**Pass rule:** a limit passes only when every file finishes within it (60s / 300s from request).

| Site | Traffic | Max | Busiest hour / 5 min | 1 min | 5 min |
|---|---|---|---|---|---|
| A | 24h, week 1 busier, very random | 1000 | 65% / 157% | no (best 97.59%) | no (best 99.51%) |
| B | 2-day burst, then shrinking random bursts | 500 | 246% / 304% | no (82.31%) | no (83.65%) |
| C | weekday 9–5, quiet weekends | 300 | 17% / 46% | 1 @ 300 only | 1 @ 100 |
| D | nightly 1:00–2:30 batch + month-end surge | 800 | 89% / 127% | no (98.91%) | 2 @ 500, 4 @ 200 |
| E | steady, 90% small + 10% huge files | 400 | 10% / 43% | 1 @ 400 only (outside grid) | 1 @ 200 |
| F | power outages on 4 random days, backlog rush after | 600 | 77% / 471% | no (96.33%) | no (97.09%) |

Bigger max rate (best of 4–64 threads × 200/300/500): A passes 5 min at ×1.5 and both at ×2;
B 5 min at ×3, both at ×4; D both at ×1.5 (4 @ 300); F 5 min at ×3, 1 min still 99.34% at ×4.

## Sizing from the concurrency curve (`python src/concurrency.py all`)

Each file is assumed to start at its request and run at rate r. Sweep +1 at start, −1 at finish → transfers
running over time (`6_concurrency.png`, `concurrency_sizing.csv`, `../concurrency_summary.csv`).
Peak = threads so nothing waits; peak × r must fit the max rate for a guaranteed pass; the simulator then
finds the real minimum. Same answers as the grid: C 1 @ 100, D 2 @ 500, E 2 @ 100 (4 guaranteed);
A, B, F fail because the curve goes over the pipe at bursts (0.06–0.42% of the time).

## What the scenarios teach

1. **1 min is set by the biggest file.** 15 GB needs 250 MB/s for a full minute with no wait, so 1 min needs a
   fast thread *and* room in the pipe at that moment.
2. **Average load lies.** Every site averages 2–5% of max. A's 5-minute peaks hit 157%, and B's burst hits 246% for a whole hour.
3. **A burst over the max rate can't be fixed by settings.** The grid plateaus (A, B, F). Fix: more max rate, or
   spread/queue the burst on purpose.
4. **More threads can make it worse.** C and D pass at 1–4 threads but have a few late files at 8+, because the
   threads split the pipe and a 15 GB file slows down.
5. **"1 thread" picks (C, E) are the bare minimum** under this model. In real life, keep at least 2–4 for
   redundancy; a stuck transfer would block everything else.
6. **Outages hurt in proportion to their length.** Site F (`site-f-events.csv` lists them):

   | Outage | Hours | Files held | Rush |
   |---|---|---|---|
   | Oct 2, 11:49 | 3.8 | 306 | 58 min |
   | Oct 19, 09:24 | 3.1 | 449 | 47 min |
   | Oct 22, 05:40 | 9.7 | 1,298 | 22 min |
   | Oct 27, 02:59 | 9.2 | 742 | 24 min |

   In the heatmap, outages show as gray (no requests) followed by a red hour (the rush).
   Request time = when the request reached NAPM after power returned. If your real data keeps the
   original creation time instead, those files would already be hours late and fail regardless.

## Caveats

- Synthetic traffic: shapes are my reading of the descriptions, not measured.
- The per-thread rates tested (25–500) are a grid, not known NAPM limits.
- Sizes: the literal uniform option (`make --sizes uniform`, median 7.5 GB) is much heavier; it hasn't been run yet.
