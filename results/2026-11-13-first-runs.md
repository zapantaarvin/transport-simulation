# First runs, 2026-11-13 (fake data)

> Historical log. These runs graded pass as p99 ≤ limit. Since 2026-11-14 the pass rule is
> **every file on time** (no p90/p95/p99), so the PASS/FAIL calls below are out of date.

Fake data: lognormal file sizes (median 20 MB, p99 about 326 MB), busier 9am–5pm,
30 days. "Real" speed baked into the fake start/end times was about 38 MB/s per transfer.

## 1. Set the speed from real data, not the spec (1k rows)

The sim at 50 MB/s gave p99 ≈ 6s, but the start/end times show 38 MB/s with p99 = 25s.
Lesson: set `--thread-mbps` from `size / (end − start)`.

## 2. Thread sweep, 1M rows, max 80 MB/s, 10 MB/s per thread

Load = 15.9 MB/s = 20% of max. Each config took about 0.5s to simulate.

| Threads | p90 | p95 | p99 | 1 min | 5 min |
|---|---|---|---|---|---|
| 2 | 14,828s | 15,855s | 16,893s | FAIL | FAIL |
| 4 | 19.8s | 28.9s | 51.9s | PASS | PASS |
| 6 | 9.7s | 14.7s | 32.9s | PASS | PASS |
| 8 | 9.4s | 14.4s | 32.7s | PASS | PASS |
| 12 (6.7 MB/s each) | 14.0s | 21.6s | 49.0s | PASS | PASS |

Takeaways:
- With too few threads the queue never catches up (2 threads).
- Past about 6 threads, adding more doesn't help. p99 is then set by the biggest files ÷ per-thread speed.
- Going over the max rate (12 × 10 > 80) slows every thread, so big files take longer and p99 gets worse.

## 3. v0.2: real bandwidth sharing (same 1M rows, 10 MB/s per thread)

v0.1 cut every thread to max ÷ threads all the time. v0.2 slows threads only while
they're all busy at once (model from the Perplexity kernels, checked against a slow
reference on 180 random cases). It takes about 1.5s per config instead of 0.5s.

| Threads | v0.1 p99 | v0.2 p99 (max 80 MB/s) |
|---|---|---|
| 8 | 32.7s | 32.7s |
| 12 | 49.0s | 32.7s |
| 16 | 65.3s | 32.7s |
| 32 | 130.5s | 32.7s |

At this load, adding threads costs nothing. v0.1 was wrong about that.

## 4. Average load hides the peak (max 30 MB/s)

Average load was 53%, but the busiest hour (Oct 5, 2pm) was 36.1 MB/s = 120% of
max. Result: p99 ≈ 2,050s and both tests fail at 4, 8, or 16 threads. Adding
threads can't fix a peak above the max rate; only a higher max rate (or spreading
requests out) can. This is why the busiest-hour check now runs first.

## 5. Auto-scaling threads (2026-11-14), 10 MB/s per thread, max 80, range 2–16

Controller every 30s: double threads if the oldest waiting file waited > 10s; remove one
after `--down-after` calm checks in a row. Checked: with min = max it matches the shared model exactly.

| Setup | p99 | Avg threads | Limit changes / month |
|---|---|---|---|
| Fixed 8 | 32.7s | 8.0 | 0 |
| Adaptive, down after 1 | 36.8s | 2.9 | 15,613 (flaps) |
| Adaptive, down after 10 | 33.3s | 4.1 | 2,115 |
| Adaptive, down after 20 | 32.9s | 4.7 | 806 |

Down-after 10 gives the same p99 as fixed 8 with about half the threads. Charts: `charts/adaptive-down1/`, `charts/adaptive-down10/`.
Open question: can real NAPM change thread count while running? If not, use a schedule (e.g. 7 threads 9–5, 3 otherwise).
