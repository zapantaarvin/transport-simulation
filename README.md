# Transport Simulation (NAPM)

Replays file transfer requests through N FTP threads to test whether transfers
finish within **1 min** and **5 min** of their request. A run **passes** a limit only when
every file makes it (`TARGET = 100.0` in `src/transport_sim.py`; the limits are `SLAS` right above it).

Status: v0.2 (real bandwidth sharing), tested on fake data only (1k and 1M rows,
about 1.5s per config). Next: real column names.

## Folder map

| Path | What's in it |
|---|---|
| `src/transport_sim.py` | The simulator |
| `src/plots.py` | Charts (PNG) for one window |
| `src/prerun.py` | Before-run charts from the input only: demand vs max, concurrent FTP needed, load-duration, file-size curve |
| `src/concurrency.py` | Concurrent FTP over time (+1 at start, −1 at finish) → threads + per-thread rate, checked by the simulator. `all` = every scenario site |
| `src/napm_output.py` | NAPM-style output: sweeps `comm_mbps` (total bandwidth) and charts % of days with p50–p99 under 5 / 1 min, latency vs bandwidth (p90/95/99 + table), and each day's p90. `all` = every scenario site, results in `results/napm/` |
| `src/real_sites.py` | Runs every real site listed in a CSV (copy `data/sites.example.csv`). Uses start/end times where a site has them; works without. Output in `results/real/` (not committed) |
| `src/scenarios.py` | Synthetic sites A–E (50k files each) + settings grid; results in `results/scenarios/` |
| `src/adaptive.py` | Fixed vs auto-scaling threads comparison + worst-day chart |
| `src/recommend.py` | Recommends threads + per-thread rate for a window (see GUIDE.md) |
| `STUDY-GUIDE.html` | Start here to learn it: 10 modules + an in-page simulator lab (open in a browser) |
| `GUIDE.md` | How to pick run settings from an input file |
| `data/samples/` | Fake data (`small.parquet` 1k rows; `big.parquet` 1M rows, not in git) |
| `data/real/` | Real work exports go here, never committed |
| `research/` | Perplexity prompt + filed research |
| `results/scenarios/REPORT.html` | All 6 sites: charts, observations, what to tune (open in a browser) |
| `results/` | Saved run outputs and findings; `results/charts/<scenario>/` for PNGs |
| `requirements.txt` | numpy, pandas, matplotlib |

## How it works

```
request ──► [ queue ] ──► thread 1..N (each ≤ per-thread rate, total ≤ max spec rate) ──► done
latency = done − request_time      ← tested against 60s / 300s
```

- **Inputs:** `file_size_bytes`, `request_time`.
- **Optional:** `start_time`, `end_time` are what really happened, so they are used
  to check the sim, not as inputs:
  - `size / (end − start)` = real per-thread speed → use it as `--thread-mbps`
  - `end − request` = real latency → compare its % on time with the simulated one
- **Date window:** `--from` / `--to` picks the window. The hour before it is loaded
  but not scored (warm-up), so threads aren't unrealistically empty at the start.
- **Bandwidth sharing (default `--model shared`):** each busy thread runs at
  `min(per-thread rate, max rate ÷ busy threads)`, recalculated every time a transfer
  starts or ends. So extra threads only slow things down when they're all busy at once.
  `--model even` is the old v0.1 version (always max ÷ threads), kept for comparison.

## Run it

```bash
cd "Work Project/Transport Simulation"
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once

# make fake data
.venv/bin/python src/transport_sim.py --make-sample data/samples/test.parquet --rows 1000 --days 30

# one config over one week
.venv/bin/python src/transport_sim.py data/samples/small.parquet --from 2026-10-01 --to 2026-10-08 \
    --threads 4 --thread-mbps 38 --max-mbps 400

# compare thread counts over a month
.venv/bin/python src/transport_sim.py data/samples/big.parquet --from 2026-10-01 --to 2026-11-01 \
    --sweep 2,4,6,8,12 --thread-mbps 10 --max-mbps 80
```

## Charts

```bash
.venv/bin/python src/plots.py data/samples/big.parquet --from 2026-10-01 --to 2026-11-01 \
    --threads 8 --thread-mbps 10 --max-mbps 80 --sweep 2,4,8,16,32 --out results/charts/max-80
```

| Chart | Question it answers |
|---|---|
| `1_load_vs_max` | Does demand in any hour go over the max rate? (red dots = yes) |
| `2_latency_spread` | How long did files take, and how many are past the 1min/5min lines? |
| `3_thread_sweep` | How many threads until % on time stops improving? |
| `4_queue_busiest_day` | On the worst day, how many files piled up and when did it clear? |
| `5_sla_heatmap` | Day × hour: what % of files met 1 min and 5 min? Blue ≥ 99%, red below, right column = whole day |

Charts stay local (PNG) so real work data never leaves this machine.

## Where the numbers came from (so far)

**Everything up to now is made up, on purpose, to show how the sim behaves.**

| Number | Value used | Where it came from |
|---|---|---|
| File sizes | median 20 MB, p99 ≈ 326 MB | Chosen guess: lots of small files, a few huge ones |
| Busy hours | 9am–5pm, 4× busier | Chosen guess |
| Real per-thread speed in the fake start/end times | ≈ 38 MB/s | Chosen guess, baked into the fake data |
| `--thread-mbps 50`, `--max-mbps 400` | first runs | Round numbers |
| `--thread-mbps 10`, `--max-mbps 80` | 1M-row runs | Picked so load is about 20% and the effects show up |
| `--max-mbps 30` | tight test | Picked so the busiest hour goes over 100% |

**With real data, set them like this:**

| Parameter | How to set it |
|---|---|
| `--max-mbps` | The spec sheet / link limit (the "max spec rate"). Not from the data. |
| `--thread-mbps` | From the data: median of `size / (end − start)`, looking only at times when few transfers ran at once (so the number isn't pulled down by sharing). If there's an FTP per-thread setting, use that and compare. |
| `--threads` | Don't guess. Sweep it (`--sweep 2,4,8,16,32`) and take the smallest N where every file is on time. |
| Date window | Run the full month, then the worst day and busiest hour the tool prints. |
| Check | Run with the real config and compare the simulated % on time to the `observed` line. If they're close, trust the what-ifs. |

## Real data

1. Put each site's parquet in `data/real/` (never committed).
2. If your column names differ, edit `COLUMNS` at the top of `src/transport_sim.py` (one place, all scripts).
3. Copy `data/sites.example.csv` to `data/real/sites.csv`, one row per site with its max rate.
4. `.venv/bin/python src/real_sites.py data/real/sites.csv --from 2026-10-01 --to 2026-11-01`

| Site has… | What you get |
|---|---|
| start + end times | measured per-thread speed + overhead, what really happened (% on time), real concurrent FTP peak, and the sizing below |
| no times (or blank columns) | sizing from the concurrency curve (request + size ÷ rate) + simulator check |
| times on only some rows | rows with both times feed the measured numbers; every row counts for the simulation. The real peak only counts rows with times, so it may read low |

## Checks before simulating (printed automatically)

1. **Load %** = total bytes ÷ time span ÷ max rate.
2. **Busiest-hour load %.** This one matters more. A month can average 53% and still
   go over 100% at 2pm, and then no thread count passes. Keep the busiest hour
   under about 70–80% for good tail latency.
3. **Min per-thread speed** = biggest file ÷ 60s (or 300s). Below that, no thread
   count can pass, since every file has to make it.

## Known limits (v0.2)

- Every thread has the same cap, and there's one shared bottleneck (the max rate).
- No per-file overhead (connection setup, TCP slow start) and no retries.
- SLA = per-request latency. A "drain every 5-min window" check isn't built yet;
  confirm which one the team means.
