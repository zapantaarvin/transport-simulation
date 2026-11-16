# Guide: choosing NAPM run settings from an input file

Short version: run `src/recommend.py` on the date window. The steps below explain what it does,
so you can do it by hand or sanity-check its answer.

```bash
.venv/bin/python src/recommend.py data/real/export.parquet --from 2026-10-01 --to 2026-11-01 \
    --max-mbps <spec rate> --limit 1min   # or --limit 5min
```

## What you need

| Input | Required? | Used for |
|---|---|---|
| file size, request time | yes | the simulation itself |
| start time, end time | strongly recommended | learning the real per-thread speed and per-file overhead |
| max spec rate | yes, from the spec | the total cap; it can't be learned from data |

## Step 1. Per-thread rate: learn it from start/end times

Each transfer's duration is a fixed cost plus the time to move the bytes:

```
duration (end − start)  =  overhead  +  size ÷ rate
                           ────────     ───────────
                           login, dir,  the part that
                           handshake    scales with size
```

Plot duration against size for many files: it's roughly a straight line.
**Intercept = overhead per file. Slope = 1 ÷ rate.** That's a plain straight-line fit.

Two traps the script avoids:
1. **Don't use the median of `size ÷ duration`.** Small files are mostly overhead, so it badly
   underestimates speed. Test: true 25 MB/s + 1.5s overhead → the naive median said 8.6 MB/s;
   the line fit said 25.1 MB/s + 1.50s.
2. **Only use "calm" transfers**, ones that started while few others were running. Busy-time
   transfers were sharing bandwidth, so they look slower than a thread really is.

No start/end times? Pass `--thread-mbps` and `--overhead-s` yourself (from the FTP config or a
small test transfer), and treat the result as a rough estimate until you can check it.

## Step 2. Feasibility: can anything pass?

| Check | Rule | If it fails |
|---|---|---|
| Busiest hour | demand ≤ 80% of max rate | No thread count helps. Raise the max rate or spread requests out. |
| Biggest files | overhead + (biggest file ÷ rate) ≤ limit | Per-thread rate too low. Need rate ≥ size ÷ (limit − overhead). Every file must make it, so the single biggest one counts. |

## Step 3. Thread count: search, don't guess

Try 1, 2, 4, 8, … threads until every simulated file finishes within the limit (`--limit 1min`
or `5min`), then narrow down to the smallest count that still passes. Each try on 1M rows takes about 1.5s.

- **Goal = 100% of files on time** (`TARGET` in `src/transport_sim.py`). The limits are fixed:
  60s and 300s from request.
- **More threads isn't always better:** they split the max rate, so a huge file can slow down
  and miss the limit (seen in Sites C and D).
- **Rule of thumb ceiling:** threads beyond `max rate ÷ per-thread rate` only share the same
  bandwidth. They can still help when many small files are waiting behind overhead, so the
  search tests it rather than assuming.

## Step 4. Confirm it

1. Run with the recommended settings over the full month, then over the worst day it reports.
2. Compare the simulated % on time with the `observed` line (real `end − request`).
   If they're close, the model is trustworthy for what-ifs.
3. Write the settings and the date window down in `results/`.

## Do I need a percentile curve, Bayesian stats, or a neural network?

| Approach | Verdict | Why |
|---|---|---|
| % of files on time (1 min / 5 min) | **Yes** | It's the output being tested. Every file must make it to pass. |
| Percentiles of per-file speed | Useful as a **check**, not as the rate | Shows spread (p10/p50/p90); the raw values are biased by overhead and sharing. |
| Straight-line fit (duration vs size) | **Yes, this is the right stats** | Two numbers, easy to check, separates overhead from speed. |
| Bootstrap (resample the data, rerun) | Later, if needed | Shows how often a pass could flip to a fail on a different month when data is thin. Simpler than Bayesian. |
| Bayesian model | Not needed now | Same uncertainty answer as bootstrap, much more work. |
| Neural network / SGD | **No** | Picking settings is a search over a simulator that's already exact and fast. A NN would only approximate the simulator, needs lots of training runs, and can't be trusted outside data it saw. SGD is just the fitting method a NN uses; the line fit above is solved exactly without it. |

When would ML earn its place? Predicting **future** demand (next month's request volume and
sizes) to feed into the simulator. That's a forecasting problem, separate from choosing settings.

## Simplifications in recommend.py

- Overhead is modeled as extra bytes at full thread speed, so it uses some bandwidth in the
  model when in reality it mostly doesn't. It's slightly pessimistic when the max rate is tight.
- "Calm" = started with ≤ 2 others running (or the bottom 25% of concurrency if busier).
- One rate for all threads; no retries.
