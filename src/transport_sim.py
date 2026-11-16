"""Transport simulation v0.2: replay file transfer requests through N threads.

Input columns (.parquet or .csv):
  file_size_bytes, request_time            (required)
  start_time, end_time                     (optional, real production times, used only to check the sim;
                                            rename your columns in COLUMNS below)

Examples:
  python transport_sim.py --make-sample sample.parquet --rows 1000 --days 30
  python transport_sim.py sample.parquet --from 2026-10-01 --to 2026-10-08 --threads 10 --thread-mbps 50 --max-mbps 400
  python transport_sim.py sample.parquet --from 2026-10-01 --to 2026-11-01 --sweep 2,4,8,16 --thread-mbps 50 --max-mbps 400
"""
import argparse
import heapq
import time
from collections import deque

import numpy as np
import pandas as pd

# The two fixed limits: a file is on time if it finishes within this many seconds of its request.
SLAS = {"1min": 60, "5min": 300}
TARGET = 100.0  # % of files that must be on time for a PASS (100 = every single file)
MB = 1_000_000

# Name the code uses -> name in your parquet. Edit the right side to match your export.
# start_time / end_time are optional: a site without them (or with them blank) still runs.
COLUMNS = {
    "file_size_bytes": "file_size_bytes",
    "request_time": "request_time",
    "start_time": "start_time",
    "end_time": "end_time",
}


def make_sample(path, rows, days, start="2026-10-01"):
    """Fake data: busier during work hours, mostly small files with a few huge ones."""
    rng = np.random.default_rng(7)
    t0 = pd.Timestamp(start)
    # request times: random seconds across the period, weighted toward 9am-5pm
    secs = rng.uniform(0, days * 86400, rows * 3)
    hour = (secs % 86400) / 3600
    keep = rng.random(secs.size) < np.where((hour >= 9) & (hour < 17), 1.0, 0.25)
    secs = np.sort(secs[keep][:rows])
    size = rng.lognormal(mean=np.log(20 * MB), sigma=1.2, size=secs.size).astype(np.int64)
    # pretend production ran at ~40 MB/s per transfer with a little queue wait
    wait = rng.exponential(5, secs.size)
    dur = size / (40 * MB) * rng.uniform(0.8, 1.3, secs.size)
    req = t0 + pd.to_timedelta(secs, unit="s")
    df = pd.DataFrame({
        "file_size_bytes": size,
        "request_time": req,
        "start_time": req + pd.to_timedelta(wait, unit="s"),
        "end_time": req + pd.to_timedelta(wait + dur, unit="s"),
    })
    write_table(df, path)
    print(f"wrote {len(df):,} rows to {path}")


def write_table(df, path):
    """Parquet if the path ends in .parquet (smaller, faster), else CSV."""
    if str(path).endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load(path, date_from, date_to, warmup_min):
    lo, hi = pd.Timestamp(date_from), pd.Timestamp(date_to)
    req_col = COLUMNS["request_time"]
    if str(path).endswith(".parquet"):
        # only the window (+ warm-up) is read from disk, so big files don't fill memory
        start = lo - pd.Timedelta(minutes=warmup_min)
        df = pd.read_parquet(path, filters=[(req_col, ">=", start), (req_col, "<", hi)])
    else:
        df = pd.read_csv(path)
    df = df.rename(columns={src: ours for ours, src in COLUMNS.items() if src != ours})
    for col in ("request_time", "start_time", "end_time"):
        if col in df:
            df[col] = pd.to_datetime(df[col])
    if {"start_time", "end_time"} <= set(df.columns) and not timed_mask(df).any():
        df = df.drop(columns=["start_time", "end_time"])  # columns exist but are empty: treat as absent
    # keep a warm-up slice before the window so threads aren't unrealistically empty at the start
    df = df[(df.request_time >= lo - pd.Timedelta(minutes=warmup_min)) & (df.request_time < hi)]
    df = df.sort_values("request_time").reset_index(drop=True)
    df["scored"] = df.request_time >= lo  # only these count toward the result
    return df


def timed_mask(df):
    """Rows that have both a start and an end time, after their request (all False if the columns are missing)."""
    if not {"start_time", "end_time"} <= set(df.columns):
        return pd.Series(False, index=df.index)
    return df.start_time.notna() & df.end_time.notna() & (df.end_time >= df.start_time)


def has_times(df):
    return bool(timed_mask(df).any())


def simulate(req_s, size, threads, rate_bps):
    """FIFO queue over N threads, each running at rate_bps. Returns finish time per file."""
    free = [0.0] * threads
    end = np.empty(req_s.size)
    for i in range(req_s.size):
        start = max(req_s[i], heapq.heappop(free))
        end[i] = start + size[i] / rate_bps
        heapq.heappush(free, end[i])
    return end


def simulate_shared(req_s, size, threads, rate_bps, max_bps):
    """FIFO admission into N threads; active threads share max_bps, each capped at rate_bps.

    Speed only drops when more than max/rate threads are busy at the same moment.
    Uses a virtual clock v (bytes moved per active thread so far): a file that starts at
    v finishes when v reaches v + size, so each event is O(log N), not O(N).
    Adapted from the Perplexity kernels (research/, 2026-11-13); checked equal to a
    step-by-step reference on 180 random cases.
    """
    end = np.empty(req_s.size)
    waiting = deque()
    active = []  # (finish tag in virtual bytes, index)
    i, t, v, m = 0, 0.0, 0.0, req_s.size
    while i < m or active or waiting:
        k = len(active)
        speed = min(rate_bps, max_bps / k) if k else 0.0
        t_arrive = req_s[i] if i < m else np.inf
        t_done = t + max(0.0, active[0][0] - v) / speed if k else np.inf
        if t_done <= t_arrive:
            v += speed * (t_done - t)
            t = t_done
            tag, j = heapq.heappop(active)
            v = max(v, tag)
            end[j] = t
        else:
            v += speed * (t_arrive - t)
            t = t_arrive
            waiting.append(i)
            i += 1
        while waiting and len(active) < threads:
            j = waiting.popleft()
            heapq.heappush(active, (v + size[j], j))
    return end


def simulate_adaptive(req_s, size, rate_bps, max_bps, n_min, n_max, tick_s=30.0, up_wait_s=10.0,
                      down_after=1):
    """Same as simulate_shared, but the thread limit moves between n_min and n_max.

    Every tick_s seconds a controller looks at the queue:
      oldest waiting file has waited > up_wait_s  -> double the threads (react fast to a pile-up)
      calm (nothing waiting, half the threads idle) for down_after checks in a row
                                                  -> remove one thread (back off slowly)
    Lowering the limit never stops a running transfer; it just admits fewer new ones.
    Returns (finish times, [(time, thread limit), ...]).
    """
    end = np.empty(req_s.size)
    waiting = deque()
    active = []
    i, t, v, m = 0, 0.0, 0.0, req_s.size
    n, next_tick, log, calm = n_min, tick_s, [(0.0, n_min)], 0
    while i < m or active or waiting:
        k = len(active)
        speed = min(rate_bps, max_bps / k) if k else 0.0
        t_arrive = req_s[i] if i < m else np.inf
        t_done = t + max(0.0, active[0][0] - v) / speed if k else np.inf
        if t_done <= t_arrive and t_done <= next_tick:
            v += speed * (t_done - t)
            t = t_done
            tag, j = heapq.heappop(active)
            v = max(v, tag)
            end[j] = t
        elif t_arrive <= next_tick:
            v += speed * (t_arrive - t)
            t = t_arrive
            waiting.append(i)
            i += 1
        else:
            v += speed * (next_tick - t)
            t = next_tick
            next_tick += tick_s
            oldest = t - req_s[waiting[0]] if waiting else 0.0
            if oldest > up_wait_s:
                n, calm = min(n_max, n * 2), 0
            elif not waiting and len(active) <= n // 2:
                calm += 1
                if calm >= down_after:
                    n, calm = max(n_min, n - 1), 0
            else:
                calm = 0
            if n != log[-1][1]:
                log.append((t, n))
        while waiting and len(active) < n:
            j = waiting.popleft()
            heapq.heappush(active, (v + size[j], j))
    return end, log


def on_time(lat):
    """% of files that finished within each limit, e.g. {"1min": 97.6, "5min": 100.0}."""
    return {name: (lat <= s).mean() * 100 for name, s in SLAS.items()}


def verdict(pct):
    return "PASS" if pct >= TARGET else "FAIL"


def report(label, lat):
    met = on_time(lat)
    cells = "   ".join(f"{name}: {np.floor(v * 100) / 100:6.2f}% on time ({int((lat > SLAS[name]).sum()):,} late) {verdict(v)}"
                       for name, v in met.items())
    print(f"{label:<11}{cells}")
    return met


def run(df, threads, thread_mbps, max_mbps, per_day, model):
    t0 = df.request_time.iloc[0]
    req_s = (df.request_time - t0).dt.total_seconds().to_numpy()
    size = df.file_size_bytes.to_numpy(dtype=float)
    tic = time.perf_counter()
    if model == "shared":
        end = simulate_shared(req_s, size, threads, thread_mbps * MB, max_mbps * MB)
        label = f"up to {thread_mbps:.1f} MB/s each, sharing {max_mbps:.0f} MB/s"
    else:  # "even": old v0.1 model, every thread always cut to max / threads
        rate = min(thread_mbps, max_mbps / threads) * MB
        end = simulate(req_s, size, threads, rate)
        label = f"fixed {rate / MB:.1f} MB/s each"
    secs = time.perf_counter() - tic
    lat = (end - req_s)[df.scored.to_numpy()]
    print(f"\nN={threads} threads, {label}  ({lat.size:,} files, sim took {secs:.2f}s)")
    met = report("simulated", lat)
    if per_day:
        days = df.request_time[df.scored].dt.date.to_numpy()
        for name, s in SLAS.items():
            late = pd.Series(lat > s).groupby(days).sum().sort_values(ascending=False)
            late = late[late > 0]
            print(f"  days with {name} late files: " + (", ".join(f"{d} ({v:,})" for d, v in late.head(3).items())
                                                       + (f" +{len(late) - 3} more" if len(late) > 3 else "")
                                                       if len(late) else "none"))
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--make-sample")
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--from", dest="date_from", default="1900-01-01")
    ap.add_argument("--to", dest="date_to", default="2100-01-01")
    ap.add_argument("--warmup-min", type=float, default=60)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--thread-mbps", type=float, default=50)
    ap.add_argument("--max-mbps", type=float, default=400)
    ap.add_argument("--sweep", help="comma list of thread counts to try, e.g. 2,4,8,16")
    ap.add_argument("--model", choices=["shared", "even"], default="shared",
                    help="shared = threads share the max rate (default); even = v0.1 fixed split")
    a = ap.parse_args()

    if a.make_sample:
        return make_sample(a.make_sample, a.rows, a.days)

    df = load(a.csv, a.date_from, a.date_to, a.warmup_min)
    sc = df[df.scored]
    span = (sc.request_time.max() - sc.request_time.min()).total_seconds()
    load_mbps = sc.file_size_bytes.sum() / MB / span
    big = sc.file_size_bytes.max() / MB
    print(f"window {a.date_from} -> {a.date_to}: {len(sc):,} files, "
          f"avg load {load_mbps:.1f} MB/s = {load_mbps / a.max_mbps:.0%} of max rate")
    hourly = sc.groupby(sc.request_time.dt.floor("h")).file_size_bytes.sum() / MB / 3600
    print(f"busiest hour {hourly.idxmax()}: {hourly.max():.1f} MB/s = {hourly.max() / a.max_mbps:.0%} "
          f"of max rate" + ("  <- over 100%, the queue will pile up" if hourly.max() > a.max_mbps else ""))
    print(f"biggest file {big:,.0f} MB -> needs >= {big / 60:.1f} MB/s per thread for 1min, "
          f"{big / 300:.1f} for 5min")
    for name, s in SLAS.items():
        too_big = int((sc.file_size_bytes > a.thread_mbps * MB * s).sum())
        if too_big:
            print(f"  {too_big:,} files can't make {name} even alone at {a.thread_mbps:g} MB/s "
                  f"-> {name} can't PASS at this per-thread rate")

    if has_times(df):
        tm = sc[timed_mask(sc)]
        obs_lat = (tm.end_time - tm.request_time).dt.total_seconds()
        obs_rate = tm.file_size_bytes / (tm.end_time - tm.start_time).dt.total_seconds().clip(lower=1e-3) / MB
        print(f"\nreal data: {len(tm):,} of {len(sc):,} files have start/end times; "
              f"per-transfer speed median {obs_rate.median():.1f} MB/s, p10 {obs_rate.quantile(0.1):.1f} MB/s")
        report("observed", obs_lat.to_numpy())

    for n in ([int(x) for x in a.sweep.split(",")] if a.sweep else [a.threads]):
        run(df, n, a.thread_mbps, a.max_mbps, per_day=not a.sweep, model=a.model)


if __name__ == "__main__":
    main()
