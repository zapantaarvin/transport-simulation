"""Recommend starting settings (threads, per-thread rate) for a date window.

  python src/recommend.py data/samples/big.parquet --from 2026-10-01 --to 2026-11-01 --max-mbps 80 --limit 1min

Steps (same order as GUIDE.md):
  1. Learn speed from real start/end times: fit  duration = overhead + size / rate
     using only transfers that ran with few others at once (so sharing doesn't drag the rate down).
  2. Feasibility: busiest hour vs max rate, biggest files vs the limit.
  3. Find the smallest thread count where TARGET % of files (default: all) finish within the limit.
"""
import argparse

import numpy as np

from transport_sim import MB, SLAS, TARGET, has_times, load, simulate_shared, timed_mask


def concurrency_at_start(start_s, end_s):
    """How many transfers were running when each one started (itself included)."""
    starts, ends = np.sort(start_s), np.sort(end_s)
    return np.searchsorted(starts, start_s, "right") - np.searchsorted(ends, start_s, "right")


def fit_speed(df):
    """Fit duration = overhead + size/rate on low-concurrency transfers. Returns (rate MB/s, overhead s, info).
    Uses only rows with both start and end times."""
    df = df[timed_mask(df)]
    t0 = df.request_time.min()
    start = (df.start_time - t0).dt.total_seconds().to_numpy()
    end = (df.end_time - t0).dt.total_seconds().to_numpy()
    size_mb = df.file_size_bytes.to_numpy(float) / MB
    dur = end - start
    conc = concurrency_at_start(start, end)
    calm = conc <= max(2, np.percentile(conc, 25))
    slope, overhead = np.polyfit(size_mb[calm], dur[calm], 1)  # dur ≈ slope*size + overhead
    per_file = size_mb / np.maximum(dur, 1e-9)
    return 1 / slope, max(overhead, 0.0), {
        "calm_rows": int(calm.sum()),
        "rate_pcts": np.percentile(per_file[calm], [10, 50, 90]),
        "conc_median": float(np.median(conc)),
    }


def latency_for(req_s, size_b, n, thread_mbps, max_mbps, overhead_s):
    # overhead modeled as extra bytes at full thread speed (simplification, see GUIDE.md)
    eff = size_b + overhead_s * thread_mbps * MB
    end = simulate_shared(req_s, eff, n, thread_mbps * MB, max_mbps * MB)
    return end - req_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--max-mbps", type=float, required=True, help="max spec rate (from the spec, not the data)")
    ap.add_argument("--limit", choices=list(SLAS), default="1min", help="which fixed limit must pass")
    ap.add_argument("--thread-mbps", type=float, help="override the fitted per-thread rate")
    ap.add_argument("--overhead-s", type=float, help="override the fitted per-file overhead")
    ap.add_argument("--max-threads", type=int, default=256)
    ap.add_argument("--warmup-min", type=float, default=60)
    a = ap.parse_args()

    df = load(a.csv, a.date_from, a.date_to, a.warmup_min)
    sc = df[df.scored]
    limit = SLAS[a.limit]
    print(f"window {a.date_from} -> {a.date_to}: {len(sc):,} files | goal: {TARGET:g}% of files finish "
          f"within {limit}s ({a.limit})\n")

    # 1. speed from data
    rate, overhead = a.thread_mbps, a.overhead_s
    if has_times(sc):
        fit_rate, fit_oh, info = fit_speed(sc)
        lo, mid, hi = info["rate_pcts"]
        print("STEP 1  speed learned from start/end times")
        print(f"  {info['calm_rows']:,} calm transfers used (typical concurrency was {info['conc_median']:.0f})")
        print(f"  fit: duration = {fit_oh:.2f}s overhead + size / {fit_rate:.1f} MB/s")
        print(f"  per-file speed p10/p50/p90 = {lo:.1f} / {mid:.1f} / {hi:.1f} MB/s")
        rate = rate or fit_rate
        overhead = fit_oh if overhead is None else overhead
    elif rate is None:
        raise SystemExit("no start/end times in the data: pass --thread-mbps (and --overhead-s) yourself")
    overhead = overhead or 0.0
    rate = min(rate, a.max_mbps)
    print(f"  -> per-thread rate {rate:.1f} MB/s, overhead {overhead:.2f}s per file\n")

    # 2. feasibility
    hourly = sc.groupby(sc.request_time.dt.floor("h")).file_size_bytes.sum() / MB / 3600
    biggest = sc.file_size_bytes.max() / MB
    one_file = overhead + biggest / rate
    too_big = int((overhead + sc.file_size_bytes / MB / rate > limit).sum())
    print("STEP 2  can anything pass?")
    print(f"  busiest hour {hourly.idxmax()}: {hourly.max():.1f} MB/s = {hourly.max() / a.max_mbps:.0%} of max rate")
    print(f"  biggest file ({biggest:,.0f} MB) alone takes {one_file:.0f}s at {rate:.1f} MB/s")
    blockers = []
    if hourly.max() > a.max_mbps:
        blockers.append("busiest hour is over the max rate: raise the max rate or spread requests out")
    if too_big and (1 - too_big / len(sc)) * 100 < TARGET:
        blockers.append(f"{too_big:,} files are too big to make {a.limit} even alone: per-thread rate must be "
                        f">= {biggest / max(limit - overhead, 1e-9):.1f} MB/s")
    for b in blockers:
        print(f"  BLOCKED: {b}")
    if blockers:
        print("  (still searching threads below so you can see how far off it is)")
    print()

    # 3. thread search: double until it passes, then binary search down
    req = (df.request_time - df.request_time.iloc[0]).dt.total_seconds().to_numpy()
    size = df.file_size_bytes.to_numpy(float)
    scored = df.scored.to_numpy()
    cache = {}

    def met(n):
        if n not in cache:
            cache[n] = (latency_for(req, size, n, rate, a.max_mbps, overhead)[scored] <= limit).mean() * 100
            print(f"  tried {n:>3} threads -> {np.floor(cache[n] * 100) / 100:.2f}% on time")
        return cache[n]

    print("STEP 3  smallest thread count that meets the goal")
    n = 1
    while met(n) < TARGET and n < a.max_threads:
        n = min(n * 2, a.max_threads)
    if met(n) < TARGET:
        print(f"\nNo thread count up to {a.max_threads} meets the goal. Fix the STEP 2 blockers first.")
        return
    lo, hi = n // 2 + 1 if n > 1 else 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if met(mid) >= TARGET else (mid + 1, hi)
    best = lo
    ceiling = int(np.ceil(a.max_mbps / rate))
    print(f"""
RECOMMENDED START
  concurrent FTP threads : {best}
  rate per thread        : {rate:.1f} MB/s
  max spec rate          : {a.max_mbps:g} MB/s
  files on time ({a.limit})  : {np.floor(cache[best] * 100) / 100:.2f}% (goal {TARGET:g}%)
  note: threads past {ceiling} ({a.max_mbps:g} / {rate:.1f}) only share the same max rate""")


if __name__ == "__main__":
    main()
