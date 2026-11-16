"""Run every real site in one go. Works whether or not a site has start/end times.

  python src/real_sites.py data/real/sites.csv --from 2026-10-01 --to 2026-11-01

sites.csv, one row per site (template: data/sites.example.csv):
  site,file,max_mbps
  north,data/real/north.parquet,1000

Per site:
  with start/end times    -> measured per-thread speed (line fit on calm transfers), what really
                             happened (% on time from end - request), real concurrent FTP peak,
                             plus everything below (the measured speed is added to the rates tried)
  without start/end times -> concurrent FTP curve from request + size / rate, simulator check
Rows with a blank start or end still count for the simulation; they're just left out of the
measured numbers. Column names are set once in COLUMNS (src/transport_sim.py).
Results: results/real/<site>/ and results/real/summary.csv (gitignored, real data stays local).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from concurrency import chart, concurrency, describe, size_from_curve
from recommend import fit_speed
from scenarios import RATES
from transport_sim import has_times, load, on_time, timed_mask


def measured(sc):
    """What the start/end times say: speed, overhead, real % on time, real concurrency peak."""
    tm = sc[timed_mask(sc)]
    t0 = sc.request_time.iloc[0]
    lat = (tm.end_time - tm.request_time).dt.total_seconds().to_numpy()
    _, level = concurrency((tm.start_time - t0).dt.total_seconds().to_numpy(),
                           (tm.end_time - t0).dt.total_seconds().to_numpy())
    out = dict(times=f"yes ({len(tm) / len(sc):.0%} of files)", observed_peak_threads=int(level.max()))
    out.update({f"observed_{k}": f"{np.floor(v * 100) / 100:.2f}%" for k, v in on_time(lat).items()})
    try:
        rate, overhead, info = fit_speed(sc)
        out.update(measured_rate=round(rate, 1), overhead_s=round(overhead, 2))
    except (ValueError, np.linalg.LinAlgError, TypeError):  # too few calm transfers to fit a line
        out.update(measured_rate=None, overhead_s=None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sites", help="CSV with columns site,file,max_mbps")
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--out", default="results/real")
    a = ap.parse_args()

    summary = []
    for cfg in pd.read_csv(a.sites).itertuples():
        mx = float(cfg.max_mbps)
        df = load(cfg.file, a.date_from, a.date_to, 60)
        sc = df[df.scored]
        if sc.empty:
            print(f"\n{cfg.site}: no requests between {a.date_from} and {a.date_to}, skipped")
            continue
        row = dict(site=cfg.site, files=len(sc), max=mx, times="no")
        rates = [r for r in RATES if r <= mx]
        if has_times(sc):
            row.update(measured(sc))
            if row.get("measured_rate"):
                rates = sorted(set(rates + [min(row["measured_rate"], mx)]))
        t, biggest = size_from_curve(df, mx, rates)
        out = Path(a.out) / str(cfg.site)
        out.mkdir(parents=True, exist_ok=True)
        t.to_csv(out / "concurrency_sizing.csv", index=False)
        chart(sc, t, mx, out, f"{cfg.site}: ")
        print(f"\n=== {cfg.site}: {len(sc):,} files, start/end times: {row['times']}")
        if row["times"] != "no":
            print(f"  measured: speed {row['measured_rate']} MB/s, overhead {row['overhead_s']}s, "
                  f"real peak {row['observed_peak_threads']} transfers at once, "
                  f"real on time 1min {row['observed_1min']} / 5min {row['observed_5min']}")
        row.update({f"pick_{k}": v for k, v in describe(cfg.site, t, biggest, mx).items()})
        summary.append(row)

    s = pd.DataFrame(summary)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    s.to_csv(Path(a.out) / "summary.csv", index=False)
    print("\n" + s.to_string(index=False))


if __name__ == "__main__":
    main()
