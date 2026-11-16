"""Compare a fixed thread count with auto-scaling threads on the same window.

  python src/adaptive.py data/samples/big.parquet --from 2026-10-01 --to 2026-11-01 \
      --thread-mbps 10 --max-mbps 80 --fixed 8 --min 2 --max 16 --out results/charts/adaptive

Prints % of files on time for 1min/5min and average/peak threads for both, and saves a chart of the worst day.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import plots  # sets the chart style
from plots import BLUE, ORANGE, plt, worst_day
from transport_sim import MB, load, on_time, simulate_adaptive, simulate_shared, verdict


def avg_threads(log, t_end):
    """Time-weighted average of the thread limit."""
    times = np.array([t for t, _ in log] + [t_end])
    ns = np.array([n for _, n in log])
    return float((ns * np.diff(times)).sum() / (t_end - times[0])), int(ns.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--thread-mbps", type=float, required=True)
    ap.add_argument("--max-mbps", type=float, required=True)
    ap.add_argument("--fixed", type=int, required=True, help="fixed thread count to compare against")
    ap.add_argument("--min", type=int, default=2)
    ap.add_argument("--max", type=int, default=16)
    ap.add_argument("--tick", type=float, default=30, help="seconds between controller checks")
    ap.add_argument("--up-wait", type=float, default=10, help="scale up when the oldest file waited this long")
    ap.add_argument("--down-after", type=int, default=10,
                    help="calm checks in a row before removing a thread (stops flapping)")
    ap.add_argument("--warmup-min", type=float, default=60)
    ap.add_argument("--out", default="results/charts/adaptive")
    a = ap.parse_args()

    df = load(a.csv, a.date_from, a.date_to, a.warmup_min)
    req = (df.request_time - df.request_time.iloc[0]).dt.total_seconds().to_numpy()
    size = df.file_size_bytes.to_numpy(float)
    scored = df.scored.to_numpy()
    rate, cap = a.thread_mbps * MB, a.max_mbps * MB

    end_f = simulate_shared(req, size, a.fixed, rate, cap)
    end_a, log = simulate_adaptive(req, size, rate, cap, a.min, a.max, a.tick, a.up_wait, a.down_after)
    t_end = max(end_f.max(), end_a.max())
    avg_a, peak_a = avg_threads(log, t_end)

    print(f"{'':<22}{'1min on time':>18}{'5min on time':>18}   avg threads  peak")
    for label, end, avg, peak in ((f"fixed {a.fixed}", end_f, a.fixed, a.fixed),
                                  (f"adaptive {a.min}-{a.max}", end_a, avg_a, peak_a)):
        met = on_time((end - req)[scored])
        cells = "".join(f"{np.floor(v * 100) / 100:>11.2f}% {verdict(v)}" for v in met.values())
        print(f"{label:<22}{cells}   {avg:>10.1f}  {peak:>4}")
    print(f"thread-limit changes: {len(log) - 1:,}")

    # chart: worst day for the adaptive run, thread limit vs files in system
    worst = worst_day(df, req, end_a, scored)
    day0 = (pd.Timestamp(worst) - df.request_time.iloc[0]).total_seconds()
    t = np.arange(day0, day0 + 86400, 60)
    hours = (t - day0) / 3600
    sreq = np.sort(req)
    in_sys_a = np.searchsorted(sreq, t, "right") - np.searchsorted(np.sort(end_a), t, "right")
    in_sys_f = np.searchsorted(sreq, t, "right") - np.searchsorted(np.sort(end_f), t, "right")
    lt, ln = zip(*log)
    limit = np.array(ln)[np.searchsorted(np.array(lt), t, "right") - 1]

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax1.step(hours, limit, where="post", color=BLUE, label=f"adaptive ({a.min}-{a.max})")
    ax1.axhline(a.fixed, color=ORANGE, ls="--", lw=2, label=f"fixed {a.fixed}")
    ax1.set_ylabel("thread limit"); ax1.legend(frameon=False, loc="upper left")
    ax1.set_title(f"Thread limit through the worst day ({worst})")
    ax2.plot(hours, in_sys_a, color=BLUE, lw=1.5, label="adaptive")
    ax2.plot(hours, in_sys_f, color=ORANGE, lw=1.5, ls="--", label=f"fixed {a.fixed}")
    ax2.set_ylabel("files waiting or transferring"); ax2.set_xlabel("hour of day")
    ax2.legend(frameon=False, loc="upper left"); ax2.set_title("Files in the system")
    ax2.set_xlim(0, 24); ax2.set_xticks(range(0, 25, 3))
    fig.tight_layout(); fig.savefig(out / "adaptive_worst_day.png", dpi=150); plt.close(fig)
    print(f"chart saved to {out / 'adaptive_worst_day.png'}")


if __name__ == "__main__":
    main()
