"""Before-run charts: read only the input file, no simulation. Answers "where do I start?"

  python src/prerun.py data/scenarios/site-a.parquet --from 2026-10-01 --to 2026-11-01 \
      --max-mbps 1000 --thread-mbps 100 --out results/scenarios/site-a

Saves 0_prerun.png with four panels:
  a. demand per 5 min vs max rate          -> is the pipe ever too small, and when?
  b. concurrent FTP needed over time        -> if every file started at once at the per-thread rate,
                                               how many would be in flight? p95/p99 = starting thread count
  c. load-duration curve                    -> % of time demand is over 80% / 100% of max
  d. file-size percentile curve             -> % of files that CAN finish in 1 / 5 min at this per-thread rate
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import plots  # chart style
from plots import BLUE, CRITICAL, INK2, ORANGE, SURFACE, plt, ref_line
from transport_sim import MB, SLAS, load


def offered_concurrency(req_s, size, rate_bps, bucket_s=300):
    """Peak number of transfers in flight per bucket if nothing ever waited and each ran at rate_bps."""
    end = req_s + size / rate_bps
    t = np.concatenate([req_s, end])
    step = np.concatenate([np.ones(req_s.size), -np.ones(req_s.size)])
    order = np.lexsort((-step, t))  # at equal times, starts before ends (counts brief overlaps)
    t, level = t[order], np.cumsum(step[order])
    b = (t // bucket_s).astype(int)
    peaks = pd.Series(level).groupby(b).max()
    full = peaks.reindex(range(b.max() + 1)).ffill().fillna(0)  # buckets with no events keep the level
    return full.index.to_numpy() * bucket_s, full.to_numpy()


def concurrency_seen(req_s, size, rate_bps):
    """Transfers already in flight when each file arrives (itself included), if nothing ever waited.
    Percentiles of this are per FILE, matching how the SLA is graded; per-time percentiles
    understate it because rushes are short but hold many files."""
    end = req_s + size / rate_bps
    return np.searchsorted(np.sort(req_s), req_s, "right") - np.searchsorted(np.sort(end), req_s, "right")


def label_right(ax, y, text, color):
    ax.annotate(text, (1, y), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                color=color, fontsize=9, xytext=(0, 3), textcoords="offset points")


def prerun(df, max_mbps, rate_mbps, out, title=""):
    sc = df[df.scored]
    t0 = sc.request_time.iloc[0]
    req = (sc.request_time - t0).dt.total_seconds().to_numpy()
    size = sc.file_size_bytes.to_numpy(float)
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    # a. demand per 5 min
    a = ax[0, 0]
    d5 = sc.groupby(sc.request_time.dt.floor("5min")).file_size_bytes.sum() / MB / 300
    d5 = d5.reindex(pd.date_range(d5.index.min(), d5.index.max(), freq="5min"), fill_value=0)
    a.plot(d5.index, d5.values, color=BLUE, lw=0.8)
    over = d5[d5 > max_mbps]
    a.scatter(over.index, over.values, s=10, color=CRITICAL, zorder=3)
    ref_line(a, y=max_mbps, text=f"max rate {max_mbps:g} MB/s", color=CRITICAL)
    a.set_title(f"a. Demand per 5 min  ({len(over)} of {len(d5)} periods over max)", fontsize=11)
    a.set_ylabel("MB/s requested"); a.tick_params(axis="x", labelrotation=30)

    # b. concurrency needed
    b = ax[0, 1]
    tb, conc = offered_concurrency(req, size, rate_mbps * MB)
    when = t0 + pd.to_timedelta(tb, unit="s")
    b.plot(when, conc, color=BLUE, lw=0.8)
    seen = concurrency_seen(req, size, rate_mbps * MB)
    p50, p95, p99 = np.percentile(seen, [50, 95, 99])
    ceiling = max_mbps / rate_mbps
    for v, lab in ((p99, f"p99 of files = {p99:.0f}"), (p95, f"p95 of files = {p95:.0f}")):
        b.axhline(v, color=INK2, lw=1.2, ls="--")
        label_right(b, v, lab, INK2)
    ref_line(b, y=ceiling, text=f"max ÷ rate = {ceiling:.0f}", color=ORANGE)
    b.set_title(f"b. Concurrent FTP needed at {rate_mbps:g} MB/s each  (peak {conc.max():.0f}, "
                f"median file sees {p50:.0f})", fontsize=11)
    b.set_ylabel("transfers in flight (5-min peak)"); b.tick_params(axis="x", labelrotation=30)

    # c. load-duration curve
    c = ax[1, 0]
    pct = np.sort(d5.values / max_mbps * 100)[::-1]
    x = (np.arange(pct.size) + 1) / pct.size * 100
    c.plot(x, pct, color=BLUE)
    c.set_xscale("log"); c.set_xlim(x[0], 100)
    for lvl, col in ((100, CRITICAL), (80, ORANGE)):
        share = (pct > lvl).mean() * 100
        c.axhline(lvl, color=col, lw=1.2, ls="--")
        txt = f"{lvl}% of max: over it {share:.2f}% of the time"
        if lvl == 100:
            ref_line(c, y=lvl, text=txt, color=col)
        else:  # right side, below the line, so it never collides with the 100% label
            c.annotate(txt, (1, lvl), xycoords=("axes fraction", "data"), ha="right", va="top",
                       color=col, fontsize=9, xytext=(0, -3), textcoords="offset points")
    c.set_title("c. Load-duration curve (5-min periods, busiest first)", fontsize=11)
    c.set_xlabel("% of time (log scale)"); c.set_ylabel("demand as % of max rate")

    # d. file-size percentile curve
    d = ax[1, 1]
    s_mb = np.sort(size / MB)
    d.plot(s_mb, np.arange(1, s_mb.size + 1) / s_mb.size * 100, color=BLUE)
    d.set_xscale("log")
    for (name, sla), col, ypos in zip(SLAS.items(), (CRITICAL, ORANGE), (0.40, 0.18)):
        cap = rate_mbps * sla  # biggest file that can finish alone within the limit
        can = (s_mb <= cap).mean() * 100
        d.axvline(cap, color=col, lw=1.2, ls="--")
        d.annotate(f"{name}: files ≤ {cap:,.0f} MB\n{can:.1f}% of files can make it", (cap, ypos),
                   xycoords=("data", "axes fraction"), ha="right", va="center", color=col, fontsize=9,
                   xytext=(-5, 0), textcoords="offset points",
                   bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"))
    big = s_mb[-1]
    d.set_title(f"d. File sizes: median {np.median(s_mb):,.0f} MB, biggest {big:,.0f} MB  "
                f"(biggest needs ≥ {big / 60:,.0f} MB/s for 1 min)", fontsize=11)
    d.set_xlabel("file size, MB (log scale)"); d.set_ylabel("% of files this size or smaller")

    fig.suptitle(f"{title}Before-run check  (max {max_mbps:g} MB/s, per-thread {rate_mbps:g} MB/s)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(); fig.savefig(Path(out) / "0_prerun.png", dpi=130); plt.close(fig)
    return dict(over_max_pct=(pct > 100).mean() * 100, conc_p95=p95, conc_p99=p99, conc_peak=conc.max(),
                can_1min=(s_mb <= rate_mbps * 60).mean() * 100, can_5min=(s_mb <= rate_mbps * 300).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--max-mbps", type=float, required=True)
    ap.add_argument("--thread-mbps", type=float, required=True)
    ap.add_argument("--out", default="results/charts/prerun")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df = load(a.csv, a.date_from, a.date_to, 0)
    print(prerun(df, a.max_mbps, a.thread_mbps, a.out))


if __name__ == "__main__":
    main()
