"""Set threads + per-thread rate from the concurrent-FTP curve: +1 when a file starts, -1 when it finishes.

  python src/concurrency.py all                                   # every site in data/scenarios
  python src/concurrency.py data/samples/big.parquet --from 2026-10-01 --to 2026-11-01 --max-mbps 80

The idea, per limit (1 min = 60s, 5 min = 300s) and per candidate per-thread rate r:
  1. r must be >= biggest file / limit, or that file is late even with no wait.
  2. Pretend every file starts the moment it's requested and runs at r: start = request,
     end = request + size / r. Sweep the starts (+1) and ends (-1) in time order.
     The running total is how many transfers run at once.
  3. Peak of that curve = threads needed so no file ever waits.
     Peak x r = bandwidth needed so no thread is ever slowed down. Must fit under the max rate.
  4. If both hold, every file finishes within size / r <= limit: a guaranteed PASS.
     The simulator then checks it, and finds how many threads are really needed
     (files can wait a little and still make the limit, so it's usually fewer).
If the data has real start_time / end_time, the same sweep on those columns shows the
concurrency production actually ran at (the observed curve).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import plots  # chart style
from plots import BLUE, CRITICAL, INK2, ORANGE, plt, ref_line
from scenarios import RATES, SITES
from transport_sim import MB, SLAS, TARGET, has_times, load, on_time, simulate_shared, timed_mask


def concurrency(start, end):
    """Sweep line: +1 at each start, -1 at each end. Returns (event times, transfers running after each event).
    At equal times the -1 goes first: a thread freed at t can take a new file at t."""
    t = np.concatenate([start, end])
    step = np.concatenate([np.ones(start.size), -np.ones(end.size)])
    order = np.lexsort((step, t))
    return t[order], np.cumsum(step[order])


def min_threads(req, size, rate_bps, max_bps, scored, limit, hi):
    """Fewest threads (1..hi) where every scored file makes the limit. Scans up one at a time:
    no binary search, because past max/rate more threads can make it worse (they split the pipe)."""
    for n in range(1, hi + 1):
        if on_time((simulate_shared(req, size, n, rate_bps, max_bps) - req)[scored])[limit] >= TARGET:
            return n
    return None


def size_from_curve(df, max_mbps, rates=RATES):
    """One row per (limit, rate): curve peak, bandwidth needed, simulated check."""
    req = (df.request_time - df.request_time.iloc[0]).dt.total_seconds().to_numpy()
    size = df.file_size_bytes.to_numpy(float)
    scored = df.scored.to_numpy()
    biggest = size[scored].max() / MB
    rows = []
    for limit, secs in SLAS.items():
        need_rate = biggest / secs
        for r in rates:
            if r > max_mbps:
                continue
            row = dict(limit=limit, rate=r, rate_ok=r >= need_rate)
            tt, level = concurrency(req, req + size / (r * MB))
            peak = int(level.max())
            ceiling = max_mbps / r  # transfers the pipe can run at full speed at once
            held = np.diff(tt, append=tt[-1])  # how long each level lasted
            over = held[level > ceiling].sum() / held.sum() * 100
            row.update(peak_threads=peak, bandwidth_needed=peak * r, pipe_ok=peak * r <= max_mbps,
                       time_over_pipe_pct=over)
            row["guaranteed"] = row["rate_ok"] and row["pipe_ok"]
            met = on_time((simulate_shared(req, size, peak, r * MB, max_mbps * MB) - req)[scored])[limit]
            row["sim_on_time_at_peak"] = met
            row["sim_min_threads"] = (min_threads(req, size, r * MB, max_mbps * MB, scored, limit, min(peak, 64))
                                      if row["rate_ok"] else None)
            rows.append(row)
    return pd.DataFrame(rows), biggest


def recommend(t, limit):
    """Per limit: a guaranteed row with the fewest threads; else the simulated pass with fewest threads."""
    d = t[t.limit == limit]
    g = d[d.guaranteed]
    if len(g):
        return g.sort_values(["peak_threads", "rate"]).iloc[0], "guaranteed"
    s = d[d.sim_min_threads.notna()]
    if len(s):
        return s.sort_values(["sim_min_threads", "rate"]).iloc[0], "simulated"
    return None, None


def chart(df, t, max_mbps, out, title):
    """Concurrent FTP over the month at the recommended rate for each limit (hourly peaks)."""
    from matplotlib.ticker import MaxNLocator
    req = (df.request_time - df.request_time.iloc[0]).dt.total_seconds().to_numpy()
    size = df.file_size_bytes.to_numpy(float)
    t0 = df.request_time.iloc[0]
    observed = has_times(df)
    panels = list(SLAS) + (["observed"] if observed else [])
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 3.6 * len(panels)), sharex=True)
    for ax, limit in zip(np.atleast_1d(axes), panels):
        if limit == "observed":
            tm = df[timed_mask(df)]
            s = (tm.start_time - t0).dt.total_seconds().to_numpy()
            e = (tm.end_time - t0).dt.total_seconds().to_numpy()
            name = f"Observed: real start/end times ({len(tm) / len(df):.0%} of files have them)"
        else:
            row, how = recommend(t, limit)
            if row is None:  # nothing passes: draw the rate that came closest
                d = t[t.limit == limit]
                d = d[d.rate_ok] if d.rate_ok.any() else d
                row = d.sort_values(["sim_on_time_at_peak", "rate"]).iloc[-1]
            r = row.rate
            s, e = req, req + size / (r * MB)
            name = f"{limit}: every file starts at request, runs at {r:g} MB/s"
        tt, level = concurrency(s, e)
        b = (tt // 3600).astype(int)
        peaks = pd.Series(level).groupby(b).max()
        peaks = peaks.reindex(range(b.max() + 1)).ffill().fillna(0)  # hours with no events keep the level
        when = t0 + pd.to_timedelta(peaks.index.to_numpy() * 3600, unit="s")
        ax.fill_between(when, peaks.to_numpy(), step="post", color=BLUE, alpha=0.25, lw=0)
        ax.step(when, peaks.to_numpy(), where="post", color=BLUE, lw=1)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(0, None)
        peak = int(level.max())
        ax.scatter([t0 + pd.to_timedelta(tt[level.argmax()], unit="s")], [peak], s=40, color=ORANGE, zorder=3)
        ax.annotate(f"peak {peak}", (t0 + pd.to_timedelta(tt[level.argmax()], unit="s"), peak), color=INK2,
                    fontsize=9, xytext=(6, -2), textcoords="offset points", va="top")
        if limit != "observed":
            ceiling = max_mbps / r
            ax.axhline(ceiling, color=CRITICAL, lw=1.2, ls="--")
            ax.annotate(f"pipe holds {ceiling:.1f} at full speed", (1, ceiling), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", color=CRITICAL, fontsize=9, xytext=(0, 3), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2", fc=plots.SURFACE, ec="none"))
            verdict = "rate too slow for the biggest file" if not row.rate_ok else \
                ("fits the pipe → guaranteed PASS" if row.pipe_ok else
                 f"over the pipe {row.time_over_pipe_pct:.2f}% of the time")
            name += f"   ({verdict})"
        ax.set_title(name, fontsize=11)
        ax.set_ylabel("transfers running")
    np.atleast_1d(axes)[-1].tick_params(axis="x", labelrotation=30)
    fig.suptitle(f"{title}Concurrent FTP over time  (+1 at start, -1 at finish, peak per hour)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(Path(out) / "6_concurrency.png", dpi=130)
    plt.close(fig)


def describe(name, t, biggest, max_mbps):
    print(f"\n{name}  (max {max_mbps:g} MB/s, biggest file {biggest:,.0f} MB)")
    print(f"  {'limit':<6}{'rate':>6}{'peak thr':>10}{'bw needed':>11}{'time over pipe':>16}"
          f"{'sim @peak':>11}{'sim min thr':>13}")
    for row in t.itertuples():
        mn = "-" if pd.isna(row.sim_min_threads) else f"{int(row.sim_min_threads)}"
        flag = "" if row.rate_ok else "   rate < biggest/limit"
        print(f"  {row.limit:<6}{row.rate:>6g}{row.peak_threads:>10}{row.bandwidth_needed:>11,.0f}"
              f"{row.time_over_pipe_pct:>15.3f}%{np.floor(row.sim_on_time_at_peak * 100) / 100:>10.2f}%"
              f"{mn:>13}{flag}")
    out = {}
    for limit in SLAS:
        row, how = recommend(t, limit)
        if row is None:
            d = t[(t.limit == limit) & t.rate_ok]
            if len(d):  # rate is fine; bursts overflow the pipe
                best = d.sort_values("sim_on_time_at_peak").iloc[-1]
                print(f"  -> {limit}: nothing passes. Rate is fast enough; the curve is over the pipe "
                      f"{best.time_over_pipe_pct:.2f}% of the time at {best.rate:g} MB/s -> raise the max rate "
                      f"or spread the bursts.")
            else:
                print(f"  -> {limit}: nothing passes. Per-thread rate must be >= {biggest / SLAS[limit]:,.0f} MB/s "
                      f"(biggest file ÷ {SLAS[limit]}s), more than the max rate allows.")
            out[limit] = "none"
        else:
            n = int(row.peak_threads) if how == "guaranteed" else int(row.sim_min_threads)
            extra = f", simulator says {int(row.sim_min_threads)} is enough" if how == "guaranteed" and \
                pd.notna(row.sim_min_threads) else ""
            print(f"  -> {limit}: {n} threads @ {row.rate:g} MB/s ({how}{extra})")
            out[limit] = f"{n} @ {row.rate:g} ({how}{extra})"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="'all' for every scenario site, or a .parquet/.csv file")
    ap.add_argument("--from", dest="date_from", default="2026-10-01")
    ap.add_argument("--to", dest="date_to", default="2026-11-01")
    ap.add_argument("--max-mbps", type=float, help="required for a single file")
    ap.add_argument("--rates", default=",".join(map(str, RATES)), help="per-thread rates to try, MB/s")
    ap.add_argument("--out", default="results/charts/concurrency")
    a = ap.parse_args()
    rates = [float(x) for x in a.rates.split(",")]

    if a.data == "all":
        rows = []
        for name, cfg in SITES.items():
            df = load(f"data/scenarios/{name}.parquet", a.date_from, a.date_to, 0)
            t, biggest = size_from_curve(df, cfg["max_mbps"], rates)
            out = Path("results/scenarios") / name
            out.mkdir(parents=True, exist_ok=True)
            t.to_csv(out / "concurrency_sizing.csv", index=False)
            chart(df, t, cfg["max_mbps"], out, f"{name}: ")
            rows.append(dict(site=name, max=cfg["max_mbps"], **describe(name, t, biggest, cfg["max_mbps"])))
        pd.DataFrame(rows).to_csv("results/scenarios/concurrency_summary.csv", index=False)
    else:
        if a.max_mbps is None:
            raise SystemExit("--max-mbps is required for a single file")
        df = load(a.data, a.date_from, a.date_to, 60)
        t, biggest = size_from_curve(df, a.max_mbps, rates)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        t.to_csv(Path(a.out) / "concurrency_sizing.csv", index=False)
        chart(df[df.scored], t, a.max_mbps, a.out, "")
        describe(Path(a.data).name, t, biggest, a.max_mbps)


if __name__ == "__main__":
    main()
