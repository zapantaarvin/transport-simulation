"""NAPM-style output: sweep the bandwidth (comm_mbps) and report latency percentiles per day.

  python src/napm_output.py data/samples/big.parquet --from 2026-10-01 --to 2026-11-01 \
      --comm 20,40,60,80,120,160 --threads 8 --out results/napm/sample
  python src/napm_output.py all                       # every scenario site, comm = 0.25x .. 3x its max

comm_mbps is the total bandwidth (the max rate), in the same MB/s as --max-mbps everywhere else.
Threads stay fixed; --thread-mbps caps one thread (default: no cap, one thread can use all of comm).

Makes, per data file:
  7_napm_heatmap.png        % of days whose pXX latency is under 5 min / 1 min   (y comm_mbps, x p50..p99)
  8_bandwidth_timeline.png  pXX latency in minutes vs comm_mbps (p90, p95, p99) + table of the
                            smallest comm_mbps that gets each percentile under 5 min / 1 min
  9_p90_daily.png           each day's p90 latency, one line per comm_mbps
  napm_daily.csv            day, comm_mbps, p50..p99 (minutes)
  napm_bandwidth.csv        percentile, bandwidth_5min, bandwidth_1min
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import plots  # chart style
from plots import CRITICAL, INK2, ORANGE, plt
from scenarios import SITES
from transport_sim import MB, SLAS, load, simulate_shared

PCTS = [50, 75, 90, 95, 99]
LINES = [90, 95, 99]
LIMIT_ORDER = ["5min", "1min"]


def sweep(df, comms, threads, thread_mbps):
    """Latency percentiles per day and over the whole window, for each comm_mbps."""
    req = (df.request_time - df.request_time.iloc[0]).dt.total_seconds().to_numpy()
    size = df.file_size_bytes.to_numpy(float)
    scored = df.scored.to_numpy()
    day = df.request_time.dt.normalize()[scored].to_numpy()
    daily, overall = [], []
    for c in comms:
        rate = c if thread_mbps is None else min(thread_mbps, c)
        lat = (simulate_shared(req, size, threads, rate * MB, c * MB) - req)[scored] / 60
        d = pd.DataFrame({"day": day, "lat": lat}).groupby("day").lat
        p = pd.concat({f"p{q}": d.quantile(q / 100) for q in PCTS}, axis=1).reset_index()
        p.insert(1, "comm_mbps", c)
        daily.append(p)
        overall.append({"comm_mbps": c, **{f"p{q}": np.percentile(lat, q) for q in PCTS}})
    return pd.concat(daily, ignore_index=True), pd.DataFrame(overall)


def days_under(daily, limit_min):
    """% of days whose pXX latency is within the limit, as a comm_mbps x percentile table."""
    cols = [f"p{q}" for q in PCTS]
    return (daily[cols] <= limit_min).groupby(daily.comm_mbps).mean() * 100


def bandwidth_table(overall):
    """Smallest comm_mbps tested where the whole-window pXX is within each limit."""
    rows = []
    for q in LINES:
        row = {"percentile": f"p{q}"}
        for name in LIMIT_ORDER:
            ok = overall[overall[f"p{q}"] <= SLAS[name] / 60]
            row[f"bandwidth_{name}"] = f"{ok.comm_mbps.min():g}" if len(ok) else f"> {overall.comm_mbps.max():g}"
        rows.append(row)
    return pd.DataFrame(rows)


def chart_heatmap(daily, out, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.5 * daily.comm_mbps.nunique() + 2.6))
    for ax, name in zip(axes, LIMIT_ORDER):
        t = days_under(daily, SLAS[name] / 60).sort_index(ascending=False)
        ax.imshow(t.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.iat[i, j]
                ax.text(j, i, f"{np.floor(v):.0f}%", ha="center", va="center", fontsize=10,
                        color="white" if v > 60 else plots.INK, fontweight="bold" if v >= 100 else "normal")
        ax.set_xticks(range(len(PCTS))); ax.set_xticklabels([f"p{q}" for q in PCTS])
        ax.set_yticks(range(t.shape[0])); ax.set_yticklabels([f"{c:g}" for c in t.index])
        ax.set_xlabel("percentile of each day's latency"); ax.set_ylabel("comm_mbps")
        ax.set_title(f"% of days under {name[0]} min", fontsize=11)
        ax.grid(False)
    fig.suptitle(f"{title}% of days where that day's percentile latency is under the limit",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(); fig.savefig(Path(out) / "7_napm_heatmap.png", dpi=130); plt.close(fig)


def chart_bandwidth(overall, table, out, title):
    fig, (ax, tx) = plt.subplots(2, 1, figsize=(10, 7.2), gridspec_kw={"height_ratios": [4, 1.1]})
    shades = plt.get_cmap("Blues")(np.linspace(0.45, 0.95, len(LINES)))
    for q, col in zip(LINES, shades):
        ax.plot(overall.comm_mbps, overall[f"p{q}"], marker="o", ms=6, color=col, label=f"p{q}")
    ax.set_yscale("log")
    ax.set_ylim(min(overall[[f"p{q}" for q in LINES]].min().min() * 0.7, 0.5), None)
    for name, col in (("5min", ORANGE), ("1min", CRITICAL)):
        plots.ref_line(ax, y=SLAS[name] / 60, text=f"{name[0]} min", color=col)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("bandwidth, comm_mbps"); ax.set_ylabel("latency, minutes (log scale)")
    ax.legend(title="percentile", frameon=False)
    ax.set_title("Latency at each bandwidth (all files in the window)", fontsize=11)
    tx.axis("off")
    cells = [[r.percentile, r.bandwidth_5min, r.bandwidth_1min] for r in table.itertuples()]
    tb = tx.table(cellText=cells, colLabels=["percentile", "bandwidth for 5 min", "bandwidth for 1 min"],
                  loc="center", cellLoc="center")
    tb.scale(1, 1.4)
    for (r, _), cell in tb.get_celld().items():
        cell.set_edgecolor(plots.GRID)
        if r == 0:
            cell.set_text_props(fontweight="bold", color=INK2)
    fig.suptitle(f"{title}Bandwidth timeline: how bandwidth changes latency",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(); fig.savefig(Path(out) / "8_bandwidth_timeline.png", dpi=130); plt.close(fig)


def chart_p90_daily(daily, out, title):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    comms = sorted(daily.comm_mbps.unique())
    shades = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, len(comms)))
    for c, col in zip(comms, shades):
        d = daily[daily.comm_mbps == c]
        ax.plot(d.day, d.p90, marker="o", ms=4, lw=1.6, color=col, label=f"{c:g}")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    for name, col in (("5min", ORANGE), ("1min", CRITICAL)):
        plots.ref_line(ax, y=SLAS[name] / 60, text=f"{name[0]} min", color=col)
    ax.set_xlabel("day"); ax.set_ylabel("p90 latency, minutes (log scale)")
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend(title="comm_mbps", frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle(f"{title}90p daily timeline: each day's p90 latency, one line per bandwidth",
                 x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(); fig.savefig(Path(out) / "9_p90_daily.png", dpi=130); plt.close(fig)


def make(df, comms, threads, thread_mbps, out, title=""):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    daily, overall = sweep(df, comms, threads, thread_mbps)
    table = bandwidth_table(overall)
    daily.assign(day=daily.day.dt.date).round(3).to_csv(out / "napm_daily.csv", index=False)
    table.to_csv(out / "napm_bandwidth.csv", index=False)
    chart_heatmap(daily, out, title)
    chart_bandwidth(overall, table, out, title)
    chart_p90_daily(daily, out, title)
    print(f"\n{title or out.name}  ({threads} threads, comm_mbps {', '.join(f'{c:g}' for c in comms)})")
    print(table.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="'all' for every scenario site, or a .parquet/.csv file")
    ap.add_argument("--from", dest="date_from", default="2026-10-01")
    ap.add_argument("--to", dest="date_to", default="2026-11-01")
    ap.add_argument("--comm", help="comm_mbps values to test, e.g. 20,40,80 (required for a single file)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--thread-mbps", type=float, help="per-thread cap (default: none)")
    ap.add_argument("--out", default="results/napm/run")
    a = ap.parse_args()

    if a.data == "all":
        for name, cfg in SITES.items():
            df = load(f"data/scenarios/{name}.parquet", a.date_from, a.date_to, 60)
            comms = [cfg["max_mbps"] * k for k in (0.25, 0.5, 1, 1.5, 2, 3)]
            make(df, comms, a.threads, a.thread_mbps, Path("results/napm") / name, f"{name}: ")
    else:
        if not a.comm:
            raise SystemExit("--comm is required for a single file, e.g. --comm 20,40,80")
        df = load(a.data, a.date_from, a.date_to, 60)
        make(df, [float(x) for x in a.comm.split(",")], a.threads, a.thread_mbps, a.out)


if __name__ == "__main__":
    main()
