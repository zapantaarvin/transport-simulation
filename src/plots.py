"""Charts for one simulation window. Saves PNGs into an output folder.

  python src/plots.py data/samples/big.parquet --from 2026-10-01 --to 2026-11-01 \
      --threads 8 --thread-mbps 10 --max-mbps 30 --sweep 2,4,8,16,32 --out results/charts

Makes:
  1_load_vs_max.png      hourly demand vs the max rate (is any hour over 100%?)
  2_latency_spread.png   how long files took, with the 1min/5min lines and % on time
  3_thread_sweep.png     % of files on time vs thread count, for each limit
  4_queue_busiest_day.png  files waiting or in transfer through the worst day
  5_sla_heatmap.png      day x hour: % of files that met the 1min / 5min limit
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transport_sim import MB, SLAS, TARGET, load, on_time, simulate, simulate_shared, verdict

# reference palette from the dataviz skill (light mode)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, CRITICAL = "#2a78d6", "#eb6834", "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2,
})


def latency(df, threads, thread_mbps, max_mbps, model="shared"):
    req = (df.request_time - df.request_time.iloc[0]).dt.total_seconds().to_numpy()
    size = df.file_size_bytes.to_numpy(float)
    if model == "shared":
        end = simulate_shared(req, size, threads, thread_mbps * MB, max_mbps * MB)
    else:
        end = simulate(req, size, threads, min(thread_mbps, max_mbps / threads) * MB)
    return req, end


def pct_down(v):
    """2 decimals, rounded down, so 99.996% shows as 99.99% (not a misleading 100.00%)."""
    return f"{np.floor(v * 100) / 100:.2f}"


def ref_line(ax, y=None, x=None, text="", color=INK2):
    if y is not None:
        ax.axhline(y, color=color, lw=1.2, ls="--")
        ax.annotate(text, (0, y), xycoords=("axes fraction", "data"), ha="left", va="bottom",
                    color=color, fontsize=9, xytext=(0, 3), textcoords="offset points")
    else:
        ax.axvline(x, color=color, lw=1.2, ls="--")
        ax.annotate(text, (x, 1), xycoords=("data", "axes fraction"), ha="left", va="top",
                    color=color, fontsize=9, xytext=(4, -4), textcoords="offset points")


def chart_load(sc, max_mbps, out):
    hourly = sc.groupby(sc.request_time.dt.floor("h")).file_size_bytes.sum() / MB / 3600
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hourly.index, hourly.values, color=BLUE, lw=1.2)
    ref_line(ax, y=max_mbps, text=f"max rate {max_mbps:g} MB/s", color=CRITICAL)
    over = hourly[hourly > max_mbps]
    if len(over):
        ax.scatter(over.index, over.values, s=24, color=CRITICAL, zorder=3, edgecolor=SURFACE, lw=1)
    ax.set_title(f"Demand per hour vs max rate  ({len(over)} hours over the max)")
    ax.set_ylabel("MB/s requested")
    fig.tight_layout(); fig.savefig(out / "1_load_vs_max.png", dpi=150); plt.close(fig)


TIME_TICKS = [(0.01, "0.01s"), (0.1, "0.1s"), (1, "1s"), (10, "10s"), (60, "1 min"), (120, "2 min"), (300, "5 min"), (600, "10 min"),
              (1800, "30 min"), (3600, "1 h"), (6 * 3600, "6 h"), (86400, "1 day")]


def time_axis(ax, lo, hi):
    """Log time axis labelled in plain units (1s, 1 min, 5 min) instead of 10^0, 10^2."""
    ax.set_xscale("log")
    ticks = [(v, t) for v, t in TIME_TICKS if lo <= v <= hi]
    ax.set_xticks([v for v, _ in ticks]); ax.set_xticklabels([t for _, t in ticks])
    ax.minorticks_off()


def chart_latency(lat, threads, out):
    """Top: every file, with the late zones shaded and counted. Bottom: zoom on only the late files,
    which are too few to see on the top panel's scale."""
    lat = np.maximum(lat, 0.01)
    one, five = SLAS["1min"], SLAS["5min"]
    n_ok, n_mid, n_late = (lat <= one).sum(), ((lat > one) & (lat <= five)).sum(), (lat > five).sum()
    met = on_time(lat)
    fig, (ax, zx) = plt.subplots(2, 1, figsize=(10, 7.5), gridspec_kw={"height_ratios": [3, 2], "hspace": 0.55})

    lo, hi = lat.min(), max(lat.max(), five * 1.5)
    ax.hist(lat, bins=np.logspace(np.log10(lo), np.log10(hi) + 0.01, 80), color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    time_axis(ax, lo, hi * 1.2)
    ax.axvspan(one, five, color=ORANGE, alpha=0.10, lw=0)
    ax.axvspan(five, hi * 1.2, color=CRITICAL, alpha=0.10, lw=0)
    for x in (one, five):
        ax.axvline(x, color=CRITICAL, lw=1.2, ls="--")
    ax.set_xlim(lo, hi * 1.2)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label=f"on time for both: {n_ok:,} files"),
                       Patch(color=ORANGE, alpha=0.35, label=f"late for 1 min only: {n_mid:,}"),
                       Patch(color=CRITICAL, alpha=0.35, label=f"late for 5 min too: {n_late:,}")],
              frameon=False, loc="upper left", fontsize=9)
    ax.set_title(f"How long each file took, request to done  ({threads} threads)\n"
                 + "   ".join(f"{name}: {pct_down(v)}% on time → {verdict(v)}" for name, v in met.items()),
                 fontsize=11)
    ax.set_ylabel("number of files")

    late = lat[lat > one]
    if late.size:
        zhi = max(late.max(), five * 1.5)
        bins = np.logspace(np.log10(one), np.log10(zhi) + 0.01, 40)
        zx.hist(late[late <= five], bins=bins, color=ORANGE, edgecolor=SURFACE, linewidth=0.5,
                label=f"late for 1 min only ({n_mid:,})")
        zx.hist(late[late > five], bins=bins, color=CRITICAL, edgecolor=SURFACE, linewidth=0.5,
                label=f"late for 5 min too ({n_late:,})")
        time_axis(zx, one, zhi * 1.2)
        zx.set_xlim(one * 0.95, zhi * 1.2)
        zx.axvline(five, color=CRITICAL, lw=1.2, ls="--")
        very = late[late > five]
        if very.size:  # too few to show as bars: one tick per file, plus a label
            ytop = zx.get_ylim()[1]
            zx.plot(very, np.full(very.size, ytop * 0.06), "|", color=CRITICAL, ms=16, mew=2)
            zx.annotate(f"{very.size:,} files past 5 min (one tick each)", (np.median(very), ytop * 0.12),
                        ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")
        zx.legend(frameon=False, loc="upper right", fontsize=9)
    else:
        zx.text(0.5, 0.5, "No late files: every file made 1 min", ha="center", va="center",
                transform=zx.transAxes, fontsize=11, color=INK2)
        zx.set_xticks([]); zx.set_yticks([])
    zx.set_title(f"Zoom: only the {late.size:,} late files (the PASS needs this to be empty)", fontsize=11)
    zx.set_xlabel("time from request to done"); zx.set_ylabel("number of files")
    fig.savefig(out / "2_latency_spread.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def chart_sweep(df, scored, sweep, thread_mbps, max_mbps, out):
    rows = []
    for n in sweep:
        req, end = latency(df, n, thread_mbps, max_mbps)
        rows.append(dict(threads=n, **{f"met_{k}": v for k, v in on_time((end - req)[scored]).items()}))
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, color in zip(SLAS, (ORANGE, BLUE)):
        ax.plot(r.threads, r[f"met_{name}"], color=color, marker="o", ms=8, mec=SURFACE, mew=2,
                label=f"{name} limit")
        last = r.iloc[-1]
        ax.annotate(f"{last[f'met_{name}']:.2f}%", (last.threads, last[f"met_{name}"]), color=INK2,
                    fontsize=9, xytext=(8, 0), textcoords="offset points", va="center")
    ref_line(ax, y=TARGET, text=f"PASS = {TARGET:g}% of files on time", color=CRITICAL)
    ax.set_xscale("log", base=2); ax.set_xticks(sweep); ax.set_xticklabels(sweep)
    ax.set_title(f"% of files on time vs number of threads  ({thread_mbps:g} MB/s each, max {max_mbps:g})")
    ax.set_xlabel("threads"); ax.set_ylabel("% of files on time")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "3_thread_sweep.png", dpi=150); plt.close(fig)
    return r


def worst_day(df, req, end, scored):
    """Day with the most files late for 5 min (1 min breaks ties, and picks the day if 5 min is clean)."""
    lat = pd.Series(end - req)[scored]
    days = df.request_time[scored].dt.date.to_numpy()
    late = pd.DataFrame({k: lat > s for k, s in SLAS.items()}).groupby(days).sum()
    return late.sort_values(["5min", "1min"], ascending=False).index[0]


def chart_queue(df, req, end, scored, out):
    worst = worst_day(df, req, end, scored)
    day0 = (pd.Timestamp(worst) - df.request_time.iloc[0]).total_seconds()
    t = np.arange(day0, day0 + 86400, 60)
    in_system = np.searchsorted(np.sort(req), t, "right") - np.searchsorted(np.sort(end), t, "right")
    fig, ax = plt.subplots(figsize=(11, 4))
    hours = (t - day0) / 3600
    ax.fill_between(hours, in_system, color=BLUE, alpha=0.25, lw=0)
    ax.plot(hours, in_system, color=BLUE, lw=1.5)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3))
    ax.set_title(f"Files waiting or transferring, worst day ({worst})")
    ax.set_xlabel("hour of day"); ax.set_ylabel("files in system")
    fig.tight_layout(); fig.savefig(out / "4_queue_busiest_day.png", dpi=150); plt.close(fig)


def chart_sla_heatmap(df, req, end, scored, threads, out, target=TARGET, floor=80.0):
    """Day x hour grid: % of files requested in that hour that finished within each limit."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", [CRITICAL, "#f0efec", BLUE])
    cmap.set_bad(GRID)  # hours with no files
    norm = TwoSlopeNorm(vmin=floor, vcenter=target - 0.01, vmax=100)  # only a PASS is blue
    t = df.request_time[scored]
    lat = (end - req)[scored]
    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, 0.26 * t.dt.date.nunique() + 2)), sharey=True,
                             gridspec_kw={"wspace": 0.18})
    for ax, (name, limit) in zip(axes, SLAS.items()):
        met = pd.DataFrame({"day": t.dt.date.to_numpy(), "hour": t.dt.hour.to_numpy(), "ok": lat <= limit})
        grid = met.pivot_table(index="day", columns="hour", values="ok", aggfunc="mean").reindex(columns=range(24)) * 100
        daily = met.groupby("day").ok.mean() * 100
        im = ax.imshow(np.ma.masked_invalid(grid.to_numpy()), aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xticks(range(0, 24, 3)); ax.set_xticklabels([f"{h}:00" for h in range(0, 24, 3)])
        ax.set_yticks(range(len(grid))); ax.set_yticklabels([d.strftime("%a %b %d") for d in grid.index], fontsize=8)
        ax.grid(False)
        ax.set_xticks(np.arange(-.5, 24, 1), minor=True); ax.set_yticks(np.arange(-.5, len(grid), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.5); ax.tick_params(which="minor", length=0)
        for y, d in enumerate(grid.index):  # day total on the right
            ax.text(24.2, y, f"{pct_down(daily[d])}%", va="center", fontsize=8,
                    color=CRITICAL if daily[d] < target else INK, fontweight="bold")
        overall = met.ok.mean() * 100
        ax.set_title(f"{name} limit: {pct_down(overall)}% of files on time, {verdict(overall)}\n"
                     f"{(grid < target).sum().sum()} of {grid.notna().sum().sum()} hours under {target:g}%   "
                     f"(right column = whole day)", fontsize=11, loc="left")
        ax.set_xlabel("hour requested")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.06)
    cb.set_label(f"% of files on time  (blue = {target:g}%, red below, ≤ {floor:g}% darkest)")
    cb.set_ticks(sorted({floor, 90, 95, 99, target, 100}))
    fig.suptitle(f"Share of files finishing in time, by day and hour  (N={threads})", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.savefig(out / "5_sla_heatmap.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--from", dest="date_from", required=True)
    ap.add_argument("--to", dest="date_to", required=True)
    ap.add_argument("--warmup-min", type=float, default=60)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--thread-mbps", type=float, default=10)
    ap.add_argument("--max-mbps", type=float, default=80)
    ap.add_argument("--sweep", default="2,4,8,16,32")
    ap.add_argument("--out", default="results/charts")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = load(a.csv, a.date_from, a.date_to, a.warmup_min)
    scored = df.scored.to_numpy()
    chart_load(df[df.scored], a.max_mbps, out)
    req, end = latency(df, a.threads, a.thread_mbps, a.max_mbps)
    chart_latency((end - req)[scored], a.threads, out)
    chart_queue(df, req, end, scored, out)
    chart_sla_heatmap(df, req, end, scored, a.threads, out)
    r = chart_sweep(df, scored, [int(x) for x in a.sweep.split(",")], a.thread_mbps, a.max_mbps, out)
    r.to_csv(out / "thread_sweep.csv", index=False)
    print(f"saved 5 charts + thread_sweep.csv to {out}")


if __name__ == "__main__":
    main()
