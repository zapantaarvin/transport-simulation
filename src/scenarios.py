"""Synthetic site scenarios: generate 50k requests per site, then search settings for each.

  python src/scenarios.py make                 # write data/scenarios/<site>.csv
  python src/scenarios.py run                  # settings grid + charts into results/scenarios/<site>/
  python src/scenarios.py make --sizes uniform # literal uniform 0.1 MB - 15 GB sizes instead of log-uniform

Traffic is built as a demand "shape" per minute over 30 days (Oct 1-30), then exactly N request
times are drawn from that shape. Rates are MB/s. Sizes are clipped to 0.1 MB - 15 GB.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import plots
from plots import BLUE, CRITICAL, GRID, INK, INK2, SURFACE, plt
from prerun import prerun
from transport_sim import MB, SLAS, TARGET, load, on_time, simulate_shared, write_table

DAYS, MIN_PER_DAY = 30, 1440
M = DAYS * MIN_PER_DAY
START = pd.Timestamp("2026-10-01")
SIZE_MIN, SIZE_MAX = 0.1 * MB, 15_000 * MB
THREADS = [1, 2, 4, 8, 16, 32, 64]
RATES = [25, 50, 100, 200, 300, 500]  # per-thread MB/s to test
PRERUN_RATE = 100  # per-thread MB/s assumed for the before-run charts


# ---------- traffic shape building blocks (all return one multiplier per minute) ----------
def minutes():
    m = np.arange(M)
    return m // MIN_PER_DAY, (m % MIN_PER_DAY) / 60  # day index, hour of day (float)


def random_regimes(rng, mean_len_min, sigma):
    """Piecewise-constant random multiplier: the 'super random' part."""
    out, i = np.empty(M), 0
    while i < M:
        n = max(1, int(rng.exponential(mean_len_min)))
        out[i:i + n] = rng.lognormal(0, sigma)
        i += n
    return out


def spikes(rng, per_day, len_range, height_range, decay_days=None):
    """Short bursts on top of the shape; optionally shrinking over the month."""
    out = np.ones(M)
    for _ in range(rng.poisson(per_day * DAYS)):
        s = rng.integers(0, M)
        n = rng.integers(*len_range)
        h = rng.uniform(*height_range)
        if decay_days:
            h = 1 + (h - 1) * np.exp(-(s / MIN_PER_DAY) / decay_days)
        out[s:s + n] *= h
    return out


def log_uniform(rng, n):
    return np.exp(rng.uniform(np.log(SIZE_MIN), np.log(SIZE_MAX), n))


# ---------- the sites ----------
def site_a(rng, n, sizes):
    """Max 1000. Active 24h, week 1 busier, very random regimes + spikes."""
    day, _ = minutes()
    shape = np.where(day < 7, 2.5, 1.0) * random_regimes(rng, 45, 1.0) * spikes(rng, 3, (5, 30), (5, 20))
    return shape, sizes(rng, n)


def site_b(rng, n, sizes):
    """Max 500. Two-day burst, then mellows; random bursts keep coming but shrink."""
    day, _ = minutes()
    d = day + (np.arange(M) % MIN_PER_DAY) / MIN_PER_DAY
    base = np.where(d < 2, 15.0, 0.4 + 4 * np.exp(-(d - 2) / 6))
    shape = base * random_regimes(rng, 60, 0.6) * spikes(rng, 1.2, (20, 180), (3, 10), decay_days=10)
    return shape, sizes(rng, n)


def site_c(rng, n, sizes):
    """Max 300. Office pattern: weekday 9-5 peak, quiet nights and weekends."""
    day, hour = minutes()
    weekday = (START + pd.to_timedelta(day, "D")).day_of_week < 5
    shape = np.where(weekday & (hour >= 9) & (hour < 17), 6.0, np.where(weekday, 1.0, 0.3))
    shape = shape * random_regimes(rng, 30, 0.3)
    s = rng.lognormal(np.log(150 * MB), 1.5, n)
    return shape, s


def site_d(rng, n, sizes):
    """Max 800. Nightly batch 1:00-2:30 plus a month-end surge on the last 2 days."""
    day, hour = minutes()
    shape = np.ones(M) * np.where((hour >= 1) & (hour < 2.5), 40.0, 1.0) * np.where(day >= DAYS - 2, 3.0, 1.0)
    shape = shape * random_regimes(rng, 30, 0.3)
    s = rng.lognormal(np.log(500 * MB), 1.3, n)
    return shape, s


def site_e(rng, n, sizes):
    """Max 400. Steady traffic, mixed sizes: 90% small (<50 MB), 10% huge (2-15 GB)."""
    shape = random_regimes(rng, 30, 0.5)
    small = np.exp(rng.uniform(np.log(0.1 * MB), np.log(50 * MB), n))
    huge = np.exp(rng.uniform(np.log(2_000 * MB), np.log(SIZE_MAX), n))
    s = np.where(rng.random(n) < 0.9, small, huge)
    return shape, s


def site_f(rng, n, sizes):
    """Max 600. Normal 24h traffic with a daytime hump; outages are applied afterwards (outage_backlog)."""
    _, hour = minutes()
    shape = (1 + 1.5 * np.exp(-((hour - 13) / 4) ** 2)) * random_regimes(rng, 40, 0.5)
    return shape, sizes(rng, n)


def outage_backlog(rng, counts, n_outages=4, hours=(2, 10), rush_min=(20, 60)):
    """Power outages on random days: no requests during the outage, then everything that would have
    arrived lands in a rush right after power returns (front-loaded, fading over rush_min minutes)."""
    counts = counts.copy()
    days = np.sort(rng.choice(np.arange(1, DAYS - 1), n_outages, replace=False))
    log = []
    for d in days:
        start = d * MIN_PER_DAY + rng.integers(0, 14 * 60)
        length = int(rng.uniform(*hours) * 60)
        held = counts[start:start + length].sum()
        counts[start:start + length] = 0
        rush = rng.integers(*rush_min)
        w = np.exp(-np.arange(rush) / (rush / 4))  # most of the backlog in the first quarter
        counts[start + length:start + length + rush] += rng.multinomial(held, w / w.sum())
        log.append(dict(outage_start=START + pd.Timedelta(minutes=int(start)), hours=round(length / 60, 1),
                        files_held=int(held), rush_minutes=int(rush)))
    return counts, pd.DataFrame(log)


SITES = {
    "site-a": dict(fn=site_a, max_mbps=1000, about="24h, week 1 busier, super random"),
    "site-b": dict(fn=site_b, max_mbps=500, about="2-day burst then mellowing random bursts"),
    "site-c": dict(fn=site_c, max_mbps=300, about="office hours weekdays, quiet weekends"),
    "site-d": dict(fn=site_d, max_mbps=800, about="nightly batch + month-end surge"),
    "site-e": dict(fn=site_e, max_mbps=400, about="steady, 90% small + 10% huge files"),
    "site-f": dict(fn=site_f, max_mbps=600, about="power outages on random days, backlog rush after",
                   post=outage_backlog),
}


def make(n, size_mode, out_dir, seed=2026):
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = log_uniform if size_mode == "log" else \
        (lambda rng, k: rng.uniform(SIZE_MIN, SIZE_MAX, k))
    for i, (name, cfg) in enumerate(SITES.items()):
        rng = np.random.default_rng(seed + i)
        shape, size = cfg["fn"](rng, n, sizes)
        counts = rng.multinomial(n, shape / shape.sum())  # exactly n requests, following the shape
        if "post" in cfg:
            counts, events = cfg["post"](rng, counts)
            events.to_csv(out_dir / f"{name}-events.csv", index=False)
            print(f"{name} events:\n{events.to_string(index=False)}")
        minute = np.repeat(np.arange(M), counts)
        secs = np.sort(minute * 60 + rng.uniform(0, 60, n))
        size = np.clip(size, SIZE_MIN, SIZE_MAX).astype(np.int64)
        df = pd.DataFrame({"file_size_bytes": rng.permutation(size),
                           "request_time": START + pd.to_timedelta(secs, unit="s")})
        write_table(df, out_dir / f"{name}.parquet")
        print(f"{name}: {len(df):,} files, total {size.sum() / 1e12:.1f} TB, "
              f"median {np.median(size) / MB:,.0f} MB, p99 {np.percentile(size, 99) / MB:,.0f} MB")


# ---------- settings grid ----------
def grid_search(df, max_mbps):
    req = (df.request_time - df.request_time.iloc[0]).dt.total_seconds().to_numpy()
    size = df.file_size_bytes.to_numpy(float)
    scored = df.scored.to_numpy()
    rows = []
    for n in THREADS:
        for r in RATES:
            if r > max_mbps:
                continue
            lat = (simulate_shared(req, size, n, r * MB, max_mbps * MB) - req)[scored]
            rows.append(dict(threads=n, rate=r, **{f"met_{k}": v for k, v in on_time(lat).items()}))
    return pd.DataFrame(rows)


def pick(g, sla_name):
    """Fewest threads, then lowest per-thread rate, where at least TARGET % of files are on time."""
    ok = g[g[f"met_{sla_name}"] >= TARGET].sort_values(["threads", "rate"])
    return None if ok.empty else ok.iloc[0]


def chart_grid(g, name, max_mbps, out):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", [CRITICAL, "#f0efec", BLUE])
    cmap.set_bad(GRID)
    norm = TwoSlopeNorm(vmin=80, vcenter=TARGET - 0.01, vmax=100)  # only a PASS is blue
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, sla in zip(axes, SLAS):
        piv = g.pivot(index="threads", columns="rate", values=f"met_{sla}").reindex(index=THREADS, columns=RATES)
        ax.imshow(np.ma.masked_invalid(piv.to_numpy()), cmap=cmap, norm=norm, aspect="auto")
        ax.grid(False)
        for y, n in enumerate(THREADS):
            for x, r in enumerate(RATES):
                v = piv.loc[n, r]
                if np.isnan(v):
                    ax.text(x, y, "over max", ha="center", va="center", fontsize=7.5, color=INK2)
                    continue
                passed = v >= TARGET
                ax.text(x, y, ("100% ✓" if passed and v == 100 else f"{np.floor(v * 10) / 10:.1f}%"
                               + (" ✓" if passed else "")), ha="center", va="center",
                        fontsize=8.5, color=SURFACE if (passed or v <= 86) else INK,
                        fontweight="bold" if passed else "normal")
        ax.set_xticks(range(len(RATES))); ax.set_xticklabels([f"{r}" for r in RATES])
        ax.set_yticks(range(len(THREADS))); ax.set_yticklabels(THREADS)
        ax.set_xlabel("per-thread rate (MB/s)")
        ax.set_title(f"{sla} limit: % of files on time   (✓ = PASS, {TARGET:g}% on time)", fontsize=11, loc="left")
    axes[0].set_ylabel("threads")
    fig.suptitle(f"{name}: which settings work  (max {max_mbps} MB/s)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "settings_grid.png", dpi=150); plt.close(fig)


def run(data_dir, res_dir):
    summary = []
    for name, cfg in SITES.items():
        mx = cfg["max_mbps"]
        df = load(data_dir / f"{name}.parquet", "2026-10-01", "2026-11-01", 0)
        out = res_dir / name
        out.mkdir(parents=True, exist_ok=True)
        sc = df[df.scored]
        plots.chart_load(sc, mx, out)
        pre = prerun(df, mx, PRERUN_RATE, out, title=f"{name}: ")
        per5 = sc.groupby(sc.request_time.dt.floor("5min")).file_size_bytes.sum() / MB / 300
        hourly = sc.groupby(sc.request_time.dt.floor("h")).file_size_bytes.sum() / MB / 3600
        avg = sc.file_size_bytes.sum() / MB / (DAYS * 86400)
        g = grid_search(df, mx)
        g.to_csv(out / "settings_grid.csv", index=False)
        chart_grid(g, name, mx, out)
        line = dict(site=name, about=cfg["about"], max=mx, avg_load=f"{avg / mx:.0%}",
                    busiest_hour=f"{hourly.max() / mx:.0%}", busiest_5min=f"{per5.max() / mx:.0%}",
                    biggest_gb=round(sc.file_size_bytes.max() / 1e9, 1),
                    over_max_time=f"{pre['over_max_pct']:.2f}%", conc_p95=round(pre['conc_p95']),
                    conc_p99=round(pre['conc_p99']), can_1min=f"{pre['can_1min']:.1f}%")
        for sla in SLAS:
            best = pick(g, sla)
            line[f"best_{sla}"] = "none" if best is None else \
                f"{int(best.threads)} thr @ {int(best.rate)} MB/s"
            top = g.sort_values([f"met_{sla}", "threads"], ascending=[False, True]).iloc[0]
            line[f"top_{sla}"] = f"{np.floor(top[f'met_{sla}'] * 100) / 100:.2f}%"
        # day x hour charts at the 5min pick (or the strongest setting if nothing passes)
        best = pick(g, "5min")
        b = best if best is not None else g.sort_values(["met_5min", "met_1min", "threads"],
                                                          ascending=[False, False, True]).iloc[0]
        req, end = plots.latency(df, int(b.threads), float(b.rate), mx)
        sc_mask = df.scored.to_numpy()
        plots.chart_latency((end - req)[sc_mask], int(b.threads), out)
        plots.chart_sla_heatmap(df, req, end, sc_mask, int(b.threads), out)
        plots.chart_queue(df, req, end, sc_mask, out)
        line["charted_at"] = f"{int(b.threads)} thr @ {int(b.rate)} MB/s"
        summary.append(line)
        print(f"{name} done")
    s = pd.DataFrame(summary)
    s.to_csv(res_dir / "summary.csv", index=False)
    print("\n" + s.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "run"])
    ap.add_argument("--files", type=int, default=50_000)
    ap.add_argument("--sizes", choices=["log", "uniform"], default="log")
    ap.add_argument("--data", default="data/scenarios")
    ap.add_argument("--results", default="results/scenarios")
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.files, a.sizes, Path(a.data))
    else:
        run(Path(a.data), Path(a.results))


if __name__ == "__main__":
    main()
