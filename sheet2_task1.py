"""Task 1: synthetic data (k-NN over-sampling) + MinMax / Gradient / Steadytime / Entropy, CDF/CCDF plots.

usage: python toolbox.py -s s2-statistics [--year 2017|2019] [--k 3 5 7] [--order random|sequential]
"""
import argparse
import json
import math
import os
import pickle
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common

OUT = "results/sheet2_task1"
STEADY_MAX_DISTINCT = 50       # Steadytime only for columns with <= 50 distinct training values
STATS = ("minmax", "gradient", "steady")


# =================================================== building blocks

def margins(lo, hi):
    """(lo, hi) -> (lo - w/4, hi + w/4) with w = hi - lo."""
    w = (hi - lo) / 4
    return lo - w, hi + w


def exceedance(value, lo, hi):
    """-> distance of value outside [lo, hi], 0 inside."""
    if value > hi:
        return value - hi
    if value < lo:
        return lo - value
    return 0.0


def slopes(values):
    """[v] -> [v[t+1] - v[t]] (dt = 1 s), length n-1."""
    return [(values[t + 1] - values[t]) / 1.0 for t in range(len(values) - 1)]


def run_lengths(values):
    """[v] -> lengths of runs of equal values, e.g. [5,5,5,2,2,5] -> [3,2,1]."""
    if len(values) == 0:
        return []
    runs = []
    length = 1
    for t in range(1, len(values)):
        if values[t] == values[t - 1]:
            length += 1
        else:
            runs.append(length)
            length = 1
    runs.append(length)
    return runs


def value_counts(values):
    """[v] -> {value: count}."""
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


def shannon_entropy(values, decimals):
    """[v], decimals -> H = -sum p log2 p over the rounded values (bit)."""
    counts = value_counts([round(v, decimals) for v in values])
    total = len(values)
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h


# =================================================== the four statistics

def learn(ref):
    """ref table -> [model per column]: {col, minmax: (lo, hi), gradient: (lo, hi), distinct, steady?: (lo, hi)}."""
    models = []
    for c, values in ref.items():
        g = slopes(values)
        m = {"col": c}
        m["minmax"] = margins(min(values), max(values))
        m["gradient"] = margins(min(g), max(g))
        m["distinct"] = len(value_counts(values))
        if m["distinct"] <= STEADY_MAX_DISTINCT:
            r = run_lengths(values)
            m["steady"] = margins(min(r), max(r))
        models.append(m)
    return models


def evaluate(models, table, decimals):
    """-> (rows: stats dict per column, raw[stat] = {'n': values checked, 'outside': [exceedances > 0]})."""
    rows = []
    raw = {stat: {"n": 0, "outside": []} for stat in STATS}
    for m in models:
        values = table[m["col"]]
        r = {"col": m["col"]}
        checked = {"minmax": values, "gradient": slopes(values)}
        if "steady" in m:
            checked["steady"] = run_lengths(values)

        for stat, v in checked.items():
            lo, hi = m[stat]
            outside = []
            for value in v:
                e = exceedance(value, lo, hi)
                if e > 0:
                    outside.append(e)
            raw[stat]["n"] += len(v)
            raw[stat]["outside"].extend(outside)
            r[f"{stat}_outside"] = len(outside) / len(v)           # fraction outside the range
            r[f"{stat}_max_exceed"] = max(outside) if outside else 0.0
            r[f"{stat}_mean_exceed"] = sum(outside) / len(outside) if outside else 0.0
        r["entropy"] = shannon_entropy(values, decimals)
        rows.append(r)
    return rows, raw


# =================================================== plots

def cdf(ax, values, label):
    """Step CDF of values on ax."""
    v = sorted(values)
    n = len(v)
    ax.step(v, [(i + 1) / n for i in range(n)], where="post", label=label)


def ccdf(ax, r, label):
    """raw[stat] -> log-log CCDF P(X > x) on ax."""
    n_all = r["n"]
    v = sorted(r["outside"])
    if not v:
        return
    p = [(len(v) - i) / n_all for i in range(len(v))]
    ax.step(v, p, where="post", label=f"{label} ({100 * len(v) / n_all:.3g}% outside)")


def plot(entropy, raw, year, ref_name, path):
    """entropy = {set: [H per column]}, raw = {set: raw} -> cdf_entropy.png, ccdf_minmax.png."""
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, h in entropy.items():
        cdf(ax, h, name)
    ax.set(xlabel="Shannon entropy per sensor / actuator [bit]", ylabel="CDF",
           title=f"WADI {year}: entropy of the real and synthetic sets")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{path}/cdf_entropy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, r in raw.items():
        if name.startswith("synthetic"):        # only the generated sets in the CCDF
            ccdf(ax, r["minmax"], name)
    ax.set(xscale="log", yscale="log",
           xlabel="MinMax: distance outside [min_err, max_err] (normalised units)",
           ylabel="CCDF  P(X > x)", title=f"WADI {year}: minmax of the synthetic sets compared to real subset {ref_name}")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{path}/ccdf_minmax.png", dpi=150)
    plt.close(fig)


# =================================================== main

def parser():
    """-> command line parser of this task."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2017, choices=(2017, 2019))
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--order", default="sequential", choices=("random", "sequential"),
                    help="random = base samples drawn at random as in sheet 1; sequential keeps the time order")
    ap.add_argument("--days", type=float, default=2.5, help="days of synthetic data per k (and size of the real subsets)")
    ap.add_argument("--decimals", type=int, default=6, help="rounding of the normalised values for the entropy")
    ap.add_argument("--no-csv", action="store_true", help="do not write the (large) synthetic CSV files")
    return ap


def main(argv=None):
    """Split + normalise, over-sample per k, learn on real A, evaluate every set, write results + plots."""
    a = parser().parse_args(argv)

    out = f"{OUT}/{a.year}"
    os.makedirs(out, exist_ok=True)
    d = common.load_year(a.year)
    cols = d["cols"]
    ts = d["normal_ts"]
    blocks = common.split_normal(d["normal"], ts, subset_days=a.days)
    (train, r_tr), (sub_a, r_a), (sub_b, r_b) = blocks
    for label, (first, last) in (("training day ", r_tr), ("real subset A", r_a), ("real subset B", r_b)):
        print(f"{label} rows {(first, last)}  "
              f"{common.format_seconds(ts[first])} .. {common.format_seconds(ts[last])}")

    # a) normalisation: one scaler (fitted on the training day) for all sets
    scaler = common.MinMax().fit(train)
    sets = {"train day": scaler.transform(train),
            "real A": scaler.transform(sub_a),
            "real B": scaler.transform(sub_b)}

    # over-sampler of sheet 1 trained on the normalised training day; neighbours only once for max(k)
    day = sets["train day"]
    sensor_cols = [c for c in cols if common.is_sensor(c)]
    print(f"k-NN (Manhattan, {len(sensor_cols)} sensor columns) on {common.n_rows(day)} samples ...")
    # the hand-written k-NN takes ~30 min, so its result is cached per year / training day / k
    knn_cache = os.path.join(common.CACHE, f"knn_{a.year}_rows{r_tr[0]}-{r_tr[1]}_k{max(a.k)}.pkl")
    if os.path.exists(knn_cache):
        print(f"k-NN neighbours from cache {knn_cache}")
        with open(knn_cache, "rb") as f:
            nn = pickle.load(f)
    else:
        sensor_rows = [[day[c][t] for c in sensor_cols] for t in range(common.n_rows(day))]
        nn = common.knn_manhattan(sensor_rows, max(a.k))
        del sensor_rows
        os.makedirs(common.CACHE, exist_ok=True)
        with open(knn_cache, "wb") as f:
            pickle.dump(nn, f, protocol=pickle.HIGHEST_PROTOCOL)
    n_out = int(a.days * common.DAY)
    for k in a.k:
        print(f"over-sampling k={k} ...")
        syn = common.oversample(day, n_out, nn, k, order=a.order, seed=k)
        # a) syn is already normalised: it is interpolated from the normalised training day
        sets[f"synthetic k={k}"] = syn
        if not a.no_csv:
            path = f"{out}/synthetic_k{k}.csv.gz"
            print(f"writing {path} (original units)")
            common.save_csv(scaler.inverse(syn), path)

    with open(f"{out}/split.json", "w") as f:
        json.dump({"train_rows": r_tr, "real_A_rows": r_a, "real_B_rows": r_b, "order": a.order,
                   "k": a.k, "days": a.days}, f, indent=1)

    # b) statistics: models learned on real subset A, every set evaluated against them
    ref = "A"
    models = learn(sets["real A"])
    n_steady = sum(1 for m in models if "steady" in m)
    print(f"Steadytime applies to {n_steady}/{len(cols)} columns (<= {STEADY_MAX_DISTINCT} distinct values)")

    entropy, raw, summary = {}, {}, []
    for name, table in sets.items():
        print(f"statistics of {name} ...")
        rows, r = evaluate(models, table, a.decimals)
        entropy[name] = [row["entropy"] for row in rows]
        if name != "real A":                    # the reference cannot exceed its own range
            raw[name] = r

        s = {"set": name, "rows": common.n_rows(table)}
        for stat in STATS:
            n_cols = sum(1 for row in rows if row.get(f"{stat}_outside", 0) > 0)
            outside = r[stat]["outside"]
            s[f"{stat}: share of values outside"] = len(outside) / r[stat]["n"] if r[stat]["n"] else 0.0
            s[f"{stat}: columns with exceedances"] = n_cols
            s[f"{stat}: max exceedance"] = max(outside) if outside else 0.0
        s["entropy: mean [bit]"] = statistics.mean(entropy[name])
        s["entropy: median [bit]"] = statistics.median(entropy[name])
        summary.append(s)

    print()
    for s in summary:
        print(s["set"])
        for k, v in s.items():
            if k != "set":
                print(f"   {k:42s} {v:.4g}" if isinstance(v, float) else f"   {k:42s} {v}")
    plot(entropy, raw, a.year, ref, out)
    print(f"\nplots in {out}/")


if __name__ == "__main__":
    main()
