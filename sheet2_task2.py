"""Task 2: per-sensor CUSUM / EWMA, weighted alert fusion, evaluation on the WADI attack data.

usage: python toolbox.py -s s2-detection --method cusum|ewma [--year 2017] [--train real_A|train_day|synthetic_k5|...]
"""
import argparse
import json
import math
import os

import common

TASK1 = "results/sheet2_task1"
OUT = "results/sheet2_task2"
CUSUM_B = [0.5, 1.0, 2.0, 4.0]                          # beta = mean|r| + b * std|r|
EWMA_ALPHA = [0.001, 0.01, 0.05, 0.1, 0.3, 0.6]
EPS = 1e-6
INF = float("inf")


# =================================================== building blocks

def mean_std(values):
    """[v] -> (mean, population std)."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def abs_changes(values):
    """[x] -> [|r_k|] with r_k = x_k - x_k-1, r_0 = 0."""
    return [0.0] + [abs(values[t] - values[t - 1]) for t in range(1, len(values))]


# =================================================== the two statistics (one sensor)

def cusum(r, beta, theta):
    """[|r_k|] of one sensor -> (alert times, max S); S = max(0, S + |r| - beta), reset to 0 after an alert."""
    alerts = []
    s = 0.0
    s_max = 0.0
    for t, change in enumerate(r):
        s = max(0.0, s + change - beta)   # S_k+1 = (S_k + |r_k| - beta)+
        s_max = max(s_max, s)
        if s > theta:
            alerts.append(t)
            s = 0.0                       # restart
    return alerts, s_max


def ewma(y, alpha, theta):
    """[y_k] of one sensor -> (alert times, max S); S = alpha*y + (1-alpha)*S, reset to 0 after an alert."""
    alerts = []
    s = y[0]                              # S_0 = y_0
    s_max = s
    for t, value in enumerate(y):
        if t > 0:
            s = alpha * value + (1.0 - alpha) * s
        s_max = max(s_max, s)
        if s > theta:
            alerts.append(t)
            s = 0.0                       # restart
    return alerts, s_max


def statistic_input(model, values):
    """-> [|r_k|] for CUSUM, the raw values for EWMA."""
    if model["method"] == "cusum":
        return abs_changes(values)
    return values


def run(model, z, param, theta):
    """Calls cusum / ewma -> (alert times, max S)."""
    if model["method"] == "cusum":
        return cusum(z, param, theta)
    return ewma(z, param, theta)


# =================================================== training: one model per sensor

def learn(method, train, val, margin):
    """-> [model per column]: {col, method, grid, param (beta|alpha), theta, val_fa (false alerts on val)}."""
    models = []
    for col, values in train.items():
        m = {"col": col, "method": method}
        if method == "cusum":
            r_mean, r_std = mean_std(abs_changes(values))
            grid = CUSUM_B                                    # ties -> smallest b
            params = [r_mean + b * r_std + EPS for b in grid]
        else:
            grid = EWMA_ALPHA[::-1]                           # ties -> largest alpha
            params = grid
        z_train = statistic_input(m, values)
        z_val = statistic_input(m, val[col])

        best = None
        for g, p in zip(grid, params):
            _, s_max = run(m, z_train, p, INF)                # no threshold: just the largest S
            theta = (1 + margin) * s_max + EPS
            false_alerts = len(run(m, z_val, p, theta)[0])
            if best is None or false_alerts < best[3]:        # strictly fewer -> keeps the first on ties
                best = (g, p, theta, false_alerts)
        m["grid"], m["param"], m["theta"], m["val_fa"] = best
        models.append(m)
    return models


def detect(models, table):
    """models, table -> {col: [alert times]}."""
    alerts = {}
    for m in models:
        z = statistic_input(m, table[m["col"]])
        alerts[m["col"]] = run(m, z, m["param"], m["theta"])[0]
    return alerts


# =================================================== fusion of the single-sensor alerts

def active_periods(times, hold):
    """alert times, hold -> merged [start, end) periods; [5, 8, 30], 10 -> [(5, 18), (30, 40)]."""
    periods = []
    for t in times:
        if periods and t < periods[-1][1]:
            periods[-1] = (periods[-1][0], t + hold)    # still active -> extend
        else:
            periods.append((t, t + hold))
    return periods


def combined_score(alerts, weights, hold, n):
    """{col: [t]}, {col: weight}, hold, n -> [score per second] = sum of weights of active sensors."""
    cols = list(weights)
    changes = {}                                          # second -> [(sensor index, +1 / -1)]
    for j, col in enumerate(cols):
        for start, end in active_periods(alerts[col], hold):
            changes.setdefault(start, []).append((j, +1))
            if end < n:
                changes.setdefault(end, []).append((j, -1))

    active = [0] * len(cols)
    score = []
    current = 0.0
    for t in range(n):
        if t in changes:
            for j, step in changes[t]:
                active[j] += step
            current = 0.0
            for j, col in enumerate(cols):
                if active[j] > 0:
                    current += weights[col]
        score.append(current)
    return score


# =================================================== evaluation

def events(y):
    """[0/1 labels] -> [(first, last)] of every attack block."""
    ev = []
    start = None
    for t, label in enumerate(y):
        if label == 1 and start is None:
            start = t
        if label != 1 and start is not None:
            ev.append((start, t - 1))
            start = None
    if start is not None:
        ev.append((start, len(y) - 1))
    return ev


def metrics(y, pred):
    """labels, predictions -> dict: tp/fp/fn/tn, precision, recall, f1, fpr, attacks detected, mean delay."""
    tp = fp = fn = tn = 0
    for label, p in zip(y, pred):
        if p and label == 1:
            tp += 1
        elif p:
            fp += 1
        elif label == 1:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    ev = events(y)
    delays = []
    for first, last in ev:
        for t in range(first, last + 1):
            if pred[t]:
                delays.append(t - first)
                break
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "attacks": len(ev), "attacks_detected": len(delays),
            "mean_delay_s": sum(delays) / len(delays) if delays else None}


# =================================================== data

def load_sets(year, train_name):
    """-> (train, val = real B, test = attack data) normalised tables, attack labels [0/1]."""
    d = common.load_year(year)
    (day, _), (sub_a, _), (sub_b, _) = common.split_normal(d["normal"], d["normal_ts"])
    scaler = common.MinMax().fit(day)

    if train_name == "real_A":
        train = sub_a
    elif train_name == "train_day":
        train = day
    elif train_name.startswith("synthetic_k"):
        path = f"{TASK1}/{year}/{train_name}.csv.gz"
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing -- run  python toolbox.py -s s2-statistics --year {year}  first")
        print(f"reading {path} ...")
        train = common.load_csv(path)                      # original units, like the WADI files
    else:
        raise SystemExit(f"unknown training set {train_name}")

    return (scaler.transform(train), scaler.transform(sub_b),
            scaler.transform(d["attack"]), list(d["attack_y"]))


# =================================================== main

def parser():
    """-> command line parser of this task."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=("cusum", "ewma"))
    ap.add_argument("--year", type=int, default=2017, choices=(2017, 2019))
    ap.add_argument("--train", default="real_A",
                    help="training set: real_A, train_day, synthetic_k3, synthetic_k5, synthetic_k7")
    ap.add_argument("--margin", type=float, default=0.1, help="theta = (1 + margin) * max training statistic")
    ap.add_argument("--hold", type=int, default=120, help="seconds an alert stays active for the fusion")
    return ap


def main(argv=None):
    """Train, tau from real B, detect on the attack data -> sensors.csv, decisions.csv, metrics.json."""
    a = parser().parse_args(argv)

    train, val, test, y = load_sets(a.year, a.train)
    n_test = common.n_rows(test)

    out = f"{OUT}/{a.year}_{a.method}_{a.train}"
    os.makedirs(out, exist_ok=True)
    print(f"{a.method.upper()}  train={a.train} ({common.n_rows(train)} rows)  "
          f"validation={common.n_rows(val)}  test(attack)={n_test}  columns={len(train)}")

    print("training + parameter search ...")
    models = learn(a.method, train, val, a.margin)
    weights = {m["col"]: 1.0 / (1.0 + m["val_fa"]) for m in models}

    print("decision threshold on the validation data ...")
    s_val = combined_score(detect(models, val), weights, a.hold, common.n_rows(val))
    tau = max(s_val)

    print("test on the attack data ...")
    alerts = detect(models, test)
    score = combined_score(alerts, weights, a.hold, n_test)
    pred = [s > tau for s in score]
    m = metrics(y, pred)
    m.update({"method": a.method, "train": a.train, "tau": tau, "hold": a.hold})

    # single sensors: chosen parameters and how often they alert inside / outside attacks
    with open(f"{out}/sensors.csv", "w") as f:
        params = "b,beta" if a.method == "cusum" else "alpha"
        f.write(f"col,{params},theta,val_false_alerts,weight,test_alerts_in_attack,test_alerts_normal\n")
        for md in models:
            col = md["col"]
            in_attack = sum(1 for t in alerts[col] if y[t] == 1)
            p = f"{md['grid']},{md['param']:.6g}" if a.method == "cusum" else f"{md['param']}"
            f.write(f"{col},{p},{md['theta']:.6g},{md['val_fa']},{weights[col]:.4g},"
                    f"{in_attack},{len(alerts[col]) - in_attack}\n")
    with open(f"{out}/decisions.csv", "w") as f:
        f.write("row,label,score,attack\n")
        for t in range(n_test):
            f.write(f"{t + 1},{y[t]},{score[t]:.4f},{int(pred[t])}\n")
    with open(f"{out}/metrics.json", "w") as f:
        json.dump(m, f, indent=1)

    print(json.dumps(m, indent=1))
    print(f"results in {out}/")


if __name__ == "__main__":
    main()
