"""Task 2: CUSUM and EWMA attack detection on WADI -- plain Python version (uses common.py;
pandas only reads the files, all calculations are plain Python: lists, dictionaries, loops).

Data (all min-max normalised with the scaler of the training day, exactly as in task1.py):
  * training   : --train set (default real subset A; train_day or the synthetic sets of Task 1
                 can be used too -- those are read from task1_results/<year>/)
  * validation : real subset B (normal data only) -- tunes the per-sensor parameters, the weights
                 of the fusion and the final decision threshold
  * test       : the WADI attack data of the same year (labelled)

Per sensor / actuator one detector runs (one "model" per column, like the statistics of Task 1):
  CUSUM  S_0 = 0,  S_k+1 = max(0, S_k + |r_k| - beta),  r_k = x_k - x_k-1
         beta = mean|r| + b * std|r| of the training data (b > 0 => E[|r| - beta] < 0 under H0)
  EWMA   S_0 = y_0,  S_k+1 = alpha * y_k+1 + (1 - alpha) * S_k
         y_k = the (normalised) sensor measurement x_k, as in the task sheet
  Both: alert if S_k > theta, then S is restarted with 0.
        theta = (1 + margin) * max S_k on the training data (no alert on the training data);
        b (CUSUM) and alpha (EWMA) are chosen per sensor from a grid: fewest false alerts on the
        validation data, ties -> the most sensitive value (smallest b / largest alpha).

Fusion of the single-sensor alerts (Task 2c), at every time step t:
  * an alert of a sensor stays "active" for --hold seconds (alerts are single points because of
    the restart)
  * every sensor gets a reliability weight w = 1 / (1 + its false alerts on the validation data)
  * score(t) = sum of the weights of the active sensors;  attack  <=>  score(t) > tau
  * tau = the highest score on the validation data (normal data -> no false alarm there)

An alert list is stored as the list of seconds at which the sensor alerted (alerts are rare, so
this is much smaller than one True/False per second).

Results go to task2_results/<year>_<method>_<train>/.

usage: python task2.py --method cusum|ewma [--year 2017] [--train real_A|train_day|synthetic_k3|...]
"""
import argparse
import json
import math
import os

import common

TASK1 = "task1_results"
OUT = "task2_results"
CUSUM_B = [0.5, 1.0, 2.0, 4.0]                          # beta = mean|r| + b * std|r|
EWMA_ALPHA = [0.001, 0.01, 0.05, 0.1, 0.3, 0.6]
EPS = 1e-6
INF = float("inf")


# =================================================== building blocks

def mean_std(values):
    """Mean and (population) standard deviation, computed by hand."""
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def abs_changes(values):
    """|r_k| with r_k = x_k - x_k-1; r_0 = 0 (there is no previous value)."""
    return [0.0] + [abs(values[t] - values[t - 1]) for t in range(1, len(values))]


# =================================================== the two statistics (one sensor)

def cusum(r, beta, theta):
    """r: the |r_k| of one sensor. Returns (alert times, largest S reached)."""
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
    """y: the measurements of one sensor. Returns (alert times, largest S reached)."""
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
    """The sequence the statistic runs on: |r_k| for CUSUM, the measurements x_k for EWMA."""
    if model["method"] == "cusum":
        return abs_changes(values)
    return values


def run(model, z, param, theta):
    if model["method"] == "cusum":
        return cusum(z, param, theta)
    return ewma(z, param, theta)


# =================================================== training: one model per sensor

def learn(method, train, val, margin):
    """For every column: try every grid value, theta from the training data, keep the value with
    the fewest false alerts on val (normal data). Returns a list of models, one per column."""
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
    """Alert times of every column of a table: {column: [t, ...]}."""
    alerts = {}
    for m in models:
        z = statistic_input(m, table[m["col"]])
        alerts[m["col"]] = run(m, z, m["param"], m["theta"])[0]
    return alerts


# =================================================== fusion of the single-sensor alerts

def active_periods(times, hold):
    """Alert times of one sensor -> periods [start, end) in which it is active. Each alert stays
    active for `hold` seconds; overlapping periods are merged.
    Example (hold = 10): [5, 8, 30] -> [(5, 18), (30, 40)]"""
    periods = []
    for t in times:
        if periods and t < periods[-1][1]:
            periods[-1] = (periods[-1][0], t + hold)    # still active -> extend
        else:
            periods.append((t, t + hold))
    return periods


def combined_score(alerts, weights, hold, n):
    """score[t] = sum of the weights of the sensors active at second t (t = 0 .. n-1).

    Only the seconds where a sensor switches on or off change the score, so those are collected
    first; between them the score stays the same. At every change the score is summed again
    over the active sensors (always in the same order -> exactly the same number for the same
    set of active sensors)."""
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
    """(first, last) row of every block of consecutive attack rows."""
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
    """Point-wise confusion matrix / precision / recall / F1 / FPR, plus per attack: detected at
    all, and after how many seconds (delay)."""
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
    """Same split and scaler as task1.py: (training table, validation table, test table, labels)."""
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
            raise SystemExit(f"{path} missing -- run  python task1.py --year {year}  first")
        print(f"reading {path} ...")
        train = common.load_csv(path)                      # original units, like the WADI files
    else:
        raise SystemExit(f"unknown training set {train_name}")

    return (scaler.transform(train), scaler.transform(sub_b),
            scaler.transform(d["attack"]), list(d["attack_y"]))


# =================================================== main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=("cusum", "ewma"))
    ap.add_argument("--year", type=int, default=2017, choices=(2017, 2019))
    ap.add_argument("--train", default="real_A",
                    help="training set: real_A, train_day, synthetic_k3, synthetic_k5, synthetic_k7")
    ap.add_argument("--margin", type=float, default=0.1, help="theta = (1 + margin) * max training statistic")
    ap.add_argument("--hold", type=int, default=120, help="seconds an alert stays active for the fusion")
    a = ap.parse_args()

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
