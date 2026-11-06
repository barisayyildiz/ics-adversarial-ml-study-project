"""Task 2: CUSUM and EWMA attack detection on WADI.

Data (from task1.py, all min-max normalised with the scaler of the training day):
  * training   : --train set (default real subset A; the synthetic sets of Task 1 can be used too)
  * validation : real subset B (normal data only) -- first half tunes the per-sensor parameters and
                 the weights of the fusion, second half the final decision threshold
  * test       : the WADI attack data of the same year (labelled)

Per sensor / actuator j one detector runs:
  CUSUM  S_0 = 0,  S_k+1 = max(0, S_k + |r_k| - beta_j),  r_k = x_k - x_k-1
         beta_j = mean|r| + b * std|r| of the training data (b > 0 => E[|r| - beta] < 0 under H0)
  EWMA   S_0 = y_0,  S_k+1 = alpha_j * y_k+1 + (1 - alpha_j) * S_k
         y_k = |x_k - mean_j|  (deviation from the training mean: the statistic is ~0 under H0 and
         grows for a shift in either direction, which is what makes the restart S = 0 meaningful)
  Both: alert if S_k > theta_j, then S is restarted with 0.
        theta_j = (1 + margin) * max S_k on the training data (no alert on the training data);
        b (CUSUM) and alpha (EWMA) are chosen per sensor from a grid: fewest false alerts on the
        first validation half, ties -> the most sensitive value (smallest b / largest alpha).

Fusion of the single-sensor alerts (Task 2c), at every time step t:
  * an alert of sensor j stays "active" for --hold seconds (alerts are single points because of the restart)
  * every sensor gets a reliability weight w_j = 1 / (1 + false alerts on the first validation half)
  * score(t) = sum_j w_j * active_j(t);  attack  <=>  score(t) > tau
  * tau = (1 - --fpr) quantile of the score on the second validation half (target false positive rate)

usage: python task2.py --method cusum|ewma [--year 2017] [--train real_A|synthetic_k3|...]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common

IN = "task1_results"
OUT = "task2_results"
CUSUM_B = [0.5, 1.0, 2.0, 4.0]                          # beta = mean|r| + b * std|r|
EWMA_ALPHA = [0.001, 0.01, 0.05, 0.1, 0.3, 0.6]
EPS = 1e-6


# =================================================== the two statistics (vectorised over sensors)

def abs_change(x):
    """|r_k| with r_k = x_k - x_k-1 (r_0 = 0)."""
    r = np.zeros_like(x)
    r[1:] = x[1:] - x[:-1]
    return np.abs(r)


def cusum(x, beta, theta):
    """x [T, n] readings, beta/theta [g, n] (g parameter sets at once).
    Returns the alert matrix [T, g, n] (bool) and the max of S per parameter set / sensor."""
    r = abs_change(x)
    s = np.zeros(beta.shape, dtype=np.float64)
    alerts = np.zeros((len(x),) + beta.shape, dtype=bool)
    smax = np.zeros(beta.shape)
    for k in range(len(x)):
        s = np.maximum(0.0, s + r[k] - beta)
        np.maximum(smax, s, out=smax)
        a = s > theta
        alerts[k] = a
        s[a] = 0.0                                     # restart
    return alerts, smax


def ewma(x, mean, alpha, theta):
    """y_k = |x_k - mean|; alpha/theta [g, n]. Same outputs as cusum()."""
    y = np.abs(x - mean)
    s = np.broadcast_to(y[0], alpha.shape).astype(np.float64)    # S_0 = y_0
    alerts = np.zeros((len(x),) + alpha.shape, dtype=bool)
    smax = s.copy()
    for k in range(len(x)):
        if k:
            s = alpha * y[k] + (1 - alpha) * s
        np.maximum(smax, s, out=smax)
        a = s > theta
        alerts[k] = a
        s[a] = 0.0                                     # restart
    return alerts, smax


class Detector:
    def __init__(self, method, margin):
        self.method, self.margin = method, margin

    def _run(self, x, param, theta):
        if self.method == "cusum":
            return cusum(x, param, theta)
        return ewma(x, self.mean, param, theta)

    def fit(self, train, val):
        """Grid of b / alpha for all sensors at once, theta from the training data, choice on val."""
        n = train.shape[1]
        if self.method == "cusum":
            r = abs_change(train)
            grid = np.array(CUSUM_B)
            param = r.mean(0) + grid[:, None] * r.std(0) + EPS          # beta [g, n]
        else:
            self.mean = train.mean(0)
            grid = np.array(EWMA_ALPHA)
            param = np.repeat(grid[:, None], n, axis=1)                 # alpha [g, n]
        _, smax = self._run(train, param, np.full(param.shape, np.inf))
        theta = (1 + self.margin) * smax + EPS
        alerts, _ = self._run(val, param, theta)
        fa = alerts.sum(0)                                               # false alerts [g, n]
        # fewest false alerts; ties -> most sensitive grid value
        order = np.arange(len(grid)) if self.method == "cusum" else np.arange(len(grid))[::-1]
        best = order[np.argmin(fa[order], axis=0)]
        idx = np.arange(n)
        self.grid_value = grid[best]
        self.param, self.theta = param[best, idx][None], theta[best, idx][None]
        self.val_fa = fa[best, idx]
        return self

    def alerts(self, x):
        return self._run(x, self.param, self.theta)[0][:, 0, :]          # [T, n]


# =================================================== fusion of the single-sensor alerts

def hold(alerts, w):
    """active[t, j] = sensor j alerted in (t - w, t]  (cumulative sum trick)."""
    c = np.cumsum(alerts, axis=0, dtype=np.int32)
    out = c.copy()
    out[w:] -= c[:-w]
    return out > 0


def fused_score(alerts, weights, w):
    return hold(alerts, w).astype(np.float32) @ weights


# =================================================== evaluation

def events(y):
    """(start, end) index pairs of the consecutive attack rows."""
    d = np.diff(np.concatenate(([0], y.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def metrics(y, pred):
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    ev = events(y)
    found, delays = 0, []
    for s, e in ev:
        hit = np.flatnonzero(pred[s:e + 1])
        if len(hit):
            found += 1
            delays.append(int(hit[0]))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "attacks": len(ev), "attacks_detected": found,
            "mean_delay_s": float(np.mean(delays)) if delays else None}


def plot(score, tau, y, title, path):
    fig, ax = plt.subplots(figsize=(11, 4))
    t = np.arange(len(score)) / 3600
    for s, e in events(y):
        ax.axvspan(s / 3600, e / 3600, color="red", alpha=.2, lw=0)
    ax.plot(t, score, lw=.6, label="fused score")
    ax.axhline(tau, color="k", ls="--", lw=1, label=f"tau = {tau:.3g}")
    ax.set(xlabel="hours since start of the attack data", ylabel="weighted active alerts", title=title)
    ax.plot([], [], color="red", alpha=.3, lw=6, label="attack (ground truth)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =================================================== main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=("cusum", "ewma"))
    ap.add_argument("--year", type=int, default=2017, choices=(2017, 2019))
    ap.add_argument("--train", default="real_A",
                    help="training set from task1_results/<year>/sets.npz: real_A, train_day, synthetic_k3, ...")
    ap.add_argument("--margin", type=float, default=0.1, help="theta = (1 + margin) * max training statistic")
    ap.add_argument("--hold", type=int, default=120, help="seconds an alert stays active for the fusion")
    ap.add_argument("--fpr", type=float, default=0.01, help="target false positive rate on validation data")
    a = ap.parse_args()

    path = f"{IN}/{a.year}/sets.npz"
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing -- run  python task1.py --year {a.year}  first")
    z = np.load(path)
    cols = list(z["cols"])
    train, val = z[a.train], z["real_B"]
    d = common.load_year(a.year)
    assert d["cols"] == cols
    test = ((d["attack"] - z["lo"]) / z["rng"]).astype(np.float32)
    y = d["attack_y"]
    val1, val2 = val[:len(val) // 2], val[len(val) // 2:]

    out = f"{OUT}/{a.year}_{a.method}_{a.train}"
    os.makedirs(out, exist_ok=True)
    print(f"{a.method.upper()}  train={a.train} {train.shape}  validation={val.shape}  test(attack)={test.shape}")

    print("training + parameter search ...")
    det = Detector(a.method, a.margin).fit(train, val1)
    weights = (1.0 / (1.0 + det.val_fa)).astype(np.float32)

    print("decision threshold on the second validation half ...")
    s_val = fused_score(det.alerts(val2), weights, a.hold)
    tau = float(np.quantile(s_val, 1 - a.fpr))

    print("test on the attack data ...")
    alerts = det.alerts(test)
    score = fused_score(alerts, weights, a.hold)
    pred = score > tau
    m = metrics(y, pred)
    m.update({"method": a.method, "train": a.train, "tau": tau, "hold": a.hold, "fpr_target": a.fpr,
              "val2_fpr": float((s_val > tau).mean())})

    # single sensors: how often do they alert inside / outside attacks
    pname = "b" if a.method == "cusum" else "alpha"
    with open(f"{out}/sensors.csv", "w") as f:
        f.write(f"col,{pname},{'beta' if a.method == 'cusum' else 'alpha_'},theta,val_false_alerts,weight,"
                "test_alerts_in_attack,test_alerts_normal\n")
        for j, c in enumerate(cols):
            f.write(f"{c},{det.grid_value[j]},{det.param[0, j]:.6g},{det.theta[0, j]:.6g},{det.val_fa[j]},"
                    f"{weights[j]:.4g},{int(alerts[y == 1, j].sum())},{int(alerts[y == 0, j].sum())}\n")
    np.savetxt(f"{out}/decisions.csv", np.c_[np.arange(1, len(y) + 1), y, score, pred],
               delimiter=",", fmt=["%d", "%d", "%.4f", "%d"], header="row,label,score,attack", comments="")
    with open(f"{out}/metrics.json", "w") as f:
        json.dump(m, f, indent=1)
    plot(score, tau, y, f"WADI {a.year} attack data -- {a.method.upper()} trained on {a.train}", f"{out}/score.png")

    print(json.dumps(m, indent=1))
    print(f"results in {out}/")


if __name__ == "__main__":
    main()
