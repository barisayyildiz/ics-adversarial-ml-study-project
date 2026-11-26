"""Task 1: PCA features (covariance, Jacobi eigen decomposition, projection); train = normal file, test = attack file.

usage: python toolbox.py -s s3-pca [--year 2017|2019] [--beta 0.998 0.895 0.879] [--components K]
"""
import argparse
import gzip
import json
import math
import operator
import os
import pickle
from array import array

import common

OUT = "results/s3_pca"


# =================================================== PCA

def covariance(table):
    """table {col: values} -> (mean [n], covariance [n x n]); cov(X, Y) = sum (x - mean x)(y - mean y) / N."""
    cols = list(table)
    N = common.n_rows(table)
    mean = [sum(table[c]) / N for c in cols]
    centred = [array("d", [v - m for v in table[c]]) for c, m in zip(cols, mean)]   # X - mean(X) per column
    n = len(cols)
    cov = [[0.0] * n for _ in range(n)]
    for a in range(n):
        print(f"  covariance row {a + 1}/{n}", end="\r", flush=True)
        for b in range(a, n):                                          # symmetric: only the upper half
            cov[a][b] = cov[b][a] = sum(x * y for x, y in zip(centred[a], centred[b])) / N
    print()
    return mean, cov


def jacobi_eigen(matrix, max_steps=100000):
    """symmetric matrix -> (eigenvalues, eigenvectors), sorted by eigenvalue, largest first.
    Jacobi method: pick the largest off-diagonal value, rotate it to 0, repeat until the matrix is diagonal."""
    n = len(matrix)
    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]   # column j = eigenvector j
    diag = sum(a[i][i] ** 2 for i in range(n))
    for _ in range(max_steps):
        # largest off-diagonal value a[p][q]
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > abs(a[p][q]):
                    p, q = i, j
        if a[p][q] ** 2 <= 1e-24 * diag:                              # (almost) diagonal: done
            break
        # c = cos, s = sin of the rotation angle that makes a[p][q] = 0 (theta is a helper, not the angle)
        theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
        t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1))
        c = 1 / math.sqrt(t * t + 1)
        s = t * c
        for k in range(n):                                            # a = a * J (columns p, q)
            akp, akq = a[k][p], a[k][q]
            a[k][p] = c * akp - s * akq
            a[k][q] = s * akp + c * akq
        for k in range(n):                                            # a = J^T * a (rows p, q)
            apk, aqk = a[p][k], a[q][k]
            a[p][k] = c * apk - s * aqk
            a[q][k] = s * apk + c * aqk
        for k in range(n):                                            # v = v * J
            vkp, vkq = v[k][p], v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq
    order = sorted(range(n), key=lambda i: a[i][i], reverse=True)
    values = [a[i][i] for i in order]
    vectors = [[v[k][i] for k in range(n)] for i in order]           # vectors[j] = j-th eigenvector
    return values, vectors


def n_components(values, beta):
    """eigenvalues (largest first), beta -> smallest K with sum(lambda_1..K) / sum(lambda_1..N) > beta."""
    total = sum(values)
    acc = 0.0
    for K, lam in enumerate(values, 1):
        acc += lam
        if acc / total > beta:
            return K
    return len(values)


def project(table, mean, vectors, K):
    """table -> [[K features] per row]: reading minus the training mean, onto the first K eigenvectors."""
    cols = list(table)
    out = []
    for t in range(common.n_rows(table)):
        x = [table[c][t] - m for c, m in zip(cols, mean)]
        # dot product x · vectors[i]: sum of x[j] * vectors[i][j] (map(mul) is ~2x faster than a generator)
        out.append([sum(map(operator.mul, x, vectors[i])) for i in range(K)])
    return out


def write(path, feats, K, ts, y):
    """first K features per row -> CSV.gz: row, time, label, f0, f1, ..."""
    with gzip.open(path, "wt", compresslevel=1) as f:
        f.write("row,time,label," + ",".join(f"f{i}" for i in range(K)) + "\n")
        for t, x in enumerate(feats):
            f.write(f"{t + 1},{common.format_seconds(ts[t])},{y[t]}," + ",".join(f"{v:.6g}" for v in x[:K]) + "\n")


# =================================================== main

def parser():
    """-> command line parser of this task."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2017, choices=(2017, 2019))
    ap.add_argument("--beta", type=float, nargs="+", default=[0.998, 0.895, 0.879],
                    help="K = smallest dimension with explained variance > beta")
    ap.add_argument("--components", type=int, help="fixed number of principal components (instead of --beta)")
    return ap


def main(argv=None):
    """Covariance + eigenvectors of the normal file, PCA features of the normal (train) and attack (test) file."""
    a = parser().parse_args(argv)
    out = f"{OUT}/{a.year}"
    os.makedirs(out, exist_ok=True)
    d = common.load_year(a.year)
    scaler = common.MinMax().fit(d["normal"])
    train = scaler.transform(d["normal"])
    test = scaler.transform(d["attack"])
    print(f"train (normal file) {common.n_rows(train)} rows, test (attack file) {common.n_rows(test)} rows, "
          f"{len(train)} columns")

    # covariance only depends on the year: computed once, then read from data/cache (delete the file to recompute)
    cache = os.path.join(common.CACHE, f"cov_{a.year}.pkl")
    if os.path.exists(cache):
        print(f"covariance matrix from {cache}")
        with open(cache, "rb") as f:
            mean, cov = pickle.load(f)
    else:
        print("covariance matrix ...")
        mean, cov = covariance(train)
        with open(cache, "wb") as f:
            pickle.dump((mean, cov), f)
    print("eigenvalues / eigenvectors ...")
    values, vectors = jacobi_eigen(cov)

    if a.components:
        choices = {f"K{a.components}": a.components}
    else:
        choices = {f"beta{beta}": n_components(values, beta) for beta in a.beta}
    for name, K in choices.items():
        print(f"{name}: K = {K}  explained variance {sum(values[:K]) / sum(values):.4f}")
    with open(f"{out}/eigenvalues.json", "w") as f:
        json.dump({"eigenvalues": values, "K": choices}, f, indent=1)

    # project once with the largest K, smaller K = the first columns
    K_max = max(choices.values())
    for role, table, ts, y in (("train", train, d["normal_ts"], [0] * common.n_rows(train)),
                               ("test", test, d["attack_ts"], d["attack_y"])):
        print(f"projecting {role} ...")
        feats = project(table, mean, vectors, K_max)
        for name, K in choices.items():
            path = f"{out}/pca_{name}_{role}.csv.gz"
            print(f"writing {path}")
            write(path, feats, K, ts, y)
    print(f"results in {out}/")


if __name__ == "__main__":
    main()
