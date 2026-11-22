"""Task 2 c/d): one-class / binary classifiers on the features of every fold, one decision per test window.

usage: python toolbox.py -s s3-classify --scenario 1|2|3 --clf ocsvm|elliptic|lof|svm|rf|knn
                         --features vae_dense_relu|vae_lstm_tanh|... [--year 2017] [--k 8]
"""
import argparse
import json
import os

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM

import common

FEATURES = "results/s3_features"
OUT = "results/s3_classify"
ONE_CLASS = ("ocsvm", "elliptic", "lof")       # learn only the normal data -> scenario 1
BINARY = ("svm", "rf", "knn")                  # need attacks in the training data -> scenarios 2 and 3
VALID = {1: ONE_CLASS, 2: BINARY, 3: BINARY}


def make(clf, a):
    """classifier name -> sklearn model (features standardised on the training data first)."""
    if clf == "ocsvm":
        model = OneClassSVM(nu=a.contamination, gamma="scale")
    elif clf == "elliptic":
        model = EllipticEnvelope(contamination=a.contamination, random_state=0)
    elif clf == "lof":
        model = LocalOutlierFactor(n_neighbors=a.neighbours, novelty=True, contamination=a.contamination)
    elif clf == "svm":
        model = SVC(class_weight="balanced")
    elif clf == "rf":
        model = RandomForestClassifier(n_estimators=a.trees, class_weight="balanced", random_state=0, n_jobs=-1)
    else:
        model = KNeighborsClassifier(n_neighbors=a.neighbours)
    return make_pipeline(StandardScaler(), model)


def predict(model, clf, x):
    """-> [1 = attack, 0 = normal]; one-class models return -1 for outliers."""
    p = model.predict(np.array(x)).tolist()
    if clf in ONE_CLASS:
        return [1 if v == -1 else 0 for v in p]
    return [int(v) for v in p]


def counts(y, pred):
    """labels, predictions -> {tp, fp, fn, tn, precision, recall, f1, fpr}."""
    tp = sum(1 for label, p in zip(y, pred) if p == 1 and label == 1)
    fp = sum(1 for label, p in zip(y, pred) if p == 1 and label == 0)
    fn = sum(1 for label, p in zip(y, pred) if p == 0 and label == 1)
    tn = len(y) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0}


def line(m):
    """metrics dict -> one printable line."""
    return (f"precision {m['precision']:.3f}  recall {m['recall']:.3f}  f1 {m['f1']:.3f}  fpr {m['fpr']:.4f}  "
            f"(tp {m['tp']} fp {m['fp']} fn {m['fn']} tn {m['tn']})")


def parser():
    """-> command line parser of this task."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", type=int, required=True, choices=(1, 2, 3),
                    help="1: no attacks in training, 2: half of the attack types, 3: one attack type")
    ap.add_argument("--clf", required=True, choices=ONE_CLASS + BINARY,
                    help="one-class (scenario 1): ocsvm, elliptic, lof; binary (scenarios 2, 3): svm, rf, knn")
    ap.add_argument("--features", required=True,
                    help="feature folder written by s3-features, e.g. vae_dense_relu")
    ap.add_argument("--year", type=int, default=2017, choices=(2017, 2019))
    ap.add_argument("--k", type=int, default=8, help="number of folds (as used for the features)")
    ap.add_argument("--M", type=int, default=10, help="readings per window (as used for the features)")
    ap.add_argument("--contamination", type=float, default=0.01, help="one-class: expected share of outliers (nu)")
    ap.add_argument("--neighbours", type=int, default=5, help="k of k-NN and LOF")
    ap.add_argument("--trees", type=int, default=100, help="trees of the random forest")
    return ap


def main(argv=None):
    """Train + test per fold, print every test decision, write decisions.csv and metrics.json."""
    a = parser().parse_args(argv)
    if a.clf not in VALID[a.scenario]:
        raise SystemExit(f"{a.clf} does not fit scenario {a.scenario}: "
                         f"scenario 1 -> {', '.join(ONE_CLASS)}; scenarios 2, 3 -> {', '.join(BINARY)}")
    src = common.features_dir(FEATURES, a.year, a.M, a.k, a.scenario, a.features)
    if not os.path.exists(f"{src}/fold1_train.csv"):
        raise SystemExit(f"{src} missing -- run  python toolbox.py -s s3-features  first")

    out = f"{OUT}/{a.year}_M{a.M}_k{a.k}/scenario{a.scenario}_{a.features}_{a.clf}"
    os.makedirs(out, exist_ok=True)
    all_y, all_pred, all_type, per_fold = [], [], [], []
    with open(f"{out}/decisions.csv", "w") as f:
        f.write("fold,window,source,first_row,time,label,attack_type,prediction\n")
        for fold in range(1, a.k + 1):
            _, x_train, y_train = common.read_features(f"{src}/fold{fold}_train.csv")
            info, x_test, y_test = common.read_features(f"{src}/fold{fold}_test.csv")
            model = make(a.clf, a)
            if a.clf in ONE_CLASS:
                model.fit(np.array(x_train))
            else:
                model.fit(np.array(x_train), np.array(y_train))
            pred = predict(model, a.clf, x_test)

            print(f"\n== fold {fold}/{a.k}: train {len(y_train)} ({sum(y_train)} attack)  test {len(y_test)}")
            for i, p in zip(info, pred):
                print(f"fold {fold}  window {i['window']:>5}  {i['time']}  ({i['source']} row {i['first_row']})  "
                      f"{'ATTACK' if p else 'no attack'}")
                f.write(f"{fold},{i['window']},{i['source']},{i['first_row']},{i['time']},"
                        f"{int(i['attack_type']) >= 0:d},{i['attack_type']},{p}\n")
            m = counts(y_test, pred)
            per_fold.append({"fold": fold, **m})
            print(f"fold {fold}: {line(m)}")
            all_y += y_test
            all_pred += pred
            all_type += [int(i["attack_type"]) for i in info]

    total = counts(all_y, all_pred)
    detected = {}                                   # attack type -> [windows detected, windows]
    for t, p in zip(all_type, all_pred):
        if t >= 0:
            detected.setdefault(t, [0, 0])
            detected[t][0] += p
            detected[t][1] += 1
    print(f"\n== {a.clf} on {a.features}, scenario {a.scenario}, {a.k} folds")
    for m in per_fold:
        print(f"fold {m['fold']}: {line(m)}")
    print(f"all  : {line(total)}")
    for t in sorted(detected):
        print(f"attack type {t:2d}: {detected[t][0]}/{detected[t][1]} windows detected")
    with open(f"{out}/metrics.json", "w") as f:
        json.dump({"scenario": a.scenario, "clf": a.clf, "features": a.features, "all": total,
                   "folds": per_fold, "attack_types": detected}, f, indent=1)
    print(f"results in {out}/")


if __name__ == "__main__":
    main()
