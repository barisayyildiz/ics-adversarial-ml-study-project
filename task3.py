"""Task 3: automatic feature extraction with a variational autoencoder (PyTorch) -- plain Python
version (uses common.py like task2.py; pandas only reads the files, no numpy: the data is
prepared with lists / dictionaries / loops, torch is only used for the neural networks).

Data, chosen with --normal-split (the split is rebuilt here, nothing of task1 has to be run first):
  sheet (default): the same split and scaler as task1.py / task2.py
  * real subset A  : training data of the reconstruction decoder
  * real subset B  : validation data of the reconstruction decoder
  full: all 14 normal days instead of the first 6
  * normal_train   : everything except the last 2.5 days (the scaler is fitted on it)
  * normal_val     : the last 2.5 days (closest in time to the attack data)
  both:
  * attack data    : labelled; test data of the reconstruction decoder,
                     train / validation / test data of the classification decoder
All sets are min-max normalised with the scaler of the training part (sheet: the training day).

Input: sets of M consecutive physical readings (all n sensors / actuators) -> one sample is an
M x n window; m such windows form the input. M, the number of features and the decoder
are read from the command line.

Models: every combination of
    layer type  : dense (fully connected), conv (1-d convolution over time), lstm (recurrent)
    activation  : relu, tanh, leakyrelu
The encoder always has 3 hidden layers (sizes h, h/2, h/4) followed by the mu / log-variance heads.

Two decoders are part of every model, --decoder picks the one that is trained / used:
    recon    : reconstructs the M x n window (sigmoid output). The reconstruction error is the
               Kullback-Leibler divergence between the readings and the reconstruction, each value
               read as a Bernoulli probability:  KL(x || x_hat) = x log(x/x_hat) + (1-x) log((1-x)/(1-x_hat))
               (0 for a perfect reconstruction). Trained on normal data (real subset A, validated on B).
    classify : classification branch on the encoded features (normal / attack), trained with the
               categorical cross-entropy on windows of the labelled attack data (stratified split).
In both cases the usual VAE regulariser beta * KL(q(z|x) || N(0, I)) is added.

Hyper-parameter search (per model): random search over learning rate, hidden size, beta, batch
size and dropout (--trials draws, early stopping on the validation loss); the best trial is kept.

Output (task3_results/<year>_M<M>_f<features>_<decoder>/, with --normal-split full: ..._<decoder>_full/):
    features_<layer>_<act>.csv  one line per window: set, window id, first/last row, first/last time,
                                label, f0..f<features-1> (the mean mu of the encoder = extracted features)
    reconstruction_<layer>_<act>.csv.gz  (recon only) the readings rebuilt by the decoder, in original
                                units: one line per second of every window (set, window id, row, time,
                                label of the window, then all n sensors / actuators)
    hparams_<layer>_<act>.json  every trial of the search and the best one
    model_<layer>_<act>.pt      weights of the best model
    summary.csv                 one line per model

usage: python task3.py --M 10 --features 8 --decoder recon|classify [--layers dense conv lstm]
                       [--acts relu tanh leakyrelu] [--trials 4] [--epochs 10] [--normal-split sheet|full]
"""
import argparse
import gzip
import itertools
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import common

OUT = "task3_results"
ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "leakyrelu": lambda: nn.LeakyReLU(0.1)}
SPACE = {
    "lr": [1e-3, 3e-4, 1e-4],
    "hidden": [64, 128, 256],
    "beta": [0.1, 0.5, 1.0],
    "batch": [128, 256],
    "dropout": [0.0, 0.1],
}
EPS = 1e-6


# =================================================== model

class SeqAct(nn.Module):
    """Stack of 3 LSTM layers with the chosen activation (and dropout) between them."""

    def __init__(self, sizes, act, dropout):
        super().__init__()
        self.rnns = nn.ModuleList(nn.LSTM(a, b, batch_first=True) for a, b in zip(sizes[:-1], sizes[1:]))
        self.act, self.drop = act(), nn.Dropout(dropout)

    def forward(self, x):
        for rnn in self.rnns:
            x = self.drop(self.act(rnn(x)[0]))
        return x


class VAE(nn.Module):
    def __init__(self, layer, act, M, n, n_features, hidden, dropout):
        super().__init__()
        self.layer, self.M, self.n = layer, M, n
        h = [hidden, max(hidden // 2, 4), max(hidden // 4, 4)]
        A = ACTS[act]

        # ---------- encoder: 3 hidden layers
        if layer == "dense":
            self.enc = nn.Sequential(nn.Flatten(),
                                     nn.Linear(M * n, h[0]), A(), nn.Dropout(dropout),
                                     nn.Linear(h[0], h[1]), A(), nn.Dropout(dropout),
                                     nn.Linear(h[1], h[2]), A())
            enc_out = h[2]
        elif layer == "conv":       # channels = sensors, convolution along the time axis
            self.enc = nn.Sequential(nn.Conv1d(n, h[0], 3, padding=1), A(), nn.Dropout(dropout),
                                     nn.Conv1d(h[0], h[1], 3, padding=1), A(), nn.Dropout(dropout),
                                     nn.Conv1d(h[1], h[2], 3, padding=1), A(), nn.Flatten())
            enc_out = h[2] * M
        elif layer == "lstm":
            self.enc = SeqAct([n] + h, A, dropout)
            enc_out = h[2]
        else:
            raise ValueError(layer)
        self.mu = nn.Linear(enc_out, n_features)
        self.logvar = nn.Linear(enc_out, n_features)

        # ---------- decoder 1: reconstruction (mirror of the encoder)
        if layer == "dense":
            self.dec = nn.Sequential(nn.Linear(n_features, h[2]), A(), nn.Linear(h[2], h[1]), A(),
                                     nn.Linear(h[1], h[0]), A(), nn.Linear(h[0], M * n))
        elif layer == "conv":
            self.dec_in = nn.Sequential(nn.Linear(n_features, h[2] * M), A())
            self.dec = nn.Sequential(nn.Conv1d(h[2], h[1], 3, padding=1), A(),
                                     nn.Conv1d(h[1], h[0], 3, padding=1), A(),
                                     nn.Conv1d(h[0], n, 3, padding=1))
        else:
            self.dec = SeqAct([n_features, h[2], h[1], h[0]], A, 0.0)
            self.dec_out = nn.Linear(h[0], n)

        # ---------- decoder 2: classification branch, 2 outputs: score of normal, score of attack
        self.cls = nn.Sequential(nn.Linear(n_features, h[2]), A(), nn.Linear(h[2], 2))

    def encode(self, x):                                   # x [B, M, n]
        e = self.enc(x.transpose(1, 2) if self.layer == "conv" else x)
        if self.layer == "lstm":
            e = e[:, -1]                                   # state after the last reading
        return self.mu(e), self.logvar(e)

    def reconstruct(self, z):
        if self.layer == "dense":
            out = self.dec(z).view(-1, self.M, self.n)
        elif self.layer == "conv":
            out = self.dec(self.dec_in(z).view(len(z), -1, self.M)).transpose(1, 2)
        else:
            out = self.dec_out(self.dec(z[:, None].repeat(1, self.M, 1)))
        return torch.sigmoid(out)

    def forward(self, x, decoder):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        out = self.reconstruct(z) if decoder == "recon" else self.cls(z)
        return out, mu, logvar


def kld_per_window(x, x_hat):
    """Bernoulli KL(x || x_hat) summed over every window -> one value per window."""
    x = x.clamp(0, 1)
    x_hat = x_hat.clamp(EPS, 1 - EPS)
    kl = x * (torch.log(x + EPS) - torch.log(x_hat)) + (1 - x) * (torch.log(1 - x + EPS) - torch.log(1 - x_hat))
    return kl.flatten(1).sum(1)


def kld_latent(mu, logvar):
    return (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)).mean()


def loss_fn(model, x, y, decoder, beta, class_w):
    out, mu, logvar = model(x, decoder)
    rec = kld_per_window(x, out).mean() if decoder == "recon" else F.cross_entropy(out, y, weight=class_w)
    return rec + beta * kld_latent(mu, logvar), rec


# =================================================== data

def to_tensor(table):
    """Table {column: array of floats} -> tensor [rows, columns] (float32, column order kept).
    torch.frombuffer reads the array("f") of a column directly (same 4-byte floats)."""
    return torch.stack([torch.frombuffer(values, dtype=torch.float32) for values in table.values()], dim=1)


def windows(x, M):
    """Cut the rows into m non-overlapping windows x[s : s+M], s = 0, M, 2M, ...
    (216 000 rows, M = 10 -> m = 21 600; leftover rows at the end are dropped).
    Returns (tensor [m, M, n], first row of every window)."""
    starts = list(range(0, len(x) - M + 1, M))
    return torch.stack([x[s:s + M] for s in starts]), starts


def load_sets(year, M, normal_split):
    """Returns (sets, scaler) with sets =
        name -> (windows, labels, first row of every window in the original file, time stamps of the file)
    in the order: normal training set, normal validation set, attack data.
      normal_split "sheet": real_A, real_B -- same split and scaler as task1.py / task2.py
      normal_split "full" : normal_train (all normal rows except the last 2.5 days), normal_val
                            (the last 2.5 days); the scaler is fitted on normal_train
    Normal windows get label 0, attack windows 1 if any reading of the window is an attack.
    Windows that would cross a gap in the time stamps (2019 normal file) are left out.
    The scaler turns the reconstructions back into original units."""
    d = common.load_year(year)
    if normal_split == "sheet":
        (day, _), (sub_a, (first_a, _)), (sub_b, (first_b, _)) = common.split_normal(d["normal"], d["normal_ts"])
        scaler = common.MinMax().fit(day)
        normal_sets = (("real_A", sub_a, first_a), ("real_B", sub_b, first_b))
    else:
        first_val = common.n_rows(d["normal"]) - int(2.5 * common.DAY)
        train = common.cut_rows(d["normal"], 0, first_val)
        scaler = common.MinMax().fit(train)
        normal_sets = (("normal_train", train, 0),
                       ("normal_val", common.cut_rows(d["normal"], first_val, common.n_rows(d["normal"])), first_val))

    sets = {}
    for name, table, offset, ts, y in ([(name, table, offset, d["normal_ts"], None) for name, table, offset in normal_sets]
                                       + [("attack", d["attack"], 0, d["attack_ts"], d["attack_y"])]):
        w, starts = windows(to_tensor(scaler.transform(table)), M)
        # keep only windows of M consecutive seconds (no gap in the recording inside the window)
        keep = [i for i, s in enumerate(starts) if ts[offset + s + M - 1] - ts[offset + s] == M - 1]
        if len(keep) < len(starts):
            print(f"  {name}: {len(starts) - len(keep)} windows cross a gap in the time stamps -> left out")
            w, starts = w[keep], [starts[i] for i in keep]
        if y is None:
            labels = [0] * len(starts)
        else:
            labels = [max(y[s:s + M]) for s in starts]
        sets[name] = (w, labels, [s + offset for s in starts], ts)
    return sets, scaler


def pick(w, y, idx):
    """Windows / labels at the positions idx -> (tensor of windows, tensor of labels)."""
    return w[idx], torch.tensor([y[i] for i in idx], dtype=torch.long)


def make_split(sets, decoder, seed, max_windows):
    """Returns (train, validation, test), each (windows, labels).
    recon:    train = the normal training set (real A / normal_train), validation = the normal
              validation set (real B / normal_val), a random sample of at most max_windows /
              max_windows // 4 windows each; test = all windows of the attack data.
    classify: stratified 70 / 15 / 15 split of the attack-data windows only. The normal sets are
              not used: they were recorded weeks before the attack data and look different, so
              the classifier would learn "which file" instead of "attack or not"."""
    rnd = random.Random(seed)
    w, y = sets["attack"][0], sets["attack"][1]
    if decoder == "recon":
        (w_a, y_a, _, _), (w_b, y_b, _, _), _ = sets.values()     # normal train, normal validation, attack
        tr = rnd.sample(range(len(y_a)), min(max_windows, len(y_a)))
        va = rnd.sample(range(len(y_b)), min(max_windows // 4, len(y_b)))
        return pick(w_a, y_a, tr), pick(w_b, y_b, va), pick(w, y, range(len(y)))

    parts = [[], [], []]
    for c in (0, 1):
        idx = [i for i, label in enumerate(y) if label == c]
        rnd.shuffle(idx)
        end_train = int(0.70 * len(idx))            # position of the 70 % mark
        end_val = int(0.85 * len(idx))              # position of the 85 % mark
        parts[0] += idx[:end_train]                 # first 70 %          -> train
        parts[1] += idx[end_train:end_val]          # from 70 % to 85 %   -> validation
        parts[2] += idx[end_val:]                   # the rest (last 15 %) -> test
    return tuple(pick(w, y, p) for p in parts)


# =================================================== training / search

def batches(x, y, bs, rnd=None):
    """Mini-batches of (windows, labels); shuffled if a random generator is given."""
    idx = list(range(len(x)))
    if rnd is not None:
        rnd.shuffle(idx)
    for s in range(0, len(idx), bs):
        i = idx[s:s + bs]
        yield x[i], y[i]


def evaluate(model, data, decoder, beta, class_w, bs=1024):
    model.eval()
    tot, rec, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in batches(*data, bs):
            l, r = loss_fn(model, xb, yb, decoder, beta, class_w)
            tot += l.item() * len(xb)
            rec += r.item() * len(xb)
            n += len(xb)
    return tot / n, rec / n


def train(layer, act, hp, train_set, val_set, a, n, class_w, seed):
    torch.manual_seed(seed)
    rnd = random.Random(seed)
    model = VAE(layer, act, a.M, n, a.features, hp["hidden"], hp["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(a.epochs):
        model.train()
        for xb, yb in batches(*train_set, hp["batch"], rnd):
            loss, _ = loss_fn(model, xb, yb, a.decoder, hp["beta"], class_w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        vl, vr = evaluate(model, val_set, a.decoder, hp["beta"], class_w)
        hist.append(round(vl, 5))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return model, best, hist


def search(layer, act, train_set, val_set, a, n, class_w):
    """Random search: --trials different configurations from SPACE, keep the lowest validation loss."""
    rnd = random.Random(f"{a.seed}-{layer}-{act}")
    all_cfg = [dict(zip(SPACE, values)) for values in itertools.product(*SPACE.values())]
    trials, best = [], None
    for t, hp in enumerate(rnd.sample(all_cfg, min(a.trials, len(all_cfg)))):
        t0 = time.time()
        model, vl, hist = train(layer, act, hp, train_set, val_set, a, n, class_w, a.seed + t)
        trials.append({**hp, "val_loss": vl, "epochs": len(hist), "history": hist, "seconds": round(time.time() - t0, 1)})
        print(f"   trial {t + 1}/{a.trials} {hp}  val loss {vl:.4f}  ({trials[-1]['seconds']} s)")
        if best is None or vl < best[1]:
            best = (model, vl, hp)
    return best, trials


# =================================================== features / evaluation

def extract(model, w, bs=2048):
    """Latent mean of every window as a list of lists (one list of features per window)."""
    model.eval()
    feats = []
    with torch.no_grad():
        for s in range(0, len(w), bs):
            feats += model.encode(w[s:s + bs])[0].tolist()
    return feats


def write_features(model, sets, M, path):
    with open(path, "w") as f:
        f.write("set,window,first_row,last_row,first_time,last_time,label,"
                + ",".join(f"f{i}" for i in range(model.mu.out_features)) + "\n")
        for name, (w, lab, first, ts) in sets.items():
            feats = extract(model, w)
            for i in range(len(lab)):
                r0, r1 = first[i], first[i] + M - 1
                f.write(f"{name},{i},{r0 + 1},{r1 + 1},{common.format_seconds(ts[r0])},"
                        f"{common.format_seconds(ts[r1])},{lab[i]},"
                        + ",".join(f"{v:.5g}" for v in feats[i]) + "\n")


def write_reconstruction(model, sets, M, scaler, path, bs=2048):
    """recon: the decoder output of every window, turned back into original units
    (value * (max - min) + min of the training day), M lines per window."""
    cols = list(scaler.lo)                                   # same column order as the windows
    lo = torch.tensor([scaler.lo[c] for c in cols])
    rng = torch.tensor([scaler.rng[c] for c in cols])
    model.eval()
    with gzip.open(path, "wt", compresslevel=1) as f, torch.no_grad():
        f.write("set,window,row,time,label," + ",".join(cols) + "\n")
        for name, (w, lab, first, ts) in sets.items():
            for s in range(0, len(w), bs):
                rebuilt = (model(w[s:s + bs], "recon")[0] * rng + lo).tolist()   # [B, M, n]
                for b, window in enumerate(rebuilt):
                    i = s + b
                    for k, reading in enumerate(window):
                        r = first[i] + k
                        f.write(f"{name},{i},{r + 1},{common.format_seconds(ts[r])},{lab[i]},"
                                + ",".join(f"{v:.5g}" for v in reading) + "\n")


def test_metrics(model, test_set, decoder):
    """classify: accuracy / precision / recall / F1 of the attack class on the held-out windows."""
    if decoder != "classify":
        return {}
    model.eval()
    pred = []
    with torch.no_grad():
        for xb, _ in batches(*test_set, 2048):
            pred += model(xb, decoder)[0].argmax(1).tolist()
    y = test_set[1].tolist()

    tp = fp = fn = correct = 0
    for label, p in zip(y, pred):
        correct += label == p
        if p == 1 and label == 1:
            tp += 1
        elif p == 1:
            fp += 1
        elif label == 1:
            fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"test_accuracy": correct / len(y), "test_precision": prec, "test_recall": rec,
            "test_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def recon_error_by_label(model, test_set):
    """recon: mean reconstruction KLD of the normal vs the attack windows of the test set (= the
    attack data; higher for attacks = the features carry information about the anomaly)."""
    w, lab = test_set[0], test_set[1].tolist()
    model.eval()
    errors = []
    with torch.no_grad():
        for s in range(0, len(w), 2048):
            xb = w[s:s + 2048]
            errors += kld_per_window(xb, model(xb, "recon")[0]).tolist()
    normal = [e for e, label in zip(errors, lab) if label == 0]
    attack = [e for e, label in zip(errors, lab) if label == 1]
    return {"attack_data_recon_kld_normal": sum(normal) / len(normal) if normal else None,
            "attack_data_recon_kld_attack": sum(attack) / len(attack) if attack else None}


# =================================================== main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=int, default=10, help="physical readings (time steps) per set")
    ap.add_argument("--features", type=int, default=8, help="number of features after encoding")
    ap.add_argument("--decoder", default="recon", choices=("recon", "classify"))
    ap.add_argument("--layers", nargs="+", default=["dense", "conv", "lstm"], choices=("dense", "conv", "lstm"))
    ap.add_argument("--acts", nargs="+", default=list(ACTS), choices=list(ACTS))
    ap.add_argument("--trials", type=int, default=4, help="configurations tried by the hyper-parameter search")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max-windows", type=int, default=20000, help="cap on training windows (CPU time)")
    ap.add_argument("--year", type=int, default=2017, choices=(2017, 2019))
    ap.add_argument("--normal-split", default="sheet", choices=("sheet", "full"),
                    help="sheet: real A / real B of task1 (first 6 days); "
                         "full: all normal days, the last 2.5 days for validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (more is slower for these small models)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    out = f"{OUT}/{a.year}_M{a.M}_f{a.features}_{a.decoder}" + ("_full" if a.normal_split == "full" else "")
    os.makedirs(out, exist_ok=True)
    print("building the windows ...")
    sets, scaler = load_sets(a.year, a.M, a.normal_split)
    train_set, val_set, test_set = make_split(sets, a.decoder, a.seed, a.max_windows)
    n = train_set[0].shape[2]

    class_w = None
    if a.decoder == "classify":     # attacks are rare -> weight the classes by inverse frequency
        y = train_set[1].tolist()
        counts = [max(y.count(c), 1) for c in (0, 1)]
        class_w = torch.tensor([len(y) / (2 * c) for c in counts], dtype=torch.float32)
    y_test = test_set[1].tolist()
    print(f"windows M={a.M} x n={n}: train {len(train_set[0])}  val {len(val_set[0])}  "
          f"test {len(y_test)} (attack share {sum(y_test) / len(y_test):.3f})")

    summary = []
    for layer in a.layers:
        for act in a.acts:
            print(f"\n== VAE {layer} / {act} / decoder={a.decoder}")
            (model, vl, hp), trials = search(layer, act, train_set, val_set, a, n, class_w)
            tag = f"{layer}_{act}"
            torch.save(model.state_dict(), f"{out}/model_{tag}.pt")
            with open(f"{out}/hparams_{tag}.json", "w") as f:
                json.dump({"best": hp, "best_val_loss": vl, "trials": trials}, f, indent=1)
            write_features(model, sets, a.M, f"{out}/features_{tag}.csv")
            row = {"model": tag, **hp, "val_loss": vl, **test_metrics(model, test_set, a.decoder)}
            if a.decoder == "recon":
                print("   writing the reconstructed readings ...")
                write_reconstruction(model, sets, a.M, scaler, f"{out}/reconstruction_{tag}.csv.gz")
                row.update(recon_error_by_label(model, test_set))
            summary.append(row)
            print("   best:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    keys = list(summary[0])
    with open(f"{out}/summary.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in summary:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"\nresults in {out}/")


if __name__ == "__main__":
    main()
