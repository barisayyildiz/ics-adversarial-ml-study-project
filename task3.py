"""Task 3: automatic feature extraction with a variational autoencoder (PyTorch).

Input: sets of M consecutive physical readings (all n sensors / actuators) -> one sample is an
M x n window; m such windows form the input. M, the number of features (latent size) and the decoder
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
    classify : classification branch on the latent code (normal / attack), trained with the
               categorical cross-entropy on windows of the labelled attack data (stratified split).
In both cases the usual VAE regulariser beta * KL(q(z|x) || N(0, I)) is added.

Hyper-parameter search (per model): random search over learning rate, hidden size, beta, batch
size and dropout (--trials draws, early stopping on the validation loss); the best trial is kept.

Output (task3_results/<year>_M<M>_z<latent>_<decoder>/):
    features_<layer>_<act>.csv  one line per window: set, window id, first/last row, first/last time,
                                label, f0..f<latent-1> (the latent mean = extracted features)
    hparams_<layer>_<act>.json  every trial of the search and the best one
    model_<layer>_<act>.pt      weights of the best model
    summary.csv                 one line per model

usage: python task3.py --M 10 --latent 8 --decoder recon|classify [--layers dense conv lstm]
                       [--acts relu tanh leakyrelu] [--trials 4] [--epochs 10]
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common

IN = "task1_results"
OUT = "task3_results"
ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "leakyrelu": lambda: nn.LeakyReLU(0.1)}
SPACE = {
    "lr": [1e-3, 3e-4, 1e-4],
    "hidden": [64, 128, 256],
    "beta": [0.1, 0.5, 1.0],
    "batch": [128, 256],
    "dropout": [0.0, 0.1],
}


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
    def __init__(self, layer, act, M, n, latent, hidden, dropout, n_classes=2):
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
        self.mu = nn.Linear(enc_out, latent)
        self.logvar = nn.Linear(enc_out, latent)

        # ---------- decoder 1: reconstruction (mirror of the encoder)
        if layer == "dense":
            self.dec = nn.Sequential(nn.Linear(latent, h[2]), A(), nn.Linear(h[2], h[1]), A(),
                                     nn.Linear(h[1], h[0]), A(), nn.Linear(h[0], M * n))
        elif layer == "conv":
            self.dec_in = nn.Sequential(nn.Linear(latent, h[2] * M), A())
            self.dec = nn.Sequential(nn.Conv1d(h[2], h[1], 3, padding=1), A(),
                                     nn.Conv1d(h[1], h[0], 3, padding=1), A(),
                                     nn.Conv1d(h[0], n, 3, padding=1))
        else:
            self.dec = SeqAct([latent, h[2], h[1], h[0]], A, 0.0)
            self.dec_out = nn.Linear(h[0], n)

        # ---------- decoder 2: classification branch
        self.cls = nn.Sequential(nn.Linear(latent, h[2]), A(), nn.Linear(h[2], n_classes))

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


def kld_recon(x, x_hat, eps=1e-6):
    """Bernoulli KL(x || x_hat) summed over the window, averaged over the batch."""
    x = x.clamp(0, 1)
    x_hat = x_hat.clamp(eps, 1 - eps)
    kl = x * (torch.log(x + eps) - torch.log(x_hat)) + (1 - x) * (torch.log(1 - x + eps) - torch.log(1 - x_hat))
    return kl.flatten(1).sum(1).mean()


def kld_latent(mu, logvar):
    return (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)).mean()


def loss_fn(model, x, y, decoder, beta, class_w):
    out, mu, logvar = model(x, decoder)
    rec = kld_recon(x, out) if decoder == "recon" else F.cross_entropy(out, y, weight=class_w)
    return rec + beta * kld_latent(mu, logvar), rec


# =================================================== data

def windows(x, M, stride):
    starts = np.arange(0, len(x) - M + 1, stride)
    return np.stack([x[s:s + M] for s in starts]), starts


def prepare(a):
    path = f"{IN}/{a.year}/sets.npz"
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing -- run  python task1.py --year {a.year}  first")
    z = np.load(path)
    split = json.load(open(f"{IN}/{a.year}/split.json"))
    d = common.load_year(a.year)
    attack = ((d["attack"] - z["lo"]) / z["rng"]).astype(np.float32)

    sets = {}   # name -> (windows, labels, first rows in the original file, time stamps of the file)
    for name, x, off, ts, y in (("real_A", z["real_A"], split["real_A_rows"][0], d["normal_ts"], None),
                                ("real_B", z["real_B"], split["real_B_rows"][0], d["normal_ts"], None),
                                ("attack", attack, 0, d["attack_ts"], d["attack_y"])):
        w, s = windows(x, a.M, a.stride)
        lab = np.zeros(len(s), np.int64) if y is None else np.array([y[i:i + a.M].max() for i in s], np.int64)
        sets[name] = (w, lab, s + off, ts)
    return sets


def make_split(sets, decoder, seed, max_windows):
    """recon: train on real A, validate on real B (normal only).
    classify: stratified 70 / 15 / 15 split of the labelled attack-data windows."""
    rng = np.random.default_rng(seed)
    if decoder == "recon":
        tr, va = sets["real_A"][0], sets["real_B"][0]
        tr = tr[rng.permutation(len(tr))[:max_windows]]
        va = va[rng.permutation(len(va))[:max_windows // 4]]
        return (tr, np.zeros(len(tr), np.int64)), (va, np.zeros(len(va), np.int64)), None
    w, y = sets["attack"][0], sets["attack"][1]
    parts = [[], [], []]
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == c))
        a, b = int(.7 * len(idx)), int(.85 * len(idx))
        for p, sl in zip(parts, (idx[:a], idx[a:b], idx[b:])):
            p.extend(sl)
    tr, va, te = (np.array(p) for p in parts)
    return (w[tr], y[tr]), (w[va], y[va]), (w[te], y[te])


# =================================================== training / search

def batches(x, y, bs, shuffle, rng=None):
    idx = rng.permutation(len(x)) if shuffle else np.arange(len(x))
    for s in range(0, len(x), bs):
        i = idx[s:s + bs]
        yield torch.from_numpy(x[i]), torch.from_numpy(y[i])


def evaluate(model, data, decoder, beta, class_w, bs=1024):
    model.eval()
    tot, rec, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in batches(*data, bs, False):
            l, r = loss_fn(model, xb, yb, decoder, beta, class_w)
            tot += l.item() * len(xb); rec += r.item() * len(xb); n += len(xb)
    return tot / n, rec / n


def train(layer, act, hp, train_set, val_set, a, n, class_w, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = VAE(layer, act, a.M, n, a.latent, hp["hidden"], hp["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(a.epochs):
        model.train()
        for xb, yb in batches(*train_set, hp["batch"], True, rng):
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
    rng = random.Random(f"{a.seed}-{layer}-{act}")
    all_cfg = [dict(zip(SPACE, v)) for v in np.array(np.meshgrid(*SPACE.values(), indexing="ij"), dtype=object)
               .reshape(len(SPACE), -1).T]
    trials, best = [], None
    for t, hp in enumerate(rng.sample(all_cfg, min(a.trials, len(all_cfg)))):
        hp = {k: (int(v) if k in ("hidden", "batch") else float(v)) for k, v in hp.items()}
        t0 = time.time()
        model, vl, hist = train(layer, act, hp, train_set, val_set, a, n, class_w, a.seed + t)
        trials.append({**hp, "val_loss": vl, "epochs": len(hist), "history": hist, "seconds": round(time.time() - t0, 1)})
        print(f"   trial {t + 1}/{a.trials} {hp}  val loss {vl:.4f}  ({trials[-1]['seconds']} s)")
        if best is None or vl < best[1]:
            best = (model, vl, hp)
    return best, trials


# =================================================== features

def extract(model, w, bs=2048):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(w), bs):
            out.append(model.encode(torch.from_numpy(w[s:s + bs]))[0].numpy())
    return np.concatenate(out)


def write_features(model, sets, M, path):
    with open(path, "w") as f:
        f.write("set,window,first_row,last_row,first_time,last_time,label,"
                + ",".join(f"f{i}" for i in range(model.mu.out_features)) + "\n")
        for name, (w, lab, first, ts) in sets.items():
            feats = extract(model, w)
            for i in range(len(w)):
                r0, r1 = int(first[i]), int(first[i]) + M - 1
                f.write(f"{name},{i},{r0 + 1},{r1 + 1},{ts[r0]},{ts[r1]},{lab[i]},"
                        + ",".join(f"{v:.5g}" for v in feats[i]) + "\n")


def test_metrics(model, test_set, decoder):
    """classify: accuracy / F1 of the attack class on the held-out windows."""
    if decoder != "classify":
        return {}
    model.eval()
    with torch.no_grad():
        pred = np.concatenate([model(xb, decoder)[0].argmax(1).numpy() for xb, _ in batches(*test_set, 2048, False)])
    y = test_set[1]
    tp, fp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"test_accuracy": float((pred == y).mean()), "test_precision": p, "test_recall": r,
            "test_f1": 2 * p * r / (p + r) if p + r else 0.0}


def recon_error_by_label(model, sets):
    """recon: mean reconstruction KLD of normal vs attack windows of the attack data (higher for
    attacks = the features carry information about the anomaly)."""
    w, lab = sets["attack"][0], sets["attack"][1]
    model.eval()
    errs = []
    with torch.no_grad():
        for s in range(0, len(w), 2048):
            xb = torch.from_numpy(w[s:s + 2048])
            out = model(xb, "recon")[0]
            x = xb.clamp(0, 1); o = out.clamp(1e-6, 1 - 1e-6)
            kl = x * (torch.log(x + 1e-6) - torch.log(o)) + (1 - x) * (torch.log(1 - x + 1e-6) - torch.log(1 - o))
            errs.append(kl.flatten(1).sum(1).numpy())
    e = np.concatenate(errs)
    return {"attack_data_recon_kld_normal": float(e[lab == 0].mean()),
            "attack_data_recon_kld_attack": float(e[lab == 1].mean())}


# =================================================== main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=int, default=10, help="physical readings (time steps) per set")
    ap.add_argument("--latent", type=int, default=8, help="number of features after encoding")
    ap.add_argument("--decoder", default="recon", choices=("recon", "classify"))
    ap.add_argument("--layers", nargs="+", default=["dense", "conv", "lstm"], choices=("dense", "conv", "lstm"))
    ap.add_argument("--acts", nargs="+", default=list(ACTS), choices=list(ACTS))
    ap.add_argument("--trials", type=int, default=4, help="configurations tried by the hyper-parameter search")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--stride", type=int, default=None, help="step between windows (default M: no overlap)")
    ap.add_argument("--max-windows", type=int, default=20000, help="cap on training windows (CPU time)")
    ap.add_argument("--year", type=int, default=2017, choices=(2017, 2019))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4, help="torch CPU threads (more is slower for these small models)")
    a = ap.parse_args()
    a.stride = a.stride or a.M
    torch.set_num_threads(a.threads)

    out = f"{OUT}/{a.year}_M{a.M}_z{a.latent}_{a.decoder}"
    os.makedirs(out, exist_ok=True)
    sets = prepare(a)
    train_set, val_set, test_set = make_split(sets, a.decoder, a.seed, a.max_windows)
    n = train_set[0].shape[2]
    class_w = None
    if a.decoder == "classify":     # attacks are rare -> weight the classes by inverse frequency
        cnt = np.bincount(train_set[1], minlength=2)
        class_w = torch.tensor(len(train_set[1]) / (2 * np.maximum(cnt, 1)), dtype=torch.float32)
    print(f"windows M={a.M} x n={n}: train {len(train_set[0])}  val {len(val_set[0])}"
          + (f"  test {len(test_set[0])} (attack share {test_set[1].mean():.3f})" if test_set else ""))

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
                row.update(recon_error_by_label(model, sets))
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
