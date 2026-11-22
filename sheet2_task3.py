"""Task 3: VAE features from windows [M x n] (dense/conv/lstm x relu/tanh/leakyrelu), recon or classify decoder.

usage: python toolbox.py -s s2-vae --M 10 --features 8 --decoder recon|classify [--epochs 10] [--normal-split sheet|full]
"""
import argparse
import gzip
import itertools
import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import common

OUT = "results/sheet2_task3"
LAYERS = ("dense", "conv", "lstm")
ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "leakyrelu": nn.LeakyReLU}
SPACE = {
    "lr": [1e-3, 3e-4, 1e-4],
    "hidden": [64, 128, 256],
    "batch": [128, 256],
}
EPS = 1e-6
SEED = 0


# =================================================== model

class LSTMStack(nn.Module):
    """Stacked LSTMs, sizes = [in, h1, h2, ...], activation after every layer."""

    def __init__(self, sizes, act):
        """sizes: layer widths, act: activation class."""
        super().__init__()
        self.rnns = nn.ModuleList(nn.LSTM(a, b, batch_first=True) for a, b in zip(sizes[:-1], sizes[1:]))
        self.act = act()

    def forward(self, x):
        """x [B, T, in] -> [B, T, sizes[-1]]."""
        for rnn in self.rnns:
            x = self.act(rnn(x)[0])
        return x


class VAE(nn.Module):
    """Encoder (3 hidden layers) -> mu/logvar; decoder recon -> [B, M, n] or classify -> [B, 2]."""
    def __init__(self, layer, act, M, n, n_features, hidden):
        """layer: 'dense'|'conv'|'lstm', act: key of ACTS, hidden layer sizes h, h/2, h/4."""
        super().__init__()
        self.layer, self.M, self.n = layer, M, n
        h = [hidden, hidden // 2, hidden // 4]
        A = ACTS[act]

        # ---------- encoder: 3 hidden layers
        if layer == "dense":
            self.enc = nn.Sequential(nn.Flatten(),
                                     nn.Linear(M * n, h[0]), A(),
                                     nn.Linear(h[0], h[1]), A(),
                                     nn.Linear(h[1], h[2]), A())
            enc_out = h[2]
        elif layer == "conv":       # channels = sensors, convolution along the time axis
            self.enc = nn.Sequential(nn.Conv1d(n, h[0], 3, padding=1), A(),
                                     nn.Conv1d(h[0], h[1], 3, padding=1), A(),
                                     nn.Conv1d(h[1], h[2], 3, padding=1), A(), nn.Flatten())
            enc_out = h[2] * M
        else:
            self.enc = LSTMStack([n] + h, A)
            enc_out = h[2]
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
            self.dec = LSTMStack([n_features, h[2], h[1], h[0]], A)
            self.dec_out = nn.Linear(h[0], n)

        # ---------- decoder 2: classification branch (scores of normal / attack)
        self.cls = nn.Sequential(nn.Linear(n_features, h[2]), A(), nn.Linear(h[2], 2))

    def encode(self, x):                                   # x [B, M, n]
        """x [B, M, n] -> (mu, logvar), each [B, n_features]."""
        e = self.enc(x.transpose(1, 2) if self.layer == "conv" else x)
        if self.layer == "lstm":
            e = e[:, -1]                                   # state after the last reading
        return self.mu(e), self.logvar(e)

    def reconstruct(self, z):
        """z [B, n_features] -> x_hat [B, M, n] in (0, 1)."""
        if self.layer == "dense":
            out = self.dec(z).view(-1, self.M, self.n)
        elif self.layer == "conv":
            out = self.dec(self.dec_in(z).view(len(z), -1, self.M)).transpose(1, 2)
        else:
            out = self.dec_out(self.dec(z[:, None].repeat(1, self.M, 1)))
        return torch.sigmoid(out)

    def forward(self, x, decoder):
        """x, decoder -> (out, mu, logvar); out = x_hat [B, M, n] or logits [B, 2]."""
        mu, logvar = self.encode(x)
        # reparameterization trick while training, the mean otherwise
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        out = self.reconstruct(z) if decoder == "recon" else self.cls(z)
        return out, mu, logvar


def kld_reconstruction(x, x_hat):
    """KL divergence between x and x_hat (values in [0, 1]), summed per window, mean over the batch."""
    x = x.clamp(0, 1)
    x_hat = x_hat.clamp(EPS, 1 - EPS)
    kl = x * (torch.log(x + EPS) - torch.log(x_hat)) + (1 - x) * (torch.log(1 - x + EPS) - torch.log(1 - x_hat))
    return kl.flatten(1).sum(1).mean()


def kld_latent(mu, logvar):
    """KL(N(mu, sigma^2) || N(0, I)), mean over the batch."""
    return (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)).mean()


def loss_fn(model, x, y, decoder, class_w):
    """-> reconstruction KLD (recon) or cross-entropy (classify) + latent KLD."""
    out, mu, logvar = model(x, decoder)
    rec = kld_reconstruction(x, out) if decoder == "recon" else F.cross_entropy(out, y, weight=class_w)
    return rec + kld_latent(mu, logvar)


# =================================================== data

def to_tensor(table):
    """table {col: array} -> tensor [rows, columns]."""
    return torch.stack([torch.frombuffer(values, dtype=torch.float32) for values in table.values()], dim=1)


def load_sets(year, M, normal_split):
    """-> (sets, scaler); sets[name] = (windows [m, M, n], labels, first rows, ts): train, val, 'attack'."""
    d = common.load_year(year)
    if normal_split == "sheet":
        (day, _), (sub_a, (first_a, _)), (sub_b, (first_b, _)) = common.split_normal(d["normal"], d["normal_ts"])
        scaler = common.MinMax().fit(day)
        normal_sets = (("real_A", sub_a, first_a), ("real_B", sub_b, first_b))
    else:
        rows = common.n_rows(d["normal"])
        first_val = rows - int(2.5 * common.DAY)
        train = common.cut_rows(d["normal"], 0, first_val)
        scaler = common.MinMax().fit(train)
        normal_sets = (("normal_train", train, 0),
                       ("normal_val", common.cut_rows(d["normal"], first_val, rows), first_val))

    sets = {}
    for name, table, offset, ts, y in ([(name, table, offset, d["normal_ts"], None) for name, table, offset in normal_sets]
                                       + [("attack", d["attack"], 0, d["attack_ts"], d["attack_y"])]):
        x = to_tensor(scaler.transform(table))
        starts = [s for s in range(0, len(x) - M + 1, M)
                  if ts[offset + s + M - 1] - ts[offset + s] == M - 1]
        w = torch.stack([x[s:s + M] for s in starts])
        labels = [0] * len(starts) if y is None else [max(y[s:s + M]) for s in starts]
        sets[name] = (w, labels, [s + offset for s in starts], ts)
    return sets, scaler


def pick(w, y, idx):
    """windows, labels, idx -> (windows[idx], label tensor)."""
    return w[idx], torch.tensor([y[i] for i in idx], dtype=torch.long)


def make_split(sets, decoder):
    """-> (train, val, test) as (windows, labels); recon: A / B, test None; classify: 70/15/15 of attack windows."""
    if decoder == "recon":
        (w_a, y_a, _, _), (w_b, y_b, _, _), _ = sets.values()     # normal train, normal validation, attack
        return (w_a, torch.tensor(y_a)), (w_b, torch.tensor(y_b)), None

    rnd = random.Random(SEED)
    w, y = sets["attack"][:2]
    tr, va, te = [], [], []
    for c in (0, 1):
        idx = [i for i, label in enumerate(y) if label == c]
        rnd.shuffle(idx)
        end_tr, end_va = int(0.70 * len(idx)), int(0.85 * len(idx))
        tr += idx[:end_tr]
        va += idx[end_tr:end_va]
        te += idx[end_va:]
    return pick(w, y, tr), pick(w, y, va), pick(w, y, te)


# =================================================== training / search

def batches(x, y, bs, rnd=None):
    """Yields (x, y) mini-batches of size bs (shuffled if rnd is given)."""
    idx = list(range(len(x)))
    if rnd is not None:
        rnd.shuffle(idx)
    for s in range(0, len(idx), bs):
        i = idx[s:s + bs]
        yield x[i], y[i]


def evaluate(model, data, decoder, class_w):
    """-> mean loss over data, no gradients."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in batches(*data, 1024):
            total += loss_fn(model, xb, yb, decoder, class_w).item() * len(xb)
            n += len(xb)
    return total / n


def train(layer, act, hp, train_set, val_set, a, n, class_w):
    """-> (model with the best epoch's weights, best validation loss)."""
    torch.manual_seed(SEED)
    rnd = random.Random(SEED)
    model = VAE(layer, act, a.M, n, a.features, hp["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    best, best_state = float("inf"), None
    for _ in range(a.epochs):
        model.train()
        for xb, yb in batches(*train_set, hp["batch"], rnd):
            loss = loss_fn(model, xb, yb, a.decoder, class_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl = evaluate(model, val_set, a.decoder, class_w)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def search(layer, act, train_set, val_set, a, n, class_w):
    """Grid search over SPACE -> ((model, val loss, hp), trials)."""
    all_cfg = [dict(zip(SPACE, values)) for values in itertools.product(*SPACE.values())]
    trials, best = [], None
    for t, hp in enumerate(all_cfg):
        model, vl = train(layer, act, hp, train_set, val_set, a, n, class_w)
        trials.append({**hp, "val_loss": vl})
        print(f"   config {t + 1}/{len(all_cfg)} {hp}  val loss {vl:.4f}")
        if best is None or vl < best[1]:
            best = (model, vl, hp)
    return best, trials


# =================================================== features

def write_features(model, sets, M, path):
    """Encoder mean mu per window -> CSV (set, window, rows, times, label, f0..)."""
    model.eval()
    with open(path, "w") as f, torch.no_grad():
        f.write("set,window,first_row,last_row,first_time,last_time,label,"
                + ",".join(f"f{i}" for i in range(model.mu.out_features)) + "\n")
        for name, (w, lab, first, ts) in sets.items():
            feats = []
            for s in range(0, len(w), 2048):
                feats += model.encode(w[s:s + 2048])[0].tolist()
            for i in range(len(lab)):
                r0, r1 = first[i], first[i] + M - 1
                f.write(f"{name},{i},{r0 + 1},{r1 + 1},{common.format_seconds(ts[r0])},"
                        f"{common.format_seconds(ts[r1])},{lab[i]},"
                        + ",".join(f"{v:.5g}" for v in feats[i]) + "\n")


def write_reconstruction(model, sets, scaler, path):
    """recon: decoder output in original units -> CSV, one line per second of every window."""
    cols = list(scaler.lo)                                   # same column order as the windows
    lo = torch.tensor([scaler.lo[c] for c in cols])
    rng = torch.tensor([scaler.rng[c] for c in cols])
    model.eval()
    with gzip.open(path, "wt", compresslevel=1) as f, torch.no_grad():
        f.write("set,window,row,time,label," + ",".join(cols) + "\n")
        for name, (w, lab, first, ts) in sets.items():
            for s in range(0, len(w), 2048):
                rebuilt = (model(w[s:s + 2048], "recon")[0] * rng + lo).tolist()   # [B, M, n]
                for b, window in enumerate(rebuilt):
                    i = s + b
                    for k, reading in enumerate(window):
                        r = first[i] + k                     # row of this second in the original file
                        f.write(f"{name},{i},{r + 1},{common.format_seconds(ts[r])},{lab[i]},"
                                + ",".join(f"{v:.5g}" for v in reading) + "\n")


def classification_results(model, test_set):
    """classify: -> {accuracy, precision, recall, f1} on the test windows."""
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(xb, "classify")[0].argmax(1) for xb, _ in batches(*test_set, 2048)])
    y = test_set[1]
    tp = ((pred == 1) & (y == 1)).sum().item()     # attack, predicted attack
    fp = ((pred == 1) & (y == 0)).sum().item()     # normal, predicted attack (false alarm)
    fn = ((pred == 0) & (y == 1)).sum().item()     # attack, predicted normal (missed)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (pred == y).float().mean().item(), "precision": precision, "recall": recall, "f1": f1}


# =================================================== main

def parser():
    """-> command line parser of this task."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=int, default=10, help="physical readings per set")
    ap.add_argument("--features", type=int, default=8, help="number of features after encoding")
    ap.add_argument("--decoder", default="classify", choices=("recon", "classify"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--year", type=int, default=2017, choices=(2017, 2019))
    ap.add_argument("--normal-split", default="sheet", choices=("sheet", "full"),
                    help="sheet: real A / real B of task1; full: all normal days, the last 2.5 days for validation")
    return ap


def main(argv=None):
    """Load windows, grid search per layer x activation, write features / reconstructions / results."""
    a = parser().parse_args(argv)

    out = f"{OUT}/{a.year}_M{a.M}_f{a.features}_{a.decoder}" + ("_full" if a.normal_split == "full" else "")
    os.makedirs(out, exist_ok=True)
    print("building the windows ...")
    sets, scaler = load_sets(a.year, a.M, a.normal_split)
    train_set, val_set, test_set = make_split(sets, a.decoder)
    n = train_set[0].shape[2]
    print(f"windows M={a.M} x n={n}: train {len(train_set[0])}  val {len(val_set[0])}"
          + (f"  test {len(test_set[0])}" if test_set else ""))

    class_w = None
    if a.decoder == "classify":     # attacks are rare -> weight the classes by inverse frequency
        y = train_set[1].tolist()
        class_w = torch.tensor([len(y) / (2 * max(y.count(c), 1)) for c in (0, 1)])

    results = []
    for layer in LAYERS:
        for act in ACTS:
            print(f"\n== VAE {layer} / {act} / decoder={a.decoder}")
            (model, vl, hp), trials = search(layer, act, train_set, val_set, a, n, class_w)
            with open(f"{out}/hparams_{layer}_{act}.json", "w") as f:
                json.dump({"best": hp, "best_val_loss": vl, "trials": trials}, f, indent=1)
            write_features(model, sets, a.M, f"{out}/features_{layer}_{act}.csv")
            print(f"   best {hp}  val loss {vl:.4f}")
            if a.decoder == "recon":
                print("   writing the reconstructed readings ...")
                write_reconstruction(model, sets, scaler, f"{out}/reconstruction_{layer}_{act}.csv.gz")
            if a.decoder == "classify":
                r = classification_results(model, test_set)
                print("   test: " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()))
                results.append({"model": f"{layer}_{act}", **r})

    if results:
        with open(f"{out}/classification_results.csv", "w") as f:
            f.write("model,accuracy,precision,recall,f1\n")
            for r in results:
                f.write(f"{r['model']},{r['accuracy']:.4f},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}\n")
    print(f"\nresults in {out}/")


if __name__ == "__main__":
    main()
