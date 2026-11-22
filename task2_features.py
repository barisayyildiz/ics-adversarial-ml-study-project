"""Task 2 b): VAE features (sheet 2 VAE) for every k-fold train/test pair; recon decoder in scenario 1, classify in 2 and 3.

usage: python toolbox.py -s s3-features [--year 2017] [--scenario 1 2 3] [--k 8] [--models dense_relu lstm_tanh] [--epochs 10]
"""
import argparse
import os
import random

import torch

import common
import sheet2_task3 as vae

OUT = "results/s3_features"
DECODER = {1: "recon", 2: "classify", 3: "classify"}     # scenario 1 has no attacks to classify
SEED = 0


def to_tensors(windows, samples):
    """samples -> (windows [m, M, n], labels [m]) as tensors."""
    x = torch.tensor([windows[s] for s in samples], dtype=torch.float32)
    y = torch.tensor([int(s[2] >= 0) for s in samples], dtype=torch.long)
    return x, y


def train_vae(model_name, x, y, decoder, a):
    """model_name 'layer_act', training windows -> trained VAE."""
    layer, act = model_name.split("_")
    torch.manual_seed(SEED)
    rnd = random.Random(SEED)
    model = vae.VAE(layer, act, a.M, x.shape[2], a.features, a.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    class_w = None
    if decoder == "classify":         # attacks are rare -> weight the classes by inverse frequency
        labels = y.tolist()
        class_w = torch.tensor([len(labels) / (2 * max(labels.count(c), 1)) for c in (0, 1)])
    for epoch in range(a.epochs):
        model.train()
        total = 0.0
        for xb, yb in vae.batches(x, y, a.batch, rnd):
            loss = vae.loss_fn(model, xb, yb, decoder, class_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        print(f"      epoch {epoch + 1}/{a.epochs}  loss {total / len(x):.4f}")
    return model


def encode(model, x):
    """windows -> [encoder mean mu per window]."""
    model.eval()
    feats = []
    with torch.no_grad():
        for s in range(0, len(x), 2048):
            feats += model.encode(x[s:s + 2048])[0].tolist()
    return feats


def parser():
    """-> command line parser of this task."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2017, choices=(2017, 2019))
    ap.add_argument("--scenario", type=int, nargs="+", default=[1, 2, 3], choices=(1, 2, 3))
    ap.add_argument("--k", type=int, default=8, help="number of folds of the cross-validation")
    ap.add_argument("--M", type=int, default=10, help="readings per window (sample)")
    ap.add_argument("--normal-windows", type=int, default=4000, help="normal windows used (spread over all normal data)")
    ap.add_argument("--attack-step", type=int, default=1, help="step between two attack windows (1 = sliding)")
    ap.add_argument("--models", nargs="+", default=["dense_relu", "lstm_tanh"],
                    help="VAEs as layer_activation, layer: dense|conv|lstm, activation: relu|tanh|leakyrelu")
    ap.add_argument("--features", type=int, default=8, help="number of features after encoding")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=256, help="width of the first hidden layer (best in sheet 2)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    return ap


def main(argv=None):
    """Per scenario, fold and VAE: train on the training windows, write the features of train and test."""
    a = parser().parse_args(argv)
    for m in a.models:
        layer, _, act = m.partition("_")
        if layer not in vae.LAYERS or act not in vae.ACTS:
            raise SystemExit(f"unknown VAE {m}, use layer_activation, e.g. dense_relu, conv_tanh, lstm_leakyrelu")
    d, normal_folds, attack_folds, windows = common.load_folds(a.year, a.M, a.k, a.normal_windows, a.attack_step)

    for scenario in a.scenario:
        decoder = DECODER[scenario]
        for fold in range(a.k):
            train, test = common.split(normal_folds, attack_folds, scenario, fold)
            x_train, y_train = to_tensors(windows, train)
            x_test, _ = to_tensors(windows, test)
            for m in a.models:
                print(f"\n== scenario {scenario}  fold {fold + 1}/{a.k}  VAE {m}  decoder={decoder}  "
                      f"train {len(train)} ({int(y_train.sum())} attack)  test {len(test)}")
                model = train_vae(m, x_train, y_train, decoder, a)
                out = common.features_dir(OUT, a.year, a.M, a.k, scenario, f"vae_{m}")
                os.makedirs(out, exist_ok=True)
                common.write_features(f"{out}/fold{fold + 1}_train.csv", d, train, encode(model, x_train))
                common.write_features(f"{out}/fold{fold + 1}_test.csv", d, test, encode(model, x_test))
    print(f"\nresults in {OUT}/{a.year}_M{a.M}_k{a.k}/")


if __name__ == "__main__":
    main()
