# Study Project – Part II: Statistical Detection Methods (WADI)

Continuation of sheet 1. The sheet-1 over-sampler (Manhattan k-NN + SMOTE-like interpolation of the
sensors, actuators copied from the base sample) is ported to `common.py` without library neighbour
functions; the k-NN search runs on all cores (~2 min on 16) and is cached in `data/cache/`.

## Setup

```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt     # numpy, pandas, matplotlib, torch (CPU is enough)
./decompress.sh                     # WADI zips -> data/raw, creates data/cache (--force to overwrite)
```

`decompress.sh` needs `unzip` and the two WADI zips next to it. Parsed CSVs are cached in
`data/cache/dict_<year>.pkl`. Every script takes `--year 2017|2019` (default 2017) and `--help`.

### Helpers

```
cat > ~/.nanorc << 'EOF'            # nano: 4-space indentation, M-m comments a line
set tabsize 4
set tabstospaces
bind M-m comment main
EOF

python -m pydoc -b                  # offline docs in the browser; or e.g. python -m pydoc dict

setxkbmap tr                        # keyboard layout
gsettings set org.gnome.desktop.input-sources sources "[('xkb', 'tr'), ('xkb', 'de')]"
```

## Files

| file | content |
|---|---|
| `common.py` | WADI loading / cleaning, labels, split, min-max scaler, over-sampler |
| `task1.py` | synthetic data (k = 3, 5, 7), MinMax / Gradient / Steadytime / Entropy, plots |
| `task2.py` | CUSUM / EWMA per sensor, alert fusion, evaluation on the attack data |
| `task3.py` | VAE features: 3 layer types × 3 activations, 2 decoders, hyper-parameter search |

All tasks share one split: day 1 of the normal data trains the over-sampler, the next 2 × 2.5 days are
real subsets A and B (2019: never across the ~3-day gap). Everything is min-max normalised with the
scaler of day 1.

## Task 1 – `python task1.py [--k 3 5 7] [--order sequential|random]`

* Per k, 2.5 days of synthetic data → `task1_results/<year>/synthetic_k<k>.csv.gz` (original units).
  `--order sequential` (default) uses the training rows in order, `random` as in sheet 1.
* One model per sensor/actuator is learned on **real A**; the training day, real B and the synthetic
  sets are evaluated against it.
  * MinMax / Gradient / Steadytime: range `[min − (max−min)/4, max + (max−min)/4]`, exceedance =
    distance outside it. Gradient `x[t+1] − x[t]`; Steadytime = run lengths of unchanged values
    (columns with ≤ 50 distinct values).
  * Entropy `H = −Σ p log2 p` over distinct values (rounded to 6 decimals).
* Output: summary in the terminal, `cdf_entropy.png`, `ccdf_minmax.png`.

## Task 2 – `python task2.py --method cusum|ewma [--train real_A|train_day|synthetic_k3|k5|k7]`

Train on `--train`, validate on real B, test on the labelled attack data. `synthetic_k*` needs
`task1.py` first.

* **CUSUM**: `S_{k+1} = max(0, S_k + |r_k| − β)`, `β = mean|r| + b·std|r|`.
* **EWMA**: `S_{k+1} = α·y_{k+1} + (1 − α)·S_k`.
* Alert if `S_k > θ` (then `S = 0`), `θ = 1.1 · max S` on training data; b and α per sensor from a grid
  (fewest false alerts on B).
* **Fusion**: alerts stay active `--hold` s (120), weight `1 / (1 + false alerts on B)`; attack ⇔
  weighted active alerts > τ = highest score on B.
* Output `task2_results/<year>_<method>_<train>/`: `metrics.json` (precision / recall / F1 / FPR,
  detected events, delay), `sensors.csv`, `decisions.csv`.

## Task 3 – `python task3.py --M 10 --features 8 --decoder recon|classify [--epochs 10] [--normal-split sheet|full]`

* Input: non-overlapping windows of M readings × 123 columns.
* Layers `dense`, `conv`, `lstm` × activations `relu`, `tanh`, `leakyrelu` → 9 VAEs; encoder with 3 hidden
  layers, `--features` latent features.
* `recon`: reconstruction with KL divergence as the error, trained on real A, validated on real B. `--normal-split full`
  trains on all normal days except the last 2.5 and validates on those.
* `classify`: normal / attack with categorical cross-entropy on a stratified 70/15/15 split of the
  attack-data windows → `classification_results.csv`.
* Grid search over lr, hidden size and batch size (`SPACE`); lowest validation loss wins.
* Output `task3_results/<year>_M<M>_f<features>_<decoder>/`: `features_<layer>_<act>.csv`,
  `hparams_<layer>_<act>.json`, and for `recon` `reconstruction_<layer>_<act>.csv.gz`.

### Dataset
* [WADI 2017](https://www.b-tu.de/owncloud/s/TLH4b6AXe3JN33n)
* [WADI 2019](https://www.b-tu.de/owncloud/s/XR79Fpb3T92tdSe)

### Results
* [Task 1 Result](https://www.b-tu.de/owncloud/s/N6qccn7WXdTim93)
* [Task 2 Result](https://www.b-tu.de/owncloud/s/sF56yfHGT7EA6qg)
* [Task 3 Result Reconstruction Set 1 (2017)](https://www.b-tu.de/owncloud/s/a8d8Jk52CnnJC8W)
* [Task 3 Result Classification Set 1 (2017)](https://www.b-tu.de/owncloud/s/ac6jtFNidcBmf44)
* [Task 3 Result Reconstruction Set 2 (2019)](https://www.b-tu.de/owncloud/s/5osg4Gk2P56wX72)
* [Task 3 Result Classification Set 2 (2019)](https://www.b-tu.de/owncloud/s/s6jdSxBWeNw5BtQ)
