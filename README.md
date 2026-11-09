# Study Project – Part II: Statistical Detection Methods (WADI)

Continuation of sheet 1 (`../sheet1_yat`). The over-sampler of sheet 1 Task 3 (Manhattan k-NN +
SMOTE-like interpolation of the sensor columns, actuators copied from the base sample) is ported to
`common.py` (`knn_manhattan`, `oversample`): same algorithm, callable from other scripts (the
original `T3.py` is an interactive script). As sheet 1 requires, no library distance or
neighbour function is used (no `cdist` / `topk` / `argsort` / `argmin`): the Manhattan distance is
written out with numpy arithmetic (the only numpy use, converted back to Python lists right away) and
the k nearest are picked by a hand-written selection. The brute-force search is spread over all CPU
cores with `multiprocessing` (~2 min on 16 cores for one training day) and its result is cached in `data/cache/knn_<year>_rows<first>-<last>_k<k>.pkl`.

## Setup

```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # numpy, pandas, matplotlib, torch (CPU is enough)
./decompress.sh                          # unpacks both WADI zips into data/raw, creates data/cache
```

`decompress.sh` expects `WaDi.A1_9 Oct 2017.zip` and `WaDi.A2_19 Nov 2019.zip` next to it and needs
`unzip`. Files that already exist are kept, so it's safe to run again; use `./decompress.sh --force`
to overwrite them (e.g. after an interrupted extraction). Resulting layout:

```
data/raw/WADI.A1_9 Oct 2017/    WADI_14days.csv, WADI_attackdata.csv, attack_description.xlsx, table_WADI.pdf
data/raw/WADI.A2_19 Nov 2019/   WADI_14days_new.csv, WADI_attackdataLABLE.csv, table_WADI.pdf
data/cache/                     parsed data, filled on the first run of any task script
```

The CSVs are parsed once and cached in `data/cache/*.npz`.
Every script takes `--year 2017|2019` (default 2017) and `--help`.

## Files

| file | content |
|---|---|
| `decompress.sh` | unpacks the 2017 and 2019 WADI zips into `data/raw` and creates `data/cache` |
| `common.py` | loading / cleaning of both WADI versions, labels, column groups, splitting, min-max scaler, sheet-1 over-sampler |
| `task1.py` | Task 1: synthetic data for k = 3, 5, 7 + MinMax, Gradient, Steadytime, Entropy + plots |
| `task2.py` | Task 2: CUSUM / EWMA per sensor, fusion of the alerts, evaluation on the attack data |
| `task3.py` | Task 3: VAE feature extraction (3 layer types x 3 activations, 2 decoders, hyper-parameter search) |

## Task 1 – `python task1.py [--k 3 5 7] [--order random|sequential]`

* The first day of normal data trains the over-sampler, the next
  2 × 2.5 days are the real subsets A and B (2019: blocks never cross the ~3-day gap in the file).
* All sets are min-max normalised with the scaler of the training day (same space for all).
* For every k, 2.5 days (216 000 rows) of synthetic data → `task1_results/<year>/synthetic_k<k>.csv.gz`
  (original units, same column layout as WADI).
* The statistics learn one model per sensor/actuator from **real subset A**; the training day,
  real subset B (real-vs-real baseline) and the synthetic sets are evaluated against it.
  The sheet's wording "communication pair / protocol" and "packet sizes" comes from the network
  version of the task; for WADI the unit is one sensor/actuator column.
  * MinMax / Gradient / Steadytime: learned range `[min − (max−min)/4, max + (max−min)/4]`,
    exceedance = distance outside the range. The slope is `(x[t+1] − x[t]) / 1 s`, computed by
    hand. Steadytime = lengths of runs of an unchanged value, only for columns with ≤ 50 distinct
    values in A.
  * Entropy: `H = −Σ p log2 p` over the distinct values (normalised, rounded to 6 decimals), with
    hand-written counting.
* Output: `models_real_A.csv`, `stats_<set>.csv` (per column), `summary.csv`, `cdf_entropy.png`,
  `ccdf_minmax.png` (the two plots the sheet asks for).
* `--order random` (default) is the sheet-1 behaviour: random base samples, so the synthetic
  rows have no temporal order. `--order sequential` uses the training rows in order.

## Task 2 – `python task2.py --method cusum|ewma [--train real_A|synthetic_k5|...]`

Plain Python like `task1.py` (pandas only reads the files). It rebuilds the same split and scaler
as `task1.py` itself; only `--train synthetic_k*` needs `task1.py` first (reads
`task1_results/<year>/synthetic_k*.csv.gz`). Training: `--train` (default real subset A;
also `train_day`, `synthetic_k3/5/7`), validation: real subset B (normal), test: the labelled attack
data. The previous numpy version is kept as `task2.bak.py`.

* **CUSUM**: `S_{k+1} = max(0, S_k + |r_k| − β)`, `β = mean|r| + b·std|r|` (b > 0 ⇒ E[|r| − β] < 0).
* **EWMA**: `S_{k+1} = α·y_{k+1} + (1 − α)·S_k` on the measurements `y_k = x_k` (as in the sheet).
* Both: alert if `S_k > θ`, then `S = 0`; `θ = 1.1 · max S` on training data. b and α are chosen per
  sensor from a grid (fewest false alerts on B).
* **Fusion (2c)**: an alert stays active for `--hold` s (120); each sensor is weighted with
  `1 / (1 + its false alerts on B)`; attack ⇔ weighted number of active alerts > τ, with τ
  = the highest score on B (no false alarm on the normal validation data).
* Output `task2_results/<year>_<method>_<train>/`: `metrics.json` (point-wise precision / recall / F1 / FPR,
  detected attack events, delay), `sensors.csv`, `decisions.csv`.

## Task 3 – `python task3.py --M 10 --features 8 --decoder recon|classify`

Plain Python like `task2.py`: it rebuilds the same split and scaler as `task1.py` (no need to run
task1 first). pandas only reads the files and numpy isn't used: windows, labels, splits, batching
and metrics are lists / loops, torch is only used for the networks (`torch.frombuffer` turns the
`array("f")` columns into a tensor).

* Input: non-overlapping windows of M consecutive readings × 123 columns (`--M`); e.g. 216 000 rows,
  M = 10 → m = 21 600 windows.
* Layer types `dense`, `conv` (Conv1d over time), `lstm`; activations `relu`, `tanh`, `leakyrelu`
  → 9 VAE models (`--layers`, `--acts` to restrict). Encoder: 3 hidden layers + μ / log σ²;
  number of features = `--features`.
* Decoder `recon`: reconstructs the window; reconstruction error = Bernoulli KL divergence between the
  readings and the reconstruction. It's trained on real A, validated on real B and tested on the
  attack data.
* `--normal-split full`: instead of real A / real B (first 6 normal days, as in task 1/2), train on all
  normal days except the last 2.5 and validate on those last 2.5 days (closest in time to the attack
  data); the scaler is fitted on that training part. Results go to `..._<decoder>_full/`.
  `--max-windows` still caps the training windows.
* Decoder `classify`: classification branch (normal / attack) trained with categorical cross-entropy on a
  stratified 70/15/15 split of the attack-data windows (real A / B are not used: they were recorded
  weeks earlier and look different, so the classifier would learn "which file" instead of "attack").
* Hyper-parameter search: random search (`--trials`) over lr, hidden size, β, batch size, dropout, with
  early stopping on validation loss.
* Output `task3_results/<year>_M<M>_f<features>_<decoder>/`: `features_<layer>_<act>.csv` (one row per
  window: set, window id, first/last row, first/last time (`YYYY-MM-DD hh:mm:ss`), label, f0…), `hparams_*.json`, `model_*.pt`,
  `summary.csv`; with `--decoder recon` also `reconstruction_<layer>_<act>.csv.gz`: the readings the decoder
  rebuilt, in original units, one row per second of every window (all 123 columns).
