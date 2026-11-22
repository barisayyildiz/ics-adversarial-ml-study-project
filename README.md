# Study Project – Part III: Feature Generation & Detection Methods (WADI)

Continuation of sheets 1 and 2. All code of the three sheets is started through one toolbox.

## Setup

```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt     # numpy, pandas, matplotlib, torch, scikit-learn, pyarrow
./decompress.sh                     # WADI zips (here or in ..) -> data/raw, data/cache, data/sheet1
```

## Toolbox – `python toolbox.py -h | -s <mode> [parameters]`

| mode | content | file |
|---|---|---|
| `s1-analysis` | sheet 1, tasks 1 + 2 (interactive, no parameters) | `sheet1/T1.py`, `sheet1/T2.py` |
| `s1-oversampling` | sheet 1, task 3 (interactive) | `sheet1/T3.py` |
| `s2-statistics` | sheet 2, task 1 | `sheet2_task1.py` |
| `s2-detection` | sheet 2, task 2 | `sheet2_task2.py` |
| `s2-vae` | sheet 2, task 3 | `sheet2_task3.py` |
| `s3-pca` | sheet 3, task 1 | `task1.py` |
| `s3-features` | sheet 3, task 2 b) | `task2_features.py` |
| `s3-classify` | sheet 3, task 2 c/d) | `task2_classify.py` |

`python toolbox.py -s <mode> -h` shows the parameters of one mode. Results are written to `results/`.

## Samples and k-fold cross-validation (`common.py`)

* Normal and attack file are merged; min-max normalisation over both files.
* A sample is a window of `--M` (10) readings. Normal windows: non-overlapping, gap-free, from the normal
  file and the attack-free part of the attack file; `--normal-windows` (4000) of them, evenly spread.
* Attack type = one attack block of the attack file (14 per year).
* Folds are contiguous in time. Every attack block is cut into k row chunks and the windows of a chunk
  (step `--attack-step`, 1 = sliding) stay inside it, so no window belongs to two folds. Every attack type
  gets the same number of windows per fold (limited by the shortest attack, 88 s).
* Iteration f: test = normal fold f + attack fold f of **all** types; train = the other normal folds +
  the other folds of the training attack types:

| scenario | attack types in training (rotating with f) | classifiers |
|---|---|---|
| 1 | none | `ocsvm`, `elliptic`, `lof` (one-class) |
| 2 | half: types f, f+1, …, f+6 (mod 14) | `svm`, `rf`, `knn` (binary) |
| 3 | exactly one: type f (mod 14) | `svm`, `rf`, `knn` (binary) |

## Task 1 – `python toolbox.py -s s3-pca [--year 2017] [--beta 0.998 0.895 0.879] [--components K]`

Train = normal file, test = attack file, min-max scaler of the normal file. Plain Python: covariance
`Σ(x−x̄)(y−ȳ)/N` of every pair of columns of the normal file, eigenvalues and eigenvectors with the Jacobi
method, K = smallest dimension with `Σ_{i≤K} λ_i / Σ λ_i > β` (or `--components`). Every reading is centred
and projected onto the first K eigenvectors.
Output `results/s3_pca/<year>/`: `pca_beta<β>_{train,test}.csv.gz` (row, time, label, f0, …), `eigenvalues.json`.
Takes ~20 min per year (1.2M rows in plain Python).

## Task 2 b) – `python toolbox.py -s s3-features [--models dense_relu lstm_tanh] [--epochs 10] [--k 8]`

The sheet 2 VAE is trained per scenario and fold on the training windows: reconstruction (KLD) decoder in
scenario 1 (only normal data), classification decoder in scenarios 2 and 3. Features = encoder mean.

Feature files of Task 2 b):
`results/s3_features/<year>_M<M>_k<k>/scenario<s>/<source>/fold<f>_{train,test}.csv`, `<source>` =
`vae_<layer>_<activation>`. Columns: `window, source, first_row, time, label,
attack_type, f0, f1, …`.

## Task 2 c/d) – `python toolbox.py -s s3-classify --scenario 1|2|3 --clf <clf> --features <source>`

Trains on `fold<f>_train.csv`, prints `ATTACK` / `no attack` for every test window, then the metrics per
fold, overall and per attack type. Classifiers that do not fit the scenario are rejected.
Output `results/s3_classify/<year>_M<M>_k<k>/scenario<s>_<source>_<clf>/`: `decisions.csv`, `metrics.json`.
