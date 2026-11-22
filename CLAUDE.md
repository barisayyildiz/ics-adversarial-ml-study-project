# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

University study project (Adversarial ML / ICS anomaly detection, BTU Cottbus), practical sheet 3 on the
WADI water-distribution dataset (2017 and 2019). Requirements are in `task_sheet_03.pdf` (deadline
02.12.2026); `README.md` documents the design. `../consultation_notes.txt` has the tutor's answers on
ambiguous points of earlier sheets.

## Commands

```
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
./decompress.sh                                   # WADI zips (here or ..) -> data/raw, data/sheet1
python toolbox.py -h                              # overview + parameters of every mode
python toolbox.py -s s3-pca [--year 2017] [--beta ...] [--components K]          # ~20 min
python toolbox.py -s s3-features [--models dense_relu lstm_tanh] [--epochs 10]
python toolbox.py -s s3-classify --scenario 1 --clf ocsvm --features vae_dense_relu
```

No tests or linter. Verify by running a mode on a small setup, e.g. `--normal-windows 800 --epochs 2`
(PCA has no small setup: it always uses the whole normal file). `s3-classify` needs the
feature files of `s3-features` with the same `--year/--M/--k` (and the same `--normal-windows`
/ `--attack-step`, which are not part of the folder name).

## Architecture

- `toolbox.py` (Task 3): `-s <mode>` imports the mode's module and calls `main(argv)` with the remaining
  arguments; `-h` prints every module's `parser()`. Every python mode must keep the
  `parser()` + `main(argv=None)` shape. Sheet 1 scripts (`sheet1/T*.py`, teammate's code, interactive,
  `input()` prompts) are run with `runpy`; their only change is `base_path` -> `data/sheet1/`.
- `sheet2_task1/2/3.py`: copies of `../sheet2/task*.py`, changed only to `parser()`/`main(argv)` and to write
  to `results/sheet2_task*`. `../sheet2/CLAUDE.md` explains them. `task2_features.py` imports the VAE,
  `loss_fn` and `batches` from `sheet2_task3.py`.
- `common.py`: sheet 2 helpers (loading, `data/cache/dict_<year>.pkl`, `MinMax`, over-sampler) plus the sheet 3
  section: sample = `(src, first_row, attack_type)` window of M readings (`attack_type` -1 = normal,
  else index of the attack block in the attack file, 14 per year). `make_folds` -> contiguous folds,
  each attack block cut into k row chunks with windows inside a chunk, equal windows per type and fold;
  `scenario_types` / `split` -> train/test of one iteration; `load_folds` normalises over the merged
  normal + attack files; `write_features` / `read_features` = the shared feature CSV format.
- `task1.py` (PCA, plain Python) is independent of the folds: covariance/eigenvectors of the normal file,
  projects normal (train) and attack (test) file -> `results/s3_pca/<year>/`.
- `task2_features.py` (VAE per fold) writes
  `results/s3_features/<year>_M<M>_k<k>/scenario<s>/<source>/fold<f>_{train,test}.csv`; `task2_classify.py`
  reads them (sklearn), `VALID` maps scenario -> allowed classifiers (1 one-class, 2/3 binary).

## What sheet 3 asks for

### Rules
- Please follow a similar language with ../sheet2/..
- Try not to excessively use libraries like numpy and pandas, you may use pandas to read a csv file but that should be converted into plain python objects
- Numpy could be use for other third party libraries like keras, but other than that we should use python data-structures
- Keep the code simple as possible, keep the docstrings short, don't overengineer
- Create the virtual environments with venv


**Task 1 – PCA feature extraction (no libraries for any of the maths).** Covariance matrix of all sensor and
actuator columns, using the population formula `Σ(x−x̄)(y−ȳ)/N`, then eigenvalues/eigenvectors (hand-written,
e.g. Jacobi rotation for the symmetric matrix; no `np.cov`, `np.linalg.eig`, `sklearn.PCA`), then projection
onto the top K components. The number of components is a CLI parameter. K can also be chosen as the smallest
K with `Σ_{i≤K} λ_i / Σ_{i≤N} λ_i > β`. Store the PCA features of every train/test pair for β = 0.998, 0.895
and 0.879. Keep these computations plain Python, like the sheet-2 rule for tasks 1 and 2.

**Task 2 – CV scenarios + classifiers.** Merge the WADI normal file with the attack file, then:
- Hand-written k-fold cross-validation (k from the CLI, no library splitters) for three scenarios:
  1. train: normal only; test: normal + all attack types (equal rows per attack type),
  2. train: normal + half of the attack types; test: normal + all attack types,
  3. train: normal + exactly one attack type; test: normal + all attack types.
  Attack types must be balanced in train and in test.
- Generate features with at least two sheet-2 VAE models using the same folds (k = 8 for the submission).
  Pick the decoder per scenario (recon for normal-only training, classify when attacks are in training). The
  output layout must make clear which fold, train/test role and VAE model each file belongs to.
- Classifiers: one-class SVM, EllipticEnvelope ("ecliptic envelope" in the sheet), LOF (one-class), and
  binary SVM, random forest, k-NN. One-class models only fit scenario 1, binary models need attacks in
  training (scenarios 2 and 3). The CLI must reject combinations that make no sense. Print an
  attack / no-attack decision for every test time step.

**Task 3 – toolbox.** One entry point with `-h` (overview + every parameter) and `-s <keyword>` that selects
the mode (sheet-1 analysis, sheet-1 over-sampling, sheet-2 tasks, sheet-3 PCA / classifiers, ...). Keep
every existing CLI parameter of the earlier scripts working through the toolbox.

## Defense constraints (shape how code should be written)

~90 min live-coding defense on lab PCs: Ubuntu 22.04.5, only Geany/Nano/Vim/Neovim plus a terminal,
Anaconda or venv, no admin rights. Any extra package (e.g. `scikit-learn`, new in sheet 3) has to be
announced by email at least 2 working days in advance.
Write code that is easy to change in a terminal editor: flat, readable, parameters on the command line.

From the sheet 1 and 2 defenses (details and the past questions are in `../sheet2/CLAUDE.md`):
- Questions are small code changes or "compute and print X" requests close to the sheet's
  requirements, one per task, each graded out of 4. Points were lost for incomplete printouts: print
  one labelled line per requested item (per set, sensor, attack, fold).
- Likely shapes: turn a constant into a CLI flag, filter/rank/count per column or set, summarise detections
  (counts, shares, per attack type or per fold), small model or parameter change.
- Put new logic where all the data is in scope (usually `main()`), and when adding a new option value
  update every place the existing values are checked (grep for them).
- Keep a fast run path (one fold, one model, few epochs, cached data) to test changes quickly.
