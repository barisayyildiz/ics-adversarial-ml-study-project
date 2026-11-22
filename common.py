"""Shared WADI helpers (sheets 2 and 3). table = {col: array('f')}, one value per second; timestamps = int seconds since 1970."""
import csv
import multiprocessing
import os
import pickle
import random
from array import array
from datetime import datetime, timedelta

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
CACHE = os.path.join(HERE, "data", "cache")

DAY = 86400                 # WADI samples at 1 Hz -> one day = 86400 rows
EPOCH = datetime(1970, 1, 1)

FILES = {
    2017: {"normal": "WADI.A1_9 Oct 2017/WADI_14days.csv",
           "attack": "WADI.A1_9 Oct 2017/WADI_attackdata.csv"},
    2019: {"normal": "WADI.A2_19 Nov 2019/WADI_14days_new.csv",
           "attack": "WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv"},
}

# WADI 2017 has no label column; the attack windows are the ones from the attack description
# (same list as in sheet 1, sheet1_yat/src/T1.py)
ATTACKS_2017 = [
    ("09/10/2017 19:25:00", "09/10/2017 19:50:16"),
    ("10/10/2017 10:24:10", "10/10/2017 10:34:00"),
    ("10/10/2017 10:55:00", "10/10/2017 11:24:00"),
    ("10/10/2017 11:07:46", "10/10/2017 11:12:15"),
    ("10/10/2017 11:30:40", "10/10/2017 11:44:50"),
    ("10/10/2017 13:39:30", "10/10/2017 13:50:40"),
    ("10/10/2017 14:48:17", "10/10/2017 14:59:55"),
    ("10/10/2017 14:53:44", "10/10/2017 15:00:32"),
    ("10/10/2017 17:40:00", "10/10/2017 17:49:40"),
    ("11/10/2017 10:55:00", "11/10/2017 10:56:27"),
    ("11/10/2017 11:17:54", "11/10/2017 11:31:20"),
    ("11/10/2017 11:36:31", "11/10/2017 11:47:00"),
    ("11/10/2017 11:59:00", "11/10/2017 12:05:00"),
    ("11/10/2017 12:07:30", "11/10/2017 12:10:52"),
    ("11/10/2017 12:16:00", "11/10/2017 12:25:36"),
    ("11/10/2017 15:26:30", "11/10/2017 15:37:00"),
]


# =================================================== small helpers

def n_rows(table):
    """table -> int, number of rows (seconds)."""
    first_column = next(iter(table.values()))
    return len(first_column)


def cut_rows(table, first, end):
    """table, first, end -> new table with rows [first, end)."""
    return {col: values[first:end] for col, values in table.items()}


def format_seconds(seconds):
    """int seconds since 1970 -> 'YYYY-MM-DD HH:MM:SS'."""
    return str(EPOCH + timedelta(seconds=seconds))


# =================================================== loading (pandas)

def _clean_name(c):
    """Raw CSV header -> sensor name (drops the 2017 path prefix)."""
    c = c.strip()
    return c.split("\\")[-1]        # 2017: "\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_AIT_001_PV"


def _read_csv(year, kind):
    """year, 'normal'|'attack' -> pandas DataFrame with clean column names."""
    path = os.path.join(RAW, FILES[year][kind])
    print(f"reading {FILES[year][kind]} ...")
    if year == 2017 and kind == "normal":
        df = pd.read_csv(path, skiprows=4)          # 4 lines of meta data before the header
    elif year == 2019 and kind == "attack":
        df = pd.read_csv(path, skiprows=1)          # first line is just "0,1,2,..."
    else:
        df = pd.read_csv(path)
    df.columns = [_clean_name(c) for c in df.columns]
    df = df.dropna(subset=["Date", "Time"])          # empty trailer lines at the end of some files
    return df.reset_index(drop=True)


def _timestamps(df, year):
    """DataFrame -> pandas Series of datetimes, one per row."""
    date = df["Date"].astype(str).str.strip()
    time = df["Time"].astype(str).str.strip()
    if year == 2019:
        # 2019 stores the time as "mm:ss.f" (hour missing) and the normal file has gaps (removed
        # unstable periods) -> rebuild the time stamp from the 1 Hz row counter
        start = pd.to_datetime(date.iloc[0], format="mixed")
        row = pd.to_numeric(df["Row"], errors="coerce")
        return start + pd.to_timedelta(row - row.iloc[0], unit="s")
    return pd.to_datetime(date + " " + time, format="mixed", errors="coerce")


def _labels(df, ts, year, kind):
    """-> pandas Series of labels per row, 1 = attack, 0 = normal."""
    label_col = [c for c in df.columns if "attack" in c.lower()]
    if label_col:                                     # 2019: 1 = no attack, -1 = attack
        return (df[label_col[0]] == -1).astype(int)
    y = pd.Series(0, index=df.index)
    if year == 2017 and kind == "attack":
        for a, b in ATTACKS_2017:
            a = pd.to_datetime(a, format="%d/%m/%Y %H:%M:%S")
            b = pd.to_datetime(b, format="%d/%m/%Y %H:%M:%S")
            y[(ts >= a) & (ts <= b)] = 1
    return y


def _read_file(year, kind):
    """-> (DataFrame of readings, [int seconds], [0/1 labels])."""
    df = _read_csv(year, kind)
    ts = _timestamps(df, year)
    y = _labels(df, ts, year, kind)

    reading_cols = [c for c in df.columns
                    if c not in ("Row", "Date", "Time") and "attack" not in c.lower()]
    values = df[reading_cols].apply(pd.to_numeric, errors="coerce")   # text -> number, bad -> NaN

    seconds = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds()
    seconds = seconds.fillna(-1).astype(int).tolist()                 # -1 = unreadable time
    return values, seconds, y.tolist()


def load_year(year):
    """Normal and attack data of one year restricted to the columns both files share and that
    are not completely empty. Missing values are filled with the last valid value (at the start
    of a column: the first valid one).

    Returns a dictionary
        cols                 list of column names
        normal, attack       tables {column name: values}
        normal_ts, attack_ts time stamps (seconds since 1970) of every row
        attack_y             label of every row of the attack file, 1 = attack, 0 = normal
    Cached in data/cache/dict_<year>.pkl."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"dict_{year}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)

    print("parsing the CSV files with pandas (only the first time, then cached) ...")
    normal, normal_ts, _ = _read_file(year, "normal")
    attack, attack_ts, attack_y = _read_file(year, "attack")

    # columns in both files that have at least one value in both files
    cols = [c for c in normal.columns
            if c in attack.columns and normal[c].notna().any() and attack[c].notna().any()]

    # fill missing values: previous value, then next value (for gaps at the start), then 0
    normal = normal[cols].ffill().bfill().fillna(0)
    attack = attack[cols].ffill().bfill().fillna(0)

    # pandas -> plain Python: one array of floats per column
    data = {"cols": cols,
            "normal": {c: array("f", normal[c].tolist()) for c in cols},
            "normal_ts": array("q", normal_ts),
            "attack": {c: array("f", attack[c].tolist()) for c in cols},
            "attack_ts": array("q", attack_ts),
            "attack_y": array("b", attack_y)}
    with open(cache, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data


# =================================================== column groups (same rules as sheet 1)

def is_sensor(col):
    """True for continuous sensors (_PV, ...), False for actuators / control values."""
    c = col.upper()
    return "_PV" in c or "LEAK_DIFF_PRESSURE" in c or "TOTAL_CONS_REQUIRED_FLOW" in c


# =================================================== Task 1a: splitting / normalisation

def split_normal(table, ts, train_days=1.0, subset_days=2.5):
    """-> [(table, (first_row, last_row))] for train day, real A, real B (gap-free blocks)."""
    # first row of every new gap-free piece: the time jumps by more (or less) than 1 s
    breaks = []
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] != 1:
            breaks.append(i)

    sizes = [int(train_days * DAY), int(subset_days * DAY), int(subset_days * DAY)]
    blocks = []
    pos = 0
    for n in sizes:
        # move the block behind the last gap that falls inside it, until it fits
        while True:
            inside = [b for b in breaks if pos < b < pos + n]
            if not inside:
                break
            pos = inside[-1]
        if pos + n > n_rows(table):
            raise ValueError(f"not enough gap-free normal data for blocks of {n} rows")
        blocks.append((cut_rows(table, pos, pos + n), (pos, pos + n - 1)))
        pos += n
    return blocks


class MinMax:
    """Min-max scaler: lo[col], rng[col] fitted on one table, applied to all."""

    def fit(self, table):
        """table -> self; stores lo and rng (1 if constant) per column."""
        self.lo, self.rng = {}, {}
        for col, values in table.items():
            lo, hi = min(values), max(values)
            self.lo[col] = lo
            self.rng[col] = hi - lo if hi != lo else 1.0
        return self

    def transform(self, table):
        """table -> table scaled with (v - lo) / rng."""
        out = {}
        for col, values in table.items():
            lo, rng = self.lo[col], self.rng[col]
            out[col] = array("f", [(v - lo) / rng for v in values])
        return out

    def inverse(self, table):
        """scaled table -> table in original units."""
        out = {}
        for col, values in table.items():
            lo, rng = self.lo[col], self.rng[col]
            out[col] = array("f", [v * rng + lo for v in values])
        return out


# =================================================== over-sampler of sheet 1, Task 3

def manhattan_to_all(data, i):
    """numpy rows [n, d], i -> [Manhattan distance from row i to every row]."""
    return abs(data - data[i]).sum(axis=1).tolist()


def k_smallest(dist, k, skip):
    """[distances], k, skip -> indices of the k nearest (ascending), skipping index skip."""
    best_d = [float("inf")] * k
    best_i = [-1] * k
    worst = best_d[-1]                        # distance of the current k-th neighbour
    for m, d in enumerate(dist):
        if d >= worst or m == skip:
            continue
        pos = k - 1                           # walk left to the insert position
        while pos > 0 and best_d[pos - 1] > d:
            best_d[pos] = best_d[pos - 1]
            best_i[pos] = best_i[pos - 1]
            pos -= 1
        best_d[pos] = d
        best_i[pos] = m
        worst = best_d[-1]
    return best_i


_DATA = None                                  # numpy copy of the rows in every worker (set by _init)


def _init(rows):
    """Pool worker init: rows -> global numpy array _DATA."""
    import numpy as np
    global _DATA
    _DATA = np.array(rows, dtype=np.float64)


def _neighbours_of(args):
    """(i, k) -> k nearest neighbour indices of row i."""
    i, k = args
    return k_smallest(manhattan_to_all(_DATA, i), k, skip=i)


def knn_manhattan(rows, k, workers=None):
    """rows [[float]] (n x d), k -> neighbours[i] = [k nearest row indices]; multiprocessing."""
    workers = workers or os.cpu_count()
    n = len(rows)
    result = []
    with multiprocessing.Pool(workers, initializer=_init, initargs=(rows,)) as pool:
        jobs = ((i, k) for i in range(n))
        for i, nearest in enumerate(pool.imap(_neighbours_of, jobs, chunksize=64), 1):
            result.append(nearest)
            if i % 2000 == 0 or i == n:
                print(f"  k-NN {i}/{n}", flush=True)
    return result


def oversample(table, n_out, neighbours, k, order="random", seed=0):
    """table, n_out, neighbours, k -> synthetic table: sensors interpolated, actuators copied from the base row."""
    rnd = random.Random(seed)
    cols = list(table)
    n = n_rows(table)
    sensor = [is_sensor(c) for c in cols]
    rows = [[table[c][t] for c in cols] for t in range(n)]       # row view: rows[t][j]

    out = {c: array("f", [0.0]) * n_out for c in cols}
    for i in range(n_out):
        if order == "random":
            b = rnd.randrange(n)
        else:
            b = i % n
        nb = neighbours[b][rnd.randrange(k)]
        alpha = rnd.random()

        base, neighbour = rows[b], rows[nb]
        for j, c in enumerate(cols):
            if sensor[j]:
                out[c][i] = base[j] + alpha * (neighbour[j] - base[j])
            else:
                out[c][i] = base[j]               # actuators stay as in the base row
    return out


# =================================================== reading / writing result files (pandas)

def load_csv(path):
    """CSV written by save_csv -> table (without Row/Date/Time)."""
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c not in ("Row", "Date", "Time")]
    return {c: array("f", df[c].tolist()) for c in cols}



def save_csv(table, path, start=datetime(2017, 10, 10)):
    """table -> CSV with Row, Date, Time + readings (.gz = compressed)."""
    df = pd.DataFrame(table)
    t = [start + timedelta(seconds=i) for i in range(len(df))]
    df.insert(0, "Time", [d.strftime("%H:%M:%S") for d in t])
    df.insert(0, "Date", [d.strftime("%d/%m/%Y") for d in t])
    df.insert(0, "Row", list(range(1, len(df) + 1)))
    df.to_csv(path, index=False, float_format="%.6g")


# =================================================== sheet 3: samples, k-fold cross-validation, scenarios
# A sample is one window of M readings: (src, first_row, attack type) with src = "normal" | "attack"
# (the WADI file) and attack type -1 = no attack, 0, 1, ... = index of the attack block in the attack file.

def attack_blocks(y):
    """[0/1 labels] -> [(first, last)] of every attack block; block i = attack type i."""
    blocks = []
    start = None
    for t, label in enumerate(y):
        if label == 1 and start is None:
            start = t
        if label != 1 and start is not None:
            blocks.append((start, t - 1))
            start = None
    if start is not None:
        blocks.append((start, len(y) - 1))
    return blocks


def make_folds(d, M, k, n_normal, attack_step=1):
    """-> (normal_folds, attack_folds): normal_folds[f] = [sample], attack_folds[type][f] = [sample].
    Folds are contiguous in time. Every fold of every attack type has the same number of samples."""
    y = d["attack_y"]

    # normal: non-overlapping gap-free windows of the normal file + the attack-free ones of the attack file
    normal = []
    for src in ("normal", "attack"):
        ts = d[src + "_ts"]
        for s in range(0, len(ts) - M + 1, M):
            if ts[s + M - 1] - ts[s] != M - 1:
                continue
            if src == "attack" and max(y[s:s + M]) == 1:
                continue
            normal.append((src, s, -1))
    if n_normal < len(normal):                        # every j-th window, keeps the time order
        step = len(normal) / n_normal
        normal = [normal[int(i * step)] for i in range(n_normal)]
    size = len(normal) // k
    normal_folds = [normal[f * size:(f + 1) * size] for f in range(k)]

    # attacks: every block is cut into k row chunks, sliding windows inside a chunk
    # (so no window has rows in two folds)
    starts = []                                       # starts[type][f] = [first rows]
    for first, last in attack_blocks(y):
        rows = last - first + 1
        per_fold = []
        for f in range(k):
            a = first + f * rows // k
            b = first + (f + 1) * rows // k           # end of the chunk (exclusive)
            per_fold.append(list(range(a, b - M + 1, attack_step)))
        starts.append(per_fold)
    n = min(len(chunk) for per_fold in starts for chunk in per_fold)
    if n == 0:
        raise SystemExit(f"the shortest attack is too short for k={k} folds of windows with M={M}")
    attack_folds = []
    for t, per_fold in enumerate(starts):             # n samples per fold, evenly spread over the chunk
        attack_folds.append([[("attack", chunk[int(i * len(chunk) / n)], t) for i in range(n)]
                             for chunk in per_fold])
    return normal_folds, attack_folds


def scenario_types(scenario, fold, n_types):
    """-> attack types in the training data: 1 none, 2 half of them, 3 exactly one (rotating with the fold)."""
    if scenario == 1:
        return []
    if scenario == 2:
        return [(fold + j) % n_types for j in range(n_types // 2)]
    return [fold % n_types]


def split(normal_folds, attack_folds, scenario, fold):
    """-> (train samples, test samples) of one cross-validation iteration."""
    train = [s for f, part in enumerate(normal_folds) if f != fold for s in part]
    for t in scenario_types(scenario, fold, len(attack_folds)):
        train += [s for f, part in enumerate(attack_folds[t]) if f != fold for s in part]
    test = list(normal_folds[fold])
    for per_fold in attack_folds:                     # all attack types in every test set
        test += per_fold[fold]
    return train, test


def load_folds(year, M, k, n_normal, attack_step):
    """-> (d, normal_folds, attack_folds, windows); windows[sample] = M rows of normalised readings.
    Min-max normalisation over the merged normal + attack file (some columns, e.g. 2_MCV_007_CO,
    are almost constant in the normal file and would reach values of 100+ otherwise)."""
    d = load_year(year)
    lo, rng = [], []
    for c in d["cols"]:
        a = min(min(d["normal"][c]), min(d["attack"][c]))
        b = max(max(d["normal"][c]), max(d["attack"][c]))
        lo.append(a)
        rng.append(b - a if b != a else 1.0)
    normal_folds, attack_folds = make_folds(d, M, k, n_normal, attack_step)
    samples = [s for part in normal_folds for s in part]
    samples += [s for per_fold in attack_folds for part in per_fold for s in part]
    print(f"WADI {year}: {len(samples)} windows of M={M}, k={k} folds, "
          f"{len(normal_folds[0])} normal + {len(attack_folds)} attack types x {len(attack_folds[0][0])} "
          f"attack windows per fold")
    windows = {}
    for src, first, t in samples:
        table = d[src]
        windows[(src, first, t)] = [[(table[c][r] - lo[j]) / rng[j] for j, c in enumerate(d["cols"])]
                                    for r in range(first, first + M)]
    return d, normal_folds, attack_folds, windows


def features_dir(root, year, M, k, scenario, source):
    """-> folder of the feature files of one scenario / feature source (pca_beta0.998, vae_dense_relu, ...)."""
    return os.path.join(root, f"{year}_M{M}_k{k}", f"scenario{scenario}", source)


def write_features(path, d, samples, feats):
    """samples, [feature vector per sample] -> CSV: window, source, first_row, time, label, attack_type, f0, ..."""
    with open(path, "w") as f:
        f.write("window,source,first_row,time,label,attack_type," + ",".join(f"f{i}" for i in range(len(feats[0]))) + "\n")
        for i, ((src, first, t), x) in enumerate(zip(samples, feats)):
            time = format_seconds(d[src + "_ts"][first])
            f.write(f"{i},{src},{first + 1},{time},{int(t >= 0)},{t},"
                    + ",".join(f"{v:.6g}" for v in x) + "\n")


def read_features(path):
    """CSV of write_features -> (info dicts, [feature vector], [0/1 label])."""
    info, x, y = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            info.append({key: row[key] for key in ("window", "source", "first_row", "time", "attack_type")})
            x.append([float(v) for key, v in row.items() if key.startswith("f") and key != "first_row"])
            y.append(int(row["label"]))
    return info, x, y
