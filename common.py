"""Shared helpers for sheet 2 -- plain Python version.

Same content as common.py: loading WADI 2017 / 2019, column groups, normalisation and the k-NN
over-sampler of sheet 1 (Task 3).

pandas is only used for reading and writing files (CSV parsing, filling missing values, dates).
Right after reading, the data is converted to plain Python and every calculation is done with
dictionaries, lists and loops. Only exception: the k-NN distances use numpy for the arithmetic
(see manhattan_to_all) and are converted back to Python lists right away.

How the data is stored
----------------------
A "table" is a dictionary  {column name: values}:

    table["1_AIT_001_PV"][t]   = value of sensor 1_AIT_001_PV at second t

The values of a column are an array.array("f", ...) from the standard library. It works exactly
like a list (indexing, slicing, len, for-loops, min, max, sum), but stores each number in 4 bytes
instead of ~32 bytes for a Python float in a list. The 2017 normal file has 1.2 million rows
x 123 columns: about 0.6 GB as arrays instead of ~5 GB as lists.

Dictionaries keep the order in which keys were added, so list(table) gives the column names in
the original file order.

Time stamps are stored as whole seconds since 1970-01-01 (ints), so "one second later" is "+1".

The cleaned data is cached with pickle under data/cache/dict_<year>.pkl, so pandas only reads
the CSV files the first time.
"""
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
    """Number of rows (seconds) of a table: the length of any of its columns."""
    first_column = next(iter(table.values()))
    return len(first_column)


def cut_rows(table, first, end):
    """Rows first .. end-1 of every column (a new, smaller table)."""
    return {col: values[first:end] for col, values in table.items()}


def format_seconds(seconds):
    """seconds since 1970-01-01 -> '2017-09-25 18:00:00'."""
    return str(EPOCH + timedelta(seconds=seconds))


# =================================================== loading (pandas)

def _clean_name(c):
    c = c.strip()
    return c.split("\\")[-1]        # 2017: "\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_AIT_001_PV"


def _read_csv(year, kind):
    """One WADI file as a pandas DataFrame with clean column names."""
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
    """Time stamp of every row as a pandas Series of datetimes."""
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
    """pandas Series with 1 = attack, 0 = normal for every row."""
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
    """Returns (DataFrame of the readings, time stamps in seconds, labels), the last two as
    plain Python lists."""
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
    """Continuous process values; everything else (status, alarms, CO, SP, speed) is treated as
    actuator / control value -- the grouping of sheet1_yat/src/T3.py."""
    c = col.upper()
    return "_PV" in c or "LEAK_DIFF_PRESSURE" in c or "TOTAL_CONS_REQUIRED_FLOW" in c


# =================================================== Task 1a: splitting / normalisation

def split_normal(table, ts, train_days=1.0, subset_days=2.5):
    """One day for training the over-sampler and two disjoint 2.5-day subsets after it.
    Every block is a gap-free piece of the recording (2019 has a hole of ~3 days); if a block
    would cross a gap it is moved to the start of the next piece.
    Returns [(table_block, (first_row, last_row)), ...] for train, subset A, subset B."""
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
    """Min-max scaler fitted on one set and applied to all others (so every set lives in the same
    space; values outside the training range end up below 0 / above 1).

    lo[col]  = minimum of the column in the training set
    rng[col] = maximum - minimum of the column (1 if it is constant, to avoid dividing by 0)"""

    def fit(self, table):
        self.lo, self.rng = {}, {}
        for col, values in table.items():
            lo, hi = min(values), max(values)
            self.lo[col] = lo
            self.rng[col] = hi - lo if hi != lo else 1.0
        return self

    def transform(self, table):
        out = {}
        for col, values in table.items():
            lo, rng = self.lo[col], self.rng[col]
            out[col] = array("f", [(v - lo) / rng for v in values])
        return out

    def inverse(self, table):
        out = {}
        for col, values in table.items():
            lo, rng = self.lo[col], self.rng[col]
            out[col] = array("f", [v * rng + lo for v in values])
        return out


# =================================================== over-sampler of sheet 1, Task 3

def manhattan_to_all(data, i):
    return abs(data - data[i]).sum(axis=1).tolist()


def k_smallest(dist, k, skip):
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
    import numpy as np
    global _DATA
    _DATA = np.array(rows, dtype=np.float64)


def _neighbours_of(args):
    i, k = args
    return k_smallest(manhattan_to_all(_DATA, i), k, skip=i)


def knn_manhattan(rows, k, workers=None):
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


# =================================================== writing (pandas)

def save_csv(table, path, start=datetime(2017, 10, 10)):
    """Same layout as the WADI files / sheet 1 output: Row, Date, Time, then the readings.
    A path ending in .gz is written compressed."""
    df = pd.DataFrame(table)
    t = [start + timedelta(seconds=i) for i in range(len(df))]
    df.insert(0, "Time", [d.strftime("%H:%M:%S") for d in t])
    df.insert(0, "Date", [d.strftime("%d/%m/%Y") for d in t])
    df.insert(0, "Row", list(range(1, len(df) + 1)))
    df.to_csv(path, index=False, float_format="%.6g")
