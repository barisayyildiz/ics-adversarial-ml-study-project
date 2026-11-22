"""Toolbox of the study project "Generation of Datasets for Anomaly Detection in ICS" (WADI), sheets 1 to 3.

usage: python toolbox.py -h
       python toolbox.py -s <mode> [parameters of the mode]
       python toolbox.py -s <mode> -h          (parameters of one mode)
"""
import importlib
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# mode -> (description, python module with parser() / main(argv), or a list of sheet 1 scripts)
MODES = {
    "s1-analysis": ("sheet 1, tasks 1 + 2: statistical analysis of the WADI readings (K-S statistic, system states)",
                    ["sheet1/T1.py", "sheet1/T2.py"]),
    "s1-oversampling": ("sheet 1, task 3: synthetic data with k-NN over-sampling (asks for its inputs)",
                        ["sheet1/T3.py"]),
    "s2-statistics": ("sheet 2, task 1: synthetic sets + MinMax / Gradient / Steadytime / Entropy", "sheet2_task1"),
    "s2-detection": ("sheet 2, task 2: CUSUM / EWMA attack detection", "sheet2_task2"),
    "s2-vae": ("sheet 2, task 3: VAE features (dense/conv/lstm x relu/tanh/leakyrelu)", "sheet2_task3"),
    "s3-pca": ("sheet 3, task 1: PCA features (train = normal file, test = attack file)", "task1"),
    "s3-features": ("sheet 3, task 2 b): VAE features for every k-fold train/test pair", "task2_features"),
    "s3-classify": ("sheet 3, task 2 c/d): one-class / binary classifiers per scenario", "task2_classify"),
}


def usage():
    """Overview + the parameters of every mode."""
    print(__doc__)
    print("modes (-s <mode>):")
    for mode, (text, _) in MODES.items():
        print(f"  {mode:16s} {text}")
    print("\nworkflow of sheet 3: s3-pca (task 1); s3-features (VAE features of the folds), then s3-classify (task 2)")
    for mode, (text, target) in MODES.items():
        print("\n" + "=" * 100 + f"\n-s {mode}: {text}\n")
        if isinstance(target, list):
            print("  no command line parameters: interactive scripts of sheet 1 " + ", ".join(target)
                  + "\n  data: data/sheet1/raw/2017/WADI_14days.csv, WADI_attackdata.csv (see decompress.sh)")
            continue
        ap = importlib.import_module(target).parser()
        ap.prog = f"toolbox.py -s {mode}"
        print(ap.format_help())


def main():
    """-h -> overview; -s <mode> [args] -> runs the mode with the remaining arguments."""
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        usage()
        return
    if argv[0] != "-s" or len(argv) < 2 or argv[1] not in MODES:
        print(f"usage: python toolbox.py -s <mode> [parameters], modes: {', '.join(MODES)}  (-h for help)")
        sys.exit(2)
    mode, rest = argv[1], argv[2:]
    target = MODES[mode][1]
    os.chdir(HERE)                                   # results/ and data/ are relative to the toolbox
    if isinstance(target, list):
        if rest:
            print(f"-s {mode} has no parameters, ignoring {' '.join(rest)}")
        for script in target:
            print(f"\n== {script}")
            runpy.run_path(script, run_name="__main__")
        return
    module = importlib.import_module(target)
    sys.argv[0] = f"toolbox.py -s {mode}"            # name shown in the usage line of argparse
    module.main(rest)


if __name__ == "__main__":
    main()
