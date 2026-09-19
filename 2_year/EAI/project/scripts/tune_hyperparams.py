"""Grid-search hyperparameter tuning driver for the train_*.py scripts.
Each trial is a subprocess call to the target script with its checkpoint
written under --out-dir; subprocess isolation avoids leftover RNG state
or CUDA memory between trials. The driver parses each trial's "Best dev
<metric>: <value>" stdout line via regex to rank trials, then deletes
all but the best checkpoint (pass --keep-all-checkpoints to keep them).
mlp_raw/mlp_encoded are the graph-free baselines, comparable to
graphsage/seal; their _hard variants use hard negatives + BPR loss and
are comparable to graphsage_hard/seal_hard instead, not to each other."""
import argparse
import itertools
import os
import re
import subprocess
import sys
import time
from csv import writer as csv_writer

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPTS_DIR, "..", "GNNs", "results")
WEIGHTS_DIR = os.path.join(SCRIPTS_DIR, "..", "GNNs", "weights")

SCRIPT_FILENAMES = {
    "graphsage": "train_graphsage.py",
    "graphsage_hard": "train_graphsage_hard.py",
    "seal": "train_seal.py",
    "seal_hard": "train_seal_hard.py",
    "mlp_raw": "train_mlp_baseline_raw.py",
    "mlp_encoded": "train_mlp_baseline_encoded.py",
    "mlp_raw_hard": "train_mlp_baseline_raw_hard.py",
    "mlp_encoded_hard": "train_mlp_baseline_encoded_hard.py",
}

# Starting grids per script -- overridden/extended per-param by --param.
DEFAULT_GRIDS = {
    "graphsage": {"lr": ["1e-3", "5e-4"], "hidden-dim": ["64", "128"], "weight-decay": ["5e-4", "1e-4"]},
    "graphsage_hard": {"lr": ["1e-3", "5e-4"], "hidden-dim": ["64", "128"], "weight-decay": ["5e-4", "1e-4"]},
    "seal": {"lr": ["1e-3", "5e-4"], "hidden-dim": ["32", "64"], "label-dim": ["16", "32"]},
    "seal_hard": {"lr": ["1e-3", "5e-4"], "hidden-dim": ["32", "64"], "label-dim": ["16", "32"]},
    # No encoder, so num-layers is unused here.
    "mlp_raw": {"lr": ["1e-3", "5e-4"], "hidden-dim": ["64", "128"], "weight-decay": ["5e-4", "1e-4"]},
    "mlp_encoded": {"lr": ["1e-3", "5e-4"], "hidden-dim": ["64", "128"],
                     "weight-decay": ["5e-4", "1e-4"], "num-layers": ["2", "3"]},
    "mlp_raw_hard": {"lr": ["1e-3", "5e-4"], "hidden-dim": ["64", "128"], "weight-decay": ["5e-4", "1e-4"]},
    "mlp_encoded_hard": {"lr": ["1e-3", "5e-4"], "hidden-dim": ["64", "128"],
                          "weight-decay": ["5e-4", "1e-4"], "num-layers": ["2", "3"]},
}

BEST_DEV_RE = re.compile(r"Best dev (\w+)(?: \(hard negatives\))?:\s*([0-9.]+)")


def parse_param_arg(spec: str) -> tuple[str, list[str]]:
    """'hidden-dim=64,128' -> ('hidden-dim', ['64', '128'])."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--param must be name=v1,v2,...: got {spec!r}")
    name, values = spec.split("=", 1)
    return name.strip(), [v.strip() for v in values.split(",") if v.strip()]


def run_trial(script_path, base_args, param_dict, out_path):
    """Runs one subprocess trial, returns (metric_name, score, elapsed);
    metric_name/score are None if the run failed or the output line was
    missing."""
    argv = [sys.executable, script_path, *base_args, "--out", out_path]
    for name, value in param_dict.items():
        argv += [f"--{name}", value]

    t0 = time.perf_counter()
    proc = subprocess.run(argv, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"  [FAILED, {elapsed:.1f}s, exit {proc.returncode}] stderr tail:")
        print("    " + "\n    ".join(proc.stderr.strip().splitlines()[-8:]))
        return None, None, elapsed

    match = BEST_DEV_RE.search(proc.stdout)
    if not match:
        print(f"  [WARNING, {elapsed:.1f}s] couldn't find a 'Best dev <metric>: <value>' line in stdout")
        return None, None, elapsed

    metric_name, score = match.group(1), float(match.group(2))
    return metric_name, score, elapsed


def main():
    parser = argparse.ArgumentParser(description="Grid-search hyperparameter tuning for the train_*.py scripts")
    parser.add_argument("--script", required=True, choices=sorted(SCRIPT_FILENAMES),
                         help="which train_*.py script to tune")
    parser.add_argument("--param", action="append", default=[], type=parse_param_arg, dest="params",
                         help="name=v1,v2,... -- repeatable; overrides/extends the default grid for that param")
    parser.add_argument("--epochs", type=int, default=None, help="passed through to every trial if set")
    parser.add_argument("--patience", type=int, default=None, help="passed through to every trial if set")
    parser.add_argument("--min-delta", type=float, default=None, help="passed through to every trial if set")
    parser.add_argument("--val-metric", type=str, default=None, choices=["auc", "ap"],
                         help="passed through to every trial if set")
    parser.add_argument("--seed", type=int, default=0, help="passed through to every trial (kept fixed across the grid)")
    parser.add_argument("--device", type=str, default=None, help="passed through to every trial if set")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="where trial checkpoints go (default: GNNs/weights/tuning/<script>)")
    parser.add_argument("--decoder-batch-size", type=int, default=None,
                         help="passed through to every trial if set; only relevant to scripts that accept "
                              "it (graphsage, graphsage_hard, mlp_raw, mlp_encoded). Lower on CUDA OOM.")
    parser.add_argument("--keep-all-checkpoints", action="store_true",
                         help="keep every trial's checkpoint instead of only the best one")
    parser.add_argument("--dry-run", action="store_true", help="print the grid and planned commands, run nothing")
    args = parser.parse_args()

    grid = dict(DEFAULT_GRIDS[args.script])
    for name, values in args.params:
        grid[name] = values

    param_names = list(grid.keys())
    combos = list(itertools.product(*(grid[name] for name in param_names)))
    print(f"Grid for --script {args.script}: " + ", ".join(f"{n}={grid[n]}" for n in param_names))
    print(f"{len(combos)} trial(s) total")

    base_args = []
    for flag, val in (("--epochs", args.epochs), ("--patience", args.patience),
                       ("--min-delta", args.min_delta), ("--val-metric", args.val_metric),
                       ("--seed", args.seed), ("--device", args.device),
                       ("--decoder-batch-size", args.decoder_batch_size)):
        if val is not None:
            base_args += [flag, str(val)]

    out_dir = args.out_dir or os.path.join(WEIGHTS_DIR, "tuning", args.script)
    script_path = os.path.join(SCRIPTS_DIR, SCRIPT_FILENAMES[args.script])

    if args.dry_run:
        os.makedirs(out_dir, exist_ok=True)  # harmless even in dry-run; keeps the printed path real
        for i, combo in enumerate(combos):
            param_dict = dict(zip(param_names, combo))
            out_path = os.path.join(out_dir, f"trial_{i:03d}.pt")
            argv = [sys.executable, script_path, *base_args, "--out", out_path]
            for name, value in param_dict.items():
                argv += [f"--{name}", value]
            print(f"  [{i}] " + " ".join(argv))
        return

    os.makedirs(out_dir, exist_ok=True)
    run_start = time.perf_counter()
    results = []  # (param_dict, metric_name, score, elapsed, out_path)
    for i, combo in enumerate(combos):
        param_dict = dict(zip(param_names, combo))
        out_path = os.path.join(out_dir, f"trial_{i:03d}.pt")
        print(f"[{i + 1}/{len(combos)}] " + ", ".join(f"{n}={v}" for n, v in param_dict.items()))
        metric_name, score, elapsed = run_trial(script_path, base_args, param_dict, out_path)
        if score is not None:
            print(f"  dev_{metric_name}={score:.4f}  ({elapsed:.1f}s)")
        results.append((param_dict, metric_name, score, elapsed, out_path))

    scored = [r for r in results if r[2] is not None]
    if not scored:
        print("\nNo trial produced a usable score -- nothing to report.")
        return
    scored.sort(key=lambda r: r[2], reverse=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_csv = os.path.join(RESULTS_DIR, f"{args.script}_tuning_results.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv_writer(f)
        writer.writerow(["rank", *param_names, "metric", "score", "elapsed_s", "checkpoint"])
        for rank, (param_dict, metric_name, score, elapsed, out_path) in enumerate(scored, start=1):
            writer.writerow([rank, *(param_dict[n] for n in param_names), metric_name, score, round(elapsed, 1), out_path])
    print(f"\nTuning results written to {summary_csv}")

    print(f"\nRanked results ({len(scored)}/{len(combos)} trials succeeded):")
    for rank, (param_dict, metric_name, score, elapsed, out_path) in enumerate(scored, start=1):
        params_str = ", ".join(f"{n}={v}" for n, v in param_dict.items())
        print(f"  #{rank}  dev_{metric_name}={score:.4f}  {params_str}  ({out_path})")

    best_params, best_metric, best_score, _, best_path = scored[0]
    print(f"\nBest: dev_{best_metric}={best_score:.4f}  " + ", ".join(f"{n}={v}" for n, v in best_params.items()))
    print(f"Best checkpoint: {best_path}")

    if not args.keep_all_checkpoints:
        removed = 0
        for _, _, _, _, out_path in results:
            if out_path != best_path and os.path.exists(out_path):
                os.remove(out_path)
                removed += 1
        print(f"Removed {removed} non-best trial checkpoint(s) (pass --keep-all-checkpoints to keep them)")

    print(f"\nTotal tuning time: {time.perf_counter() - run_start:.1f}s")


if __name__ == "__main__":
    main()
