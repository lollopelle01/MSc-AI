"""Shared training-loop scaffolding used by the GraphSAGE/SEAL/MLP training
scripts: seeding, output-path construction, early-stopping bookkeeping,
epoch/eval CSV writing, and the `--seeds` multi-run driver. Doesn't include
the training step itself or model construction, since those differ per
family (loss, batching) in ways that aren't incidental duplication."""
import csv
import os
import random
import time

import torch


def _phase(msg: str, start: float) -> None:
    """Prints a phase name plus elapsed wall-clock time since `start`."""
    print(f"[phase] {msg} ({time.perf_counter() - start:.1f}s)")


def seed_run(seed: int) -> None:
    """Seeds torch, random, and (if available) CUDA from `seed`."""
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_out_path(args_out: str, tag: str = "", out_dir: str | None = None) -> str:
    """Builds the checkpoint output path from args.out + tag + out_dir.
    `tag` (e.g. "_seed3") namespaces a multi-seed run's filenames; `out_dir`,
    if given, redirects the checkpoint into that folder."""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        stem, ext = os.path.splitext(os.path.basename(args_out))
        return os.path.join(out_dir, f"{stem}{tag}{ext}")
    return args_out if not tag else f"{os.path.splitext(args_out)[0]}{tag}{os.path.splitext(args_out)[1]}"


class EarlyStopper:
    """Tracks best-dev-metric-so-far and patience-based early stopping:
    improves if `metric > best + min_delta`, else counts a non-improving
    epoch and signals stop once `patience` accumulate (patience <= 0
    disables early stopping). Does not save checkpoints itself."""

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = -1.0
        self.epochs_no_improve = 0

    def update(self, metric: float) -> bool:
        """Returns True if `metric` is an improvement (caller should save a
        checkpoint), False otherwise. Updates internal state either way."""
        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.epochs_no_improve = 0
            return True
        self.epochs_no_improve += 1
        return False

    def should_stop(self) -> bool:
        return self.patience > 0 and self.epochs_no_improve >= self.patience


def write_epoch_csv(path: str, epoch_rows: list) -> None:
    """Writes the epoch-level metrics CSV: ["epoch", "train_loss", "dev_loss",
    "dev_auc", "dev_ap", "elapsed_s"], one row per epoch."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "dev_loss", "dev_auc", "dev_ap", "elapsed_s"])
        writer.writerows(epoch_rows)


def write_eval_csv(path: str, split_rows: list) -> None:
    """Writes the eval-level metrics CSV: ["split", "mode", "auc", "ap"].
    `split_rows` is a list of (split, mode, auc, ap) tuples; the row count
    varies by caller (e.g. SEAL has no held-out train split)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "mode", "auc", "ap"])
        writer.writerows(split_rows)


def run_multi_seed(args, run_one_seed, seed_csv_name: str, seed_csv_fieldnames: list, results_dir: str) -> None:
    """Shared `--seeds` driver: calls `run_one_seed(args, tag, out_dir)` once
    per seed in args.seeds, collects each result dict, and writes one row
    per seed to a CSV. `run_one_seed` must accept (args, tag, out_dir) and
    return a dict whose keys match `seed_csv_fieldnames`."""
    rows = []
    for seed in args.seeds:
        args.seed = seed
        rows.append(run_one_seed(args, tag=f"_seed{seed}", out_dir=args.seed_dir))
    seed_dir = args.seed_dir or results_dir
    os.makedirs(seed_dir, exist_ok=True)
    seed_csv = os.path.join(seed_dir, seed_csv_name)
    with open(seed_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seed_csv_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nper-seed eval metrics written to {seed_csv}")
