"""Shared atomic on-disk pickle-cache pattern: load tolerating a
missing/corrupted file, write atomically via temp file + os.replace so a
crash mid-write never leaves a truncated cache."""
import os
import pickle
from typing import Literal


def load_cache(path: str, on_corrupt: Literal["backup", "discard"] = "discard") -> dict:
    """Loads a pickle cache, tolerating a missing or corrupted (truncated)
    file. `on_corrupt="backup"` renames the corrupted file to `<path>.broken`
    and prints a message before starting fresh; `on_corrupt="discard"`
    (the default) starts fresh silently."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (EOFError, pickle.UnpicklingError) as e:
        if on_corrupt == "backup":
            backup = path + ".broken"
            os.replace(path, backup)
            print(f"[cache] {path} was corrupted ({e}); moved to {backup}, starting fresh.")
        return {}


def save_cache(path: str, obj: dict) -> None:
    """Atomic write: temp file + rename, so a crash mid-write never leaves a
    truncated cache."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)
