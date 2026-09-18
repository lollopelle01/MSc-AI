"""
generate_toc.py
===============
Genera e/o inserisce un indice (Table of Contents) in un notebook Jupyter.

Uso
---
    python generate_toc.py notebook.ipynb            # stampa la TOC
    python generate_toc.py notebook.ipynb --insert   # inserisce come prima cella
    python generate_toc.py notebook.ipynb --insert --position 2  # inserisce alla cella 2
"""

import re
import json
import argparse
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def _make_anchor(text: str) -> str:
    """GitHub-flavored markdown anchor da un titolo."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)   # rimuove punteggiatura
    text = re.sub(r"\s+", "-", text)        # spazi → trattini
    return text


def _first_heading(cell_source: list[str]) -> tuple[int, str] | None:
    """Restituisce (level, text) del primo heading markdown nella cella, o None."""
    for line in cell_source:
        line = line.rstrip("\n")
        if not line.startswith("#"):
            continue
        level = len(line) - len(line.lstrip("#"))
        text  = line.lstrip("#").strip()
        return level, text
    return None


def build_toc(nb: dict, title: str = "## Table of Contents") -> str:
    """Costruisce la stringa markdown della TOC."""
    lines = [f"{title}\n\n"]
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        result = _first_heading(cell["source"])
        if result is None:
            continue
        level, text = result
        indent = "  " * (level - 1)
        anchor = _make_anchor(text)
        lines.append(f"{indent}- [{text}](#{anchor})\n")
    return "".join(lines)


def _make_toc_cell(toc_md: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata":  {},
        "source":    toc_md.splitlines(keepends=True),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_toc(
    nb_path: str | Path,
    insert: bool = False,
    position: int = 0,
    toc_title: str = "## Table of Contents",
) -> str:
    """
    Genera la TOC di un notebook Jupyter.

    Parameters
    ----------
    nb_path   : percorso al file .ipynb
    insert    : se True, inserisce la TOC nel notebook e salva
    position  : indice della cella dove inserire (default 0 = in cima)
    toc_title : titolo markdown della sezione TOC

    Returns
    -------
    str : stringa markdown della TOC generata
    """
    nb_path = Path(nb_path)
    if not nb_path.exists():
        raise FileNotFoundError(f"File non trovato: {nb_path}")

    with nb_path.open(encoding="utf-8") as f:
        nb = json.load(f)

    toc_md = build_toc(nb, title=toc_title)

    if insert:
        # Rimuove eventuali TOC già presenti (celle che iniziano col titolo)
        nb["cells"] = [
            c for c in nb["cells"]
            if not (
                c["cell_type"] == "markdown"
                and "".join(c["source"]).startswith(toc_title)
            )
        ]
        toc_cell = _make_toc_cell(toc_md)
        nb["cells"].insert(position, toc_cell)

        with nb_path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

        print(f"TOC inserita in posizione {position} in '{nb_path}'.")

    return toc_md


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera (e opzionalmente inserisce) la TOC di un notebook Jupyter."
    )
    parser.add_argument("notebook", help="Percorso al file .ipynb")
    parser.add_argument(
        "--insert", action="store_true",
        help="Inserisce la TOC nel notebook e salva il file"
    )
    parser.add_argument(
        "--position", type=int, default=0, metavar="N",
        help="Indice della cella dove inserire la TOC (default: 0)"
    )
    parser.add_argument(
        "--title", default="## Table of Contents", metavar="TITOLO",
        help='Titolo della sezione TOC (default: "## Table of Contents")'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    toc  = generate_toc(
        nb_path   = args.notebook,
        insert    = args.insert,
        position  = args.position,
        toc_title = args.title,
    )
    print(toc)