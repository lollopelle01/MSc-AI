"""
plot_utils.py
=============
Utilities for plotting K-means HPC experiment grids (OpenMP + CUDA).

Main entry points
-----------------
    plot_grid(df, metric, x_var, ...)          – single-experiment grid
    plot_experiments(experiments, metric, ...) – multi-experiment comparison
    grouped_plots(df, metric, x_var, ...)      – nested group grid (e.g. N × D × K)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional, Union

# ──────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

TASKS  = ["classify", "update", "tot"]
LABELS = {"classify": "Classify", "update": "Update", "tot": "Total"}
TASK_COLORS = {
    "classify": "#1f77b4",   # blue
    "update":   "#2ca02c",   # green
    "tot":      "#d62728",   # red
}

VAR_COLS = {
    "N": "n_points",
    "D": "n_dims",
    "K": "k",
    "P": "threads",    # OMP
    "B": "block_dim",  # CUDA
}

OMP_METRICS = {
    "time":       ("t_{task}_mean",         "t_{task}_std",          "Time (s)"),
    "speedup":    ("speedup_{task}_mean",    "speedup_{task}_std",    "Speedup"),
    "strong_eff": ("strong_eff_{task}_mean", "strong_eff_{task}_std", "Strong Efficiency"),
    "weak_eff":   ("weak_eff_{task}_mean",   "weak_eff_{task}_std",   "Weak Efficiency"),
}

CUDA_METRICS = {
    "time":       ("t_{task}_mean",         "t_{task}_std",          "Time (s)"),
    "throughput": ("throughput_{task}_mean", "throughput_{task}_std", "Throughput (op/s)"),
}

# ──────────────────────────────────────────────────────────────────────────────
# 2.  INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _metric_cols(metric: str, task: str, backend: str):
    table = OMP_METRICS if backend == "omp" else CUDA_METRICS
    if metric not in table:
        raise ValueError(
            f"Unknown metric '{metric}' for backend '{backend}'. "
            f"Available: {list(table)}"
        )
    tmpl_mean, tmpl_std, label = table[metric]
    return (
        tmpl_mean.replace("{task}", task),
        tmpl_std.replace("{task}",  task),
        label,
    )


def _col_label(key: str) -> str:
    labels = {
        "N": "N (points)", "D": "D (dims)", "K": "K (clusters)",
        "P": "Threads (p)", "B": "Block dim",
        "n_points": "N (points)", "n_dims": "D (dims)", "k": "K",
        "threads": "Threads (p)", "block_dim": "Block dim",
    }
    return labels.get(key, key)


def _squarish(n: int) -> tuple:
    nrows = math.floor(math.sqrt(n))
    while n % nrows != 0:
        nrows -= 1
    return nrows, n // nrows


def _resolve_layout(layout, n_panels: int) -> tuple:
    if layout == "row":
        return 1, n_panels
    elif layout == "square":
        return _squarish(n_panels) if n_panels > 0 else (1, 1)
    else:
        R, C = layout
        if R * C < n_panels:
            raise ValueError(f"Layout {R}x{C} too small for {n_panels} panels.")
        return R, C


def _resolve_col(key, df):
    if key is None:
        return None
    if key in VAR_COLS:
        return VAR_COLS[key]
    if key in df.columns:
        return key
    raise ValueError(f"Unknown variable key '{key}'. Known: {list(VAR_COLS)}")


def _apply_fixed(df, fixed: dict):
    for k, v in (fixed or {}).items():
        c = VAR_COLS.get(k, k)
        if c in df.columns:
            df = df[df[c] == v]
    return df


def _apply_filter_vals(df, filter_vals: dict):
    """Keep only rows where column values are in the given lists.

    Example: filter_vals={"N": [50_000, 400_000]} keeps only those two N values.
    """
    for k, vals in (filter_vals or {}).items():
        c = VAR_COLS.get(k, k)
        if c in df.columns:
            df = df[df[c].isin(vals)]
    return df


def _draw_errorbar(ax, xs, ys, errs, color, style: str):
    """Draw error representation: 'fill' = shaded area, 'bar' = standard errorbars."""
    if style == "bar":
        ax.errorbar(xs, ys, yerr=errs, fmt="none", ecolor=color,
                    elinewidth=1.2, capsize=3, capthick=1.2, alpha=0.7)
    else:  # fill
        ax.fill_between(xs, ys - errs, ys + errs, alpha=0.15, color=color)


def _set_discrete_xticks(ax, x_values, log_x: bool = False, xtick_fmt: str = "auto"):
    """
    Imposta i tick x per ogni valore discreto presente nei dati.

    Parameters
    ----------
    log_x     : se True la scala è logaritmica (base 2) e viene aggiunto un
                piccolo margine moltiplicativo ai limiti.
    xtick_fmt : "auto"  – etichette come interi normali  (1, 2, 4, 8, ...)
                "pow2"  – etichette come potenze di 2     ($2^0$, $2^1$, ...)
                          su qualsiasi scala (lineare o logaritmica).
                "k"     – valori divisi per 1000 con suffisso 'k'
                          (50000 → 50k, 400000 → 400k, ...)

    Nota: log_x e xtick_fmt sono indipendenti — si può avere scala log con
    etichette normali, o scala lineare con etichette $2^n$.
    """
    vals = sorted(np.unique(x_values).tolist())

    ax.set_xticks(vals)

    if xtick_fmt == "pow2":
        def _fmt(x, pos):
            try:
                exp = int(round(np.log2(x)))
                return f"$2^{{{exp}}}$"
            except Exception:
                return str(x)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    elif xtick_fmt == "k":
        def _fmt_k(x, pos):
            v = x / 1_000
            return f"{int(v)}k" if v == int(v) else f"{v:.1f}k"
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_k))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    else:  # "auto"
        ax.set_xticklabels([
            str(int(v) if v == int(v) else v) for v in vals
        ])
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    # Su scala log il margine di default è eccessivo → lo stringiamo
    if log_x:
        ax.set_xlim(vals[0] * 0.85, vals[-1] * 1.15)


def _build_grid_panels(row_col, col_col, row_vals, col_vals, layout):
    if row_col is None and col_col is None:
        return 1, 1, [[(None, None)]]

    all_panels = [
        (rv, cv)
        for rv in (row_vals if row_col else [None])
        for cv in (col_vals if col_col else [None])
    ]
    n = len(all_panels)

    # layout esplicito ha sempre la precedenza
    if layout not in ("row", "square"):
        R, C = layout
        if R * C < n:
            raise ValueError(f"Layout {R}x{C} troppo piccolo per {n} pannelli.")
    elif row_col and col_col and len(row_vals) > 1 and len(col_vals) > 1:
        # caso classico N×M: usa la griglia naturale
        R, C = len(row_vals), len(col_vals)
    else:
        R, C = _resolve_layout(layout, n)

    grid = [all_panels[i:i+C] for i in range(0, n, C)]
    # padding dell'ultima riga se necessario
    while len(grid[-1]) < C:
        grid[-1].append((None, None))

    return R, C, grid


def _filter_panel(df, row_col, rv, col_col, cv):
    sub = df
    if row_col and rv is not None:
        sub = sub[sub[row_col] == rv]
    if col_col and cv is not None:
        sub = sub[sub[col_col] == cv]
    return sub


def _fmt_val(v):
    try:
        return str(int(v)) if float(v) == int(float(v)) else str(v)
    except Exception:
        return str(v)


def _panel_title(row_var, rv, col_var, cv, row_col, col_col,
                 row_var2=None, rv2=None, col_var2=None, cv2=None,
                 row_col2=None, col_col2=None):
    parts = []
    if row_col and rv is not None:
        label = f"{row_var}={_fmt_val(rv)}"
        if row_col2 and rv2 is not None:
            label += f", {row_var2}={_fmt_val(rv2)}"
        parts.append(label)
    if col_col and cv is not None:
        label = f"{col_var}={_fmt_val(cv)}"
        if col_col2 and cv2 is not None:
            label += f", {col_var2}={_fmt_val(cv2)}"
        parts.append(label)
    return " | ".join(parts)


def _resolve_eb_style(errorbar):
    if errorbar is False or errorbar is None:
        return None
    elif errorbar is True or errorbar == "fill":
        return "fill"
    elif errorbar == "bar":
        return "bar"
    else:
        raise ValueError(
            f"errorbar must be False, True, 'fill', or 'bar'. Got: {errorbar!r}"
        )


def _collect_legend_handles(fig):
    """Raccoglie handle e label unici da tutti gli assi della figura."""
    handles, labels = [], []
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    return handles, labels


def _finalize_fig(fig, title, legend, legend_y, legend_fontsize, savepath, show,
                  legend_position: Optional[str] = None,
                  legend_ncol: Optional[int] = None):
    """Shared end-of-function logic: legend, suptitle, save, display, close.

    Parameters
    ----------
    legend_position : None      → legenda per-subplot (default originale)
                      "top"     → legenda condivisa sopra tutti i pannelli
                      "bottom"  → legenda condivisa sotto tutti i pannelli
    legend_ncol     : None      → ncol = numero totale di label (tutto su una riga)
                      int       → numero fisso di colonne della legenda
    """
    fig.tight_layout()

    # ── legenda condivisa sopra / sotto ───────────────────────────────────────
    if legend_position in ("top", "bottom"):
        if title:
            fig.suptitle(title, fontsize=12,
                         y=1.03 if legend_position == "top" else 1.01)
        if legend:
            handles, labels = _collect_legend_handles(fig)
            if handles:
                ncol = legend_ncol if legend_ncol is not None else len(labels)
                if legend_position == "top":
                    fig.legend(handles, labels,
                               loc="upper center",
                               bbox_to_anchor=(0.5, 1.10),
                               ncol=ncol,
                               fontsize=legend_fontsize,
                               frameon=True)
                else:  # "bottom"
                    fig.legend(handles, labels,
                               loc="lower center",
                               bbox_to_anchor=(0.5, -0.06),
                               ncol=ncol,
                               fontsize=legend_fontsize,
                               frameon=True)

    # ── vecchio comportamento con legend_y float ──────────────────────────────
    elif legend_y is not None:
        _ly = max(0.0, min(legend_y, 0.5))
        if _ly > 0:
            fig.subplots_adjust(top=max(1.0 - _ly, 0.55))
        if title:
            fig.suptitle(title, fontsize=12, y=1.0 - _ly * 0.1)
        if legend:
            handles, labels = _collect_legend_handles(fig)
            ncol = legend_ncol if legend_ncol is not None else len(labels)
            fig.legend(handles, labels,
                       loc="upper center",
                       bbox_to_anchor=(0.5, 1.0 - _ly * 0.55),
                       ncol=ncol,
                       fontsize=legend_fontsize,
                       frameon=True)

    # ── legenda per-subplot (default) ────────────────────────────────────────
    else:
        if title:
            fig.suptitle(title, fontsize=12, y=1.01)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=150)

    if show:
        from IPython.display import display as _ipy_display
        _ipy_display(fig)
    plt.close(fig)
    plt.ion()


# ──────────────────────────────────────────────────────────────────────────────
# 3.  CORE FUNCTION: plot_grid
# ──────────────────────────────────────────────────────────────────────────────

def plot_grid(
    df,
    metric: str,
    *,
    x_var: str,
    row_var: Optional[str] = None,
    row_var2: Optional[str] = None,
    col_var: Optional[str] = None,
    col_var2: Optional[str] = None,
    tasks: list = TASKS,
    backend: str = "omp",
    fixed: Optional[dict] = None,
    filter_vals: Optional[dict] = None,
    layout = "square",
    # uniform axes across panels
    sharey: str = "row",
    sharex: str = "col",
    # visuals
    log_x: bool = False,
    log_y: bool = False,
    xtick_fmt: str = "auto",
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    ideal_line: Optional[str] = None,
    figsize_per_panel: tuple = (4.5, 3.5),
    title: Optional[str] = None,
    legend: bool = True,
    legend_loc: Optional[str] = None,
    legend_fontsize: int = 9,
    legend_y: Optional[float] = None,
    legend_position: Optional[str] = None,
    legend_ncol: Optional[int] = None,
    errorbar: Union[bool, str] = True,
    colormap: str = "tab10",
    savepath: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a grid of subplots from a single HPC experiment DataFrame.

    Parameters
    ----------
    df           : result DataFrame
    metric       : "time" | "speedup" | "strong_eff" | "weak_eff"  (omp)
                   "time" | "throughput"                            (cuda)
    x_var        : x-axis variable key: "N" "D" "K" "P" "B"
    row_var      : variable whose values become grid rows  (None = no rows)
    col_var      : variable whose values become grid cols  (None = no cols)
    tasks        : tasks to overlay per panel
    backend      : "omp" | "cuda"
    fixed        : exact-value filters, e.g. {"K": 5}
    filter_vals  : subset filters, e.g. {"N": [50_000, 400_000]}
    layout       : panel arrangement when <=1 facet var: "row"|"square"|(R,C)
    sharey       : "row" | "col" | "all" | "none"
    sharex       : "row" | "col" | "all" | "none"
    log_x/log_y  : logarithmic scale
    xtick_fmt    : "auto"  – integer labels (1, 2, 4, 8, ...)
                   "pow2"  – power-of-2 labels ($2^0$, $2^1$, ...)
                   "k"     – divide by 1000 with 'k' suffix (50k, 400k, ...)
    xlim         : (xmin, xmax) applied to every panel
    ylim         : (ymin, ymax) applied to every panel
    ideal_line   : "speedup" = diagonal, "efficiency" = y=1
    errorbar     : False | True/"fill" | "bar"
    legend_position : None      → per-subplot legend
                      "top"     → single shared legend above all panels
                      "bottom"  → single shared legend below all panels
    legend_ncol  : number of columns for shared legend (None = all in one row)
    legend_y     : float 0–0.5 = shared legend above (legacy parameter)
    legend_loc   : location string for per-subplot legend
    """
    df = df.copy()

    x_col    = _resolve_col(x_var,    df)
    row_col  = _resolve_col(row_var,  df)
    row_col2 = _resolve_col(row_var2, df)
    col_col  = _resolve_col(col_var,  df)
    col_col2 = _resolve_col(col_var2, df)

    df = _apply_fixed(df, fixed)
    df = _apply_filter_vals(df, filter_vals)

    eb_style = _resolve_eb_style(errorbar)

    if row_col and row_col2:
        row_vals = sorted(set(zip(df[row_col], df[row_col2])))
    else:
        row_vals = sorted(df[row_col].unique()) if row_col else [None]

    if col_col and col_col2:
        col_vals = sorted(set(zip(df[col_col], df[col_col2])))
    else:
        col_vals = sorted(df[col_col].unique()) if col_col else [None]

    nR, nC, grid_panels = _build_grid_panels(
        row_col, col_col, row_vals, col_vals, layout
    )

    cmap   = plt.get_cmap(colormap)
    colors = {
        t: TASK_COLORS.get(t, cmap(i / max(len(tasks) - 1, 1)))
        for i, t in enumerate(tasks)
    }

    # se c'è una legenda condivisa, non metterla nei singoli pannelli
    per_panel_legend = legend and (legend_position is None) and (legend_y is None)

    plt.ioff()
    fig, axes = plt.subplots(
        nR, nC,
        figsize=(figsize_per_panel[0] * nC, figsize_per_panel[1] * nR),
        sharey=sharey,
        sharex=sharex,
        squeeze=False,
    )

    y_label_global = None

    for ri in range(nR):
        for ci in range(nC):
            ax = axes[ri][ci]

            try:
                rv, cv = grid_panels[ri][ci]
            except IndexError:
                ax.set_visible(False)
                continue

            if row_col and row_col2 and isinstance(rv, tuple):
                rv, rv2 = rv
            else:
                rv2 = None
            if col_col and col_col2 and isinstance(cv, tuple):
                cv, cv2 = cv
            else:
                cv2 = None

            sub = _filter_panel(df, row_col, rv, col_col, cv)
            if rv2 is not None and row_col2:
                sub = sub[sub[row_col2] == rv2]
            if cv2 is not None and col_col2:
                sub = sub[sub[col_col2] == cv2]

            if sub.empty:
                ax.set_visible(False)
                continue

            for task in tasks:
                mean_col, std_col, y_label = _metric_cols(metric, task, backend)
                y_label_global = y_label
                if mean_col not in sub.columns:
                    continue
                agg = {mean_col: "mean"}
                if std_col in sub.columns:
                    agg[std_col] = "mean"
                grp = (sub.groupby(x_col, as_index=False)
                          .agg(agg)
                          .sort_values(x_col))
                xs, ys = grp[x_col].values, grp[mean_col].values
                ax.plot(xs, ys, marker="o", label=LABELS[task], color=colors[task])
                if eb_style and std_col in grp.columns:
                    _draw_errorbar(ax, xs, ys, grp[std_col].values,
                                   colors[task], eb_style)

            if ideal_line == "speedup":
                xl = np.array(sorted(sub[x_col].unique()))
                ax.plot(xl, xl, "k--", lw=0.8, label="Ideal")
            elif ideal_line == "efficiency":
                ax.axhline(1.0, color="k", ls="--", lw=0.8, label="Ideal")

            if log_x:
                ax.set_xscale("log", base=2)
            if log_y:
                ax.set_yscale("log")
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            _set_discrete_xticks(ax, sub[x_col].values,
                                 log_x=log_x, xtick_fmt=xtick_fmt)

            ax.set_title(_panel_title(
                row_var, rv, col_var, cv, row_col, col_col,
                row_var2, rv2, col_var2, cv2, row_col2, col_col2,
            ), fontsize=9)
            ax.grid(True, alpha=0.3)

            if ri == nR - 1:
                ax.set_xlabel(_col_label(x_var))
            if ci == 0:
                ax.set_ylabel(y_label_global or "")

            if per_panel_legend:
                kw_leg = dict(fontsize=legend_fontsize)
                if legend_loc:
                    kw_leg["loc"] = legend_loc
                ax.legend(**kw_leg)

    _finalize_fig(fig, title, legend, legend_y, legend_fontsize, savepath, show,
                  legend_position=legend_position, legend_ncol=legend_ncol)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MULTI-EXPERIMENT COMPARISON: plot_experiments
# ──────────────────────────────────────────────────────────────────────────────

def plot_experiments(
    experiments: dict,
    metric: str,
    *,
    x_var: str,
    row_var: Optional[str] = None,
    row_var2: Optional[str] = None,
    col_var: Optional[str] = None,
    col_var2: Optional[str] = None,
    tasks: list = ("tot",),
    backend: str = "omp",
    fixed: Optional[dict] = None,
    filter_vals: Optional[dict] = None,
    layout = "square",
    sharey: str = "row",
    sharex: str = "col",
    log_x: bool = False,
    log_y: bool = False,
    xtick_fmt: str = "auto",
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    ideal_line: Optional[str] = None,
    figsize_per_panel: tuple = (4.5, 3.5),
    title: Optional[str] = None,
    legend: bool = True,
    legend_loc: Optional[str] = None,
    legend_fontsize: int = 9,
    legend_y: Optional[float] = None,
    legend_position: Optional[str] = None,
    legend_ncol: Optional[int] = None,
    errorbar: Union[bool, str] = True,
    savepath: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compare multiple experiments in the same panel grid.

    Parameters
    ----------
    experiments : dict  {label: (df, backend)}  or  {label: df}
        Examples:
            {"OMP base":      (res_omp_base,      "omp"),
             "OMP optimized": (res_omp_optimized,  "omp"),
             "CUDA":          (res_cuda_rtx2080,   "cuda")}
        If all experiments share the same backend, pass just the df and set
        the `backend` parameter accordingly (default "omp").
    metric      : metric name valid for each experiment's backend
    x_var       : x-axis variable ("N","D","K","P","B")
    row_var     : panel row variable
    col_var     : panel column variable
    tasks       : tasks to draw; default ("tot",) for clarity
    backend     : default backend ("omp" | "cuda") used for experiments passed
                  as a plain DataFrame. Experiments passed as (df, backend)
                  tuples ignore this value.
    xtick_fmt   : "auto" | "pow2" | "k"  (see _set_discrete_xticks)
    errorbar    : False | True/"fill" | "bar"
    legend_position : None      → per-subplot legend
                      "top"     → single shared legend above all panels
                      "bottom"  → single shared legend below all panels
    legend_ncol : number of columns for shared legend (None = all in one row)
    """
    _experiments = {}
    for label, val in experiments.items():
        if isinstance(val, tuple) and len(val) == 2 and isinstance(val[1], str):
            _experiments[label] = val
        else:
            _experiments[label] = (val, backend)

    exp_items = list(_experiments.items())
    eb_style  = _resolve_eb_style(errorbar)

    first_df, _ = exp_items[0][1]
    first_df = _apply_fixed(first_df.copy(), fixed)
    first_df = _apply_filter_vals(first_df, filter_vals)
    row_col  = _resolve_col(row_var,  first_df)
    row_col2 = _resolve_col(row_var2, first_df)
    col_col  = _resolve_col(col_var,  first_df)
    col_col2 = _resolve_col(col_var2, first_df)

    if row_col and row_col2:
        row_vals = sorted(set(zip(first_df[row_col], first_df[row_col2])))
    else:
        row_vals = sorted(first_df[row_col].unique()) if row_col else [None]

    if col_col and col_col2:
        col_vals = sorted(set(zip(first_df[col_col], first_df[col_col2])))
    else:
        col_vals = sorted(first_df[col_col].unique()) if col_col else [None]

    nR, nC, grid_panels = _build_grid_panels(
        row_col, col_col, row_vals, col_vals, layout
    )

    exp_cmap    = plt.get_cmap("tab10")
    task_styles = {t: s for t, s in zip(TASKS, ["-", "--", "-."])}
    markers     = ["o", "s", "^", "D", "v", "P", "*", "X"]
    n_exp       = len(exp_items)
    colors      = {
        lbl: exp_cmap(i / max(n_exp - 1, 1))
        for i, (lbl, _) in enumerate(exp_items)
    }

    # se c'è una legenda condivisa, non metterla nei singoli pannelli
    per_panel_legend = legend and (legend_position is None) and (legend_y is None)

    plt.ioff()
    fig, axes = plt.subplots(
        nR, nC,
        figsize=(figsize_per_panel[0] * nC, figsize_per_panel[1] * nR),
        sharey=sharey,
        sharex=sharex,
        squeeze=False,
    )

    y_label_global = None

    for ri in range(nR):
        for ci in range(nC):
            ax = axes[ri][ci]

            try:
                rv, cv = grid_panels[ri][ci]
            except IndexError:
                ax.set_visible(False)
                continue

            if row_col and row_col2 and isinstance(rv, tuple):
                rv, rv2 = rv
            else:
                rv2 = None
            if col_col and col_col2 and isinstance(cv, tuple):
                cv, cv2 = cv
            else:
                cv2 = None

            all_x_vals = []
            has_data   = False

            for ei, (label, (df_e, backend_e)) in enumerate(exp_items):
                df_e  = _apply_fixed(df_e.copy(), fixed)
                df_e  = _apply_filter_vals(df_e, filter_vals)
                x_col = _resolve_col(x_var,    df_e)
                rc    = _resolve_col(row_var,   df_e)
                rc2   = _resolve_col(row_var2,  df_e)
                cc    = _resolve_col(col_var,   df_e)
                cc2   = _resolve_col(col_var2,  df_e)
                sub   = _filter_panel(df_e, rc, rv, cc, cv)
                if rv2 is not None and rc2:
                    sub = sub[sub[rc2] == rv2]
                if cv2 is not None and cc2:
                    sub = sub[sub[cc2] == cv2]

                if sub.empty:
                    continue

                all_x_vals.extend(sub[x_col].values)
                has_data = True

                for task in tasks:
                    try:
                        mean_col, std_col, y_label = _metric_cols(
                            metric, task, backend_e
                        )
                    except ValueError:
                        continue
                    y_label_global = y_label
                    if mean_col not in sub.columns:
                        continue

                    agg = {mean_col: "mean"}
                    if std_col in sub.columns:
                        agg[std_col] = "mean"
                    grp = (sub.groupby(x_col, as_index=False)
                              .agg(agg)
                              .sort_values(x_col))
                    xs  = grp[x_col].values
                    ys  = grp[mean_col].values
                    ls  = task_styles.get(task, "-")
                    mk  = markers[ei % len(markers)]
                    clr = colors[label]
                    lbl = (f"{label} – {LABELS[task]}"
                           if len(tasks) > 1 else label)

                    ax.plot(xs, ys, ls=ls, marker=mk, color=clr, label=lbl)
                    if eb_style and std_col in grp.columns:
                        _draw_errorbar(ax, xs, ys, grp[std_col].values,
                                       clr, eb_style)

            if not has_data:
                ax.set_visible(False)
                continue

            if ideal_line == "speedup":
                xl = np.array(sorted(set(all_x_vals)))
                ax.plot(xl, xl, "k--", lw=0.8, label="Ideal")
            elif ideal_line == "efficiency":
                ax.axhline(1.0, color="k", ls="--", lw=0.8, label="Ideal")

            if log_x:
                ax.set_xscale("log", base=2)
            if log_y:
                ax.set_yscale("log")
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            _set_discrete_xticks(ax, all_x_vals,
                                 log_x=log_x, xtick_fmt=xtick_fmt)

            ax.set_title(_panel_title(
                row_var, rv, col_var, cv, row_col, col_col,
                row_var2, rv2, col_var2, cv2, row_col2, col_col2,
            ), fontsize=9)
            ax.grid(True, alpha=0.3)

            if ri == nR - 1:
                ax.set_xlabel(_col_label(x_var))
            if ci == 0:
                ax.set_ylabel(y_label_global or "")

            if per_panel_legend:
                kw_leg = dict(fontsize=legend_fontsize)
                if legend_loc:
                    kw_leg["loc"] = legend_loc
                ax.legend(**kw_leg)

    _finalize_fig(fig, title, legend, legend_y, legend_fontsize, savepath, show,
                  legend_position=legend_position, legend_ncol=legend_ncol)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 5.  NESTED GROUP GRID: grouped_plots
# ──────────────────────────────────────────────────────────────────────────────

def grouped_plots(
    df,
    metric: str,
    *,
    x_var: str,
    group_var: str,
    row_var: Optional[str] = None,
    col_var: Optional[str] = None,
    tasks: list = TASKS,
    backend: str = "cuda",
    fixed: Optional[dict] = None,
    filter_vals: Optional[dict] = None,
    # scale / axes
    log_x: bool = True,
    log_y: bool = False,
    xtick_fmt: str = "pow2",
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
    ideal_line: Optional[str] = None,
    sharey: str = "row",
    sharex: str = "all",
    # visuals
    figsize_per_panel: tuple = (3.0, 2.5),
    errorbar: Union[bool, str] = "bar",
    colormap: str = "tab10",
    # labels / legend
    title: Optional[str] = None,
    title_y: float = 1.05,
    group_label_y: float = 1.01,
    group_label_fontsize: int = 11,
    legend: bool = True,
    legend_fontsize: int = 8,
    legend_ncol: Optional[int] = None,
    # group spacing & separators
    group_spacing: float = 0.02,
    group_separator: bool = True,
    group_separator_color: str = "lightgray",
    # output
    savepath: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Nested grid: major column groups by `group_var`, inner rows/cols by
    `row_var` / `col_var`.

    Layout example (group_var="N", row_var="D", col_var="K"):
    ┌──── N=50k ────┐     ┌──── N=200k ────┐     ┌──── N=400k ────┐
    │  D=20, K=3…10 │     │  D=20, K=3…10  │     │  D=20, K=3…10  │
    │  D=30, K=3…10 │     │  D=30, K=3…10  │     │  D=30, K=3…10  │
    └───────────────┘     └────────────────┘     └────────────────┘

    Parameters
    ----------
    log_x         : logarithmic scale on x axis (default True for block_dim)
    xtick_fmt     : "auto" | "pow2" | "k"  (default "pow2" for CUDA block dims)
    group_spacing : extra horizontal gap between groups in figure-coord units
    group_separator : draw a dashed vertical line between groups
    legend_ncol   : columns for shared legend (None = all in one row)
    """
    # ── 0. Prepare data ───────────────────────────────────────────────────────
    df = df.copy()
    df = _apply_fixed(df, fixed)
    df = _apply_filter_vals(df, filter_vals)
    eb_style = _resolve_eb_style(errorbar)

    x_col   = _resolve_col(x_var,    df)
    grp_col = _resolve_col(group_var, df)
    row_col = _resolve_col(row_var,   df)
    col_col = _resolve_col(col_var,   df)

    grp_vals = sorted(df[grp_col].unique())
    row_vals = sorted(df[row_col].unique()) if row_col else [None]
    col_vals = sorted(df[col_col].unique()) if col_col else [None]

    n_groups = len(grp_vals)
    nR       = len(row_vals)
    nC_sub   = len(col_vals)
    nC_total = n_groups * nC_sub

    # ── 1. Colors ─────────────────────────────────────────────────────────────
    cmap   = plt.get_cmap(colormap)
    colors = {
        t: TASK_COLORS.get(t, cmap(i / max(len(tasks) - 1, 1)))
        for i, t in enumerate(tasks)
    }

    # ── 2. Figure & axes ──────────────────────────────────────────────────────
    plt.ioff()
    fig, axes = plt.subplots(
        nR, nC_total,
        figsize=(figsize_per_panel[0] * nC_total,
                 figsize_per_panel[1] * nR),
        sharey=sharey,
        sharex=sharex,
        squeeze=False,
    )

    # ── 3. Fill panels ────────────────────────────────────────────────────────
    y_label_global = None

    for gi, gv in enumerate(grp_vals):
        sub_g = df[df[grp_col] == gv]

        for ri, rv in enumerate(row_vals):
            for ci, cv in enumerate(col_vals):
                ax_col = gi * nC_sub + ci
                ax     = axes[ri][ax_col]

                sub = sub_g.copy()
                if row_col and rv is not None:
                    sub = sub[sub[row_col] == rv]
                if col_col and cv is not None:
                    sub = sub[sub[col_col] == cv]

                if sub.empty:
                    ax.set_visible(False)
                    continue

                for task in tasks:
                    mean_col, std_col, y_label = _metric_cols(metric, task, backend)
                    y_label_global = y_label
                    if mean_col not in sub.columns:
                        continue
                    agg = {mean_col: "mean"}
                    if std_col in sub.columns:
                        agg[std_col] = "mean"
                    grp = (sub.groupby(x_col, as_index=False)
                              .agg(agg)
                              .sort_values(x_col))
                    xs, ys = grp[x_col].values, grp[mean_col].values
                    ax.plot(xs, ys, marker="o", markersize=3,
                            label=LABELS[task], color=colors[task])
                    if eb_style and std_col in grp.columns:
                        _draw_errorbar(ax, xs, ys, grp[std_col].values,
                                       colors[task], eb_style)

                if ideal_line == "speedup":
                    xl = np.array(sorted(sub[x_col].unique()))
                    ax.plot(xl, xl, "k--", lw=0.8, label="Ideal")
                elif ideal_line == "efficiency":
                    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="Ideal")

                if log_x:
                    ax.set_xscale("log", base=2)
                if log_y:
                    ax.set_yscale("log")
                if xlim:
                    ax.set_xlim(xlim)
                if ylim:
                    ax.set_ylim(ylim)

                _set_discrete_xticks(ax, sub[x_col].values,
                                     log_x=log_x, xtick_fmt=xtick_fmt)

                parts = []
                if rv is not None:
                    parts.append(f"{row_var}={_fmt_val(rv)}")
                if cv is not None:
                    parts.append(f"{col_var}={_fmt_val(cv)}")
                ax.set_title(", ".join(parts), fontsize=8)
                ax.grid(True, alpha=0.3)

                if ri == nR - 1:
                    ax.set_xlabel(_col_label(x_var), fontsize=8)
                if ax_col == gi * nC_sub:
                    ax.set_ylabel(y_label_global or "", fontsize=8)

    # ── 4. Initial layout pass ────────────────────────────────────────────────
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.canvas.draw()

    # ── 5. Inject extra horizontal space between groups ───────────────────────
    if group_spacing > 0 and n_groups > 1:
        for gi in range(n_groups):
            shift = gi * group_spacing
            for ri in range(nR):
                for ci in range(nC_sub):
                    ax  = axes[ri][gi * nC_sub + ci]
                    pos = ax.get_position()
                    ax.set_position([
                        pos.x0 + shift, pos.y0,
                        pos.width, pos.height,
                    ])

        all_axes = [axes[ri][ci]
                    for ri in range(nR) for ci in range(nC_total)]
        max_x1   = max(ax.get_position().x1 for ax in all_axes)
        min_x0   = min(ax.get_position().x0 for ax in all_axes)
        avail    = 0.97 - min_x0
        scale    = avail / (max_x1 - min_x0)

        for ax in all_axes:
            pos = ax.get_position()
            ax.set_position([
                min_x0 + (pos.x0 - min_x0) * scale,
                pos.y0,
                pos.width * scale,
                pos.height,
            ])

        fig.canvas.draw()

    # ── 6. Suptitle ───────────────────────────────────────────────────────────
    if title:
        fig.suptitle(title, fontsize=12, y=title_y, fontweight="bold")

    # ── 7. Group header labels & separators ───────────────────────────────────
    for gi, gv in enumerate(grp_vals):
        mid_ci  = gi * nC_sub + nC_sub // 2
        mid_pos = axes[0][mid_ci].get_position()
        x_label = (mid_pos.x0 + mid_pos.x1) / 2

        fig.text(
            x_label, group_label_y,
            f"{group_var} = {_fmt_val(gv)}",
            ha="center", va="bottom",
            fontsize=group_label_fontsize,
            fontweight="bold",
            transform=fig.transFigure,
        )

        if group_separator and gi < n_groups - 1:
            right_pos = axes[0][gi * nC_sub + nC_sub - 1].get_position()
            next_pos  = axes[0][(gi + 1) * nC_sub].get_position()
            x_sep     = (right_pos.x1 + next_pos.x0) / 2
            fig.add_artist(plt.Line2D(
                [x_sep, x_sep], [0.03, 0.96],
                transform=fig.transFigure,
                color=group_separator_color,
                linewidth=1.2,
                linestyle="--",
                zorder=0,
            ))

    # ── 8. Shared legend at the bottom ────────────────────────────────────────
    if legend:
        handles, labels_leg = _collect_legend_handles(fig)
        if handles:
            ncol = legend_ncol if legend_ncol is not None else len(labels_leg)
            fig.legend(
                handles, labels_leg,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=ncol,
                fontsize=legend_fontsize,
                frameon=True,
            )

    # ── 9. Save / display / close ─────────────────────────────────────────────
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
    if show:
        from IPython.display import display as _ipy_display
        _ipy_display(fig)
    plt.close(fig)
    plt.ion()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 6.  PRESETS
# ──────────────────────────────────────────────────────────────────────────────

def plot_omp_time(df, row_var="N", col_var="D", fixed=None, **kw):
    return plot_grid(df, "time", x_var="P", row_var=row_var, col_var=col_var,
                     backend="omp", fixed=fixed,
                     title="OMP - Execution Time", **kw)

def plot_omp_speedup(df, row_var="N", col_var="D", fixed=None, **kw):
    return plot_grid(df, "speedup", x_var="P", row_var=row_var, col_var=col_var,
                     backend="omp", fixed=fixed, ideal_line="speedup",
                     title="OMP - Strong Scaling Speedup", **kw)

def plot_omp_strong_efficiency(df, row_var="N", col_var="D", fixed=None, **kw):
    return plot_grid(df, "strong_eff", x_var="P", row_var=row_var, col_var=col_var,
                     backend="omp", fixed=fixed, ideal_line="efficiency",
                     title="OMP - Strong Scaling Efficiency", **kw)

def plot_omp_weak_efficiency(df, row_var="N", col_var="D", fixed=None, **kw):
    return plot_grid(df, "weak_eff", x_var="P", row_var=row_var, col_var=col_var,
                     backend="omp", fixed=fixed, ideal_line="efficiency",
                     title="OMP - Weak Scaling Efficiency", **kw)

def plot_cuda_time(df, row_var="N", col_var="D", fixed=None, **kw):
    return plot_grid(df, "time", x_var="B", row_var=row_var, col_var=col_var,
                     backend="cuda", fixed=fixed,
                     title="CUDA - Execution Time", **kw)

def plot_cuda_throughput(df, row_var="N", col_var="D", fixed=None, **kw):
    return plot_grid(df, "throughput", x_var="B", row_var=row_var, col_var=col_var,
                     backend="cuda", fixed=fixed,
                     title="CUDA - Throughput", **kw)

def plot_omp_comparison(dfs: dict, metric="speedup", row_var="N", col_var="D",
                        fixed=None, **kw):
    """Confronta più varianti OMP.  dfs: {"OMP base": df_base, "OMP opt": df_opt}"""
    return plot_experiments(
        dfs, metric,
        x_var="P", row_var=row_var, col_var=col_var, fixed=fixed,
        ideal_line=("speedup" if metric == "speedup" else
                    "efficiency" if "eff" in metric else None),
        title=f"OMP Comparison - {metric}", **kw
    )

def plot_cuda_comparison(dfs: dict, metric="throughput", row_var="N", col_var="D",
                         fixed=None, **kw):
    """Confronta più varianti CUDA.  dfs: {"CUDA v1": df1, "CUDA v2": df2}"""
    experiments = {label: (df, "cuda") for label, df in dfs.items()}
    return plot_experiments(
        experiments, metric,
        x_var="B", row_var=row_var, col_var=col_var, fixed=fixed,
        title=f"CUDA Comparison - {metric}", **kw
    )


# ──────────────────────────────────────────────────────────────────────────────
# 7.  PLOT NOTEVOLI  (suggested analysis plots)
# ──────────────────────────────────────────────────────────────────────────────

def plot_notable_omp_vs_cuda_time(df_omp, df_cuda,
                                  filter_vals=None, fixed=None, **kw):
    """[NOTEVOLE] Confronto diretto OMP vs CUDA sul tempo totale."""
    return plot_experiments(
        {"OMP": (df_omp, "omp"), "CUDA": (df_cuda, "cuda")},
        "time",
        x_var="N",
        row_var="D", col_var=None,
        tasks=["tot"],
        fixed=fixed,
        filter_vals=filter_vals,
        errorbar="bar",
        log_y=True,
        title="OMP vs CUDA – Execution Time (log scale)",
        **kw
    )

def plot_notable_speedup_vs_N(df_omp, filter_vals=None, fixed=None, **kw):
    """[NOTEVOLE] Speedup OMP al variare di N, pannelli per D."""
    return plot_grid(
        df_omp, "speedup",
        x_var="P",
        row_var="N", col_var="D",
        backend="omp",
        tasks=["tot"],
        fixed=fixed,
        filter_vals=filter_vals,
        ideal_line="speedup",
        errorbar="bar",
        title="OMP Speedup vs Threads – panels by N and D",
        **kw
    )

def plot_notable_efficiency_heatmap_style(df_omp, metric="strong_eff",
                                          filter_vals=None, fixed=None, **kw):
    """[NOTEVOLE] Efficienza OMP con layout N x D come heatmap testuale."""
    label = {
        "strong_eff": "Strong Efficiency",
        "weak_eff":   "Weak Efficiency",
    }.get(metric, metric)
    return plot_grid(
        df_omp, metric,
        x_var="P",
        row_var="N", col_var="D",
        backend="omp",
        tasks=["classify", "update", "tot"],
        fixed=fixed,
        filter_vals=filter_vals,
        ideal_line="efficiency",
        errorbar="fill",
        sharey="all",
        title=f"OMP {label} – N×D grid",
        **kw
    )

def plot_notable_cuda_block_sweep(df_cuda, filter_vals=None, fixed=None, **kw):
    """[NOTEVOLE] CUDA throughput al variare del block_dim, pannelli per N."""
    return plot_grid(
        df_cuda, "throughput",
        x_var="B",
        row_var="N", col_var="D",
        backend="cuda",
        tasks=["classify", "update", "tot"],
        fixed=fixed,
        filter_vals=filter_vals,
        errorbar="bar",
        log_x=True,
        xtick_fmt="pow2",
        title="CUDA Throughput vs Block Dim – panels by N and D",
        **kw
    )

def plot_notable_omp_base_vs_opt(df_base, df_opt,
                                  metric="speedup", tasks=["tot"],
                                  filter_vals=None, fixed=None, **kw):
    """[NOTEVOLE] Confronto OMP base vs OMP ottimizzato (O3 + SIMD)."""
    ideal = ("speedup" if metric == "speedup" else
             "efficiency" if "eff" in metric else None)
    return plot_experiments(
        {"OMP base": (df_base, "omp"), "OMP O3+SIMD": (df_opt, "omp")},
        metric,
        x_var="P",
        row_var="N", col_var="D",
        tasks=tasks,
        fixed=fixed,
        filter_vals=filter_vals,
        ideal_line=ideal,
        errorbar="bar",
        title=f"OMP base vs O3+SIMD – {metric}",
        **kw
    )