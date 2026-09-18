import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compress_dfs(dfs):
    '''
    This aggregate each experiments to extract the mean and standard deviation of executio times
    '''
    
    rows = []
    for df in dfs:
        row = df.iloc[0, :-3].to_dict()
        for t_col in ["t_classify", "t_update", "t_tot"]:
            row[t_col + "_mean"] = df[t_col].mean()
            row[t_col + "_std"] = df[t_col].std()
        rows.append(row)
    return pd.DataFrame(rows)

def error_propagation(val_a, std_a, val_b, std_b, op="ratio"):
    """
    This computes the standard deviation propagation in a ratio or a product between 2 values:
        
        (val_a +/- std_a)  op  (val_b +/- std_b)  =  f +/- sigma_f
        
    Returns (nan, nan) wherever the operation is undefined (e.g. division by zero).
    """
    val_a = np.asarray(val_a, dtype=float)
    val_b = np.asarray(val_b, dtype=float)
    std_a = np.asarray(std_a, dtype=float)
    std_b = np.asarray(std_b, dtype=float)

    if op == "ratio":
        # Se il denominatore è 0 → risultato indefinito → nan
        safe_b = np.where(val_b != 0, val_b, np.nan)
        f = val_a / safe_b
    elif op == "product":
        f = val_a * val_b
    else:
        raise ValueError(f"Operation not recognized: '{op}'. op='ratio' or op='product'.")

    rel_a = np.where(val_a != 0, std_a / np.where(val_a != 0, val_a, np.nan), 0.0)
    rel_b = np.where(val_b != 0, std_b / np.where(val_b != 0, val_b, np.nan), 0.0)
    sigma_f = np.abs(f) * np.sqrt(rel_a**2 + rel_b**2)

    return f, sigma_f

def compute_stats(df, df_type):
    df_result = df.copy()
    
    if df_type == "omp":
        baselines = df[df["threads"] == 1.0]
    
    for i, row in df.iterrows():
        
        if df_type == "cuda":
            n_operations_classify = row["n_points"] * row["n_dims"] * row["k"] * row["n_iter"]
            n_operations_update   = row["n_points"] * row["n_dims"]              * row["n_iter"]
            # n_operations_tot      = n_operations_classify + n_operations_update
            n_operations_tot      = n_operations_classify
            
            
            for n_op, task in [(n_operations_classify, "classify"),
                               (n_operations_update,   "update"),
                               (n_operations_tot,      "tot")]:
                mean, std = error_propagation(
                    n_op, 0,
                    row[f"t_{task}_mean"], row[f"t_{task}_std"],
                    op="ratio"
                )
                df_result.loc[i, f"throughput_{task}_mean"] = mean
                df_result.loc[i, f"throughput_{task}_std"]  = std
                
        elif df_type == "omp":
            # Strong scaling baseline: stessa dimensione del problema, threads=1
            strong_baseline = baselines[
                (baselines["n_points"] == row["n_points"]) &
                (baselines["n_dims"]   == row["n_dims"])   &
                (baselines["k"]        == row["k"])
            ]
            
            # Weak scaling baseline: stesso carico per thread (n_points/threads), threads=1
            points_per_thread = row["n_points"] / row["threads"]
            weak_baseline = baselines[
                (baselines["n_points"] == points_per_thread) &
                (baselines["n_dims"]   == row["n_dims"])     &
                (baselines["k"]        == row["k"])
            ]
            
            for task in ["classify", "update", "tot"]:
                # Speedup = T(1) / T(p)
                speedup_mean, speedup_std = error_propagation(
                    strong_baseline[f"t_{task}_mean"].values[0], strong_baseline[f"t_{task}_std"].values[0],
                    row[f"t_{task}_mean"], row[f"t_{task}_std"],
                    op="ratio"
                )
                df_result.loc[i, f"speedup_{task}_mean"] = speedup_mean
                df_result.loc[i, f"speedup_{task}_std"]  = speedup_std
                
                # Strong scaling efficiency = speedup / p
                strong_mean, strong_std = error_propagation(
                    speedup_mean, speedup_std,
                    row["threads"], 0,
                    op="ratio"
                )
                df_result.loc[i, f"strong_eff_{task}_mean"] = strong_mean
                df_result.loc[i, f"strong_eff_{task}_std"]  = strong_std
                
                # Weak scaling efficiency = T_serial(N/p) / T_parallel(N)
                if len(weak_baseline) > 0:
                    weak_mean, weak_std = error_propagation(
                        weak_baseline[f"t_{task}_mean"].values[0], weak_baseline[f"t_{task}_std"].values[0],
                        row[f"t_{task}_mean"], row[f"t_{task}_std"],
                        op="ratio"
                    )
                    df_result.loc[i, f"weak_eff_{task}_mean"] = weak_mean
                    df_result.loc[i, f"weak_eff_{task}_std"]  = weak_std
                else:
                    # Nessuna baseline con quel carico per thread → NaN
                    df_result.loc[i, f"weak_eff_{task}_mean"] = np.nan
                    df_result.loc[i, f"weak_eff_{task}_std"]  = np.nan
        else:
            raise ValueError(f"Type not recognized: '{df_type}'. df_type='cuda' or df_type='omp'.")
            
    return df_result