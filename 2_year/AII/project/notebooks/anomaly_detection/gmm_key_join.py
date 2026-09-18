"""
notebooks/anomaly_detection/gmm_key_join.py
Same direct RID + nearest-EXAMDATE_DX join as final_key_join.py, applied to
gmm_hi_trajectories.csv instead of pca_hi_trajectories.csv, so GMM's HI can
be plugged into the pipeline the same way PCA's and the autoencoder's already are.
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets")
GMM_HI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "method_b_autoencoder_hi", "gmm_hi_trajectories.csv")
WINDOW_DAYS = 180

def main():
    final = pd.read_csv(f"{DATA_DIR}/final.csv", low_memory=False)
    final["EXAMDATE_DX"] = pd.to_datetime(final["EXAMDATE_DX"], errors="coerce")
    final_small = final[["RID", "VISCODE2_norm", "EXAMDATE_DX"]].dropna(subset=["EXAMDATE_DX"])

    hi = pd.read_csv(GMM_HI_PATH)
    hi["EXAMDATE"] = pd.to_datetime(hi["EXAMDATE"], errors="coerce")

    print(f"gmm_hi_trajectories.csv rows: {len(hi)}  ({hi['RID'].nunique()} patients)")

    cand = hi.merge(final_small, on="RID", how="inner")
    cand["_gap"] = (cand["EXAMDATE"] - cand["EXAMDATE_DX"]).dt.days.abs()
    cand = cand[cand["_gap"] <= WINDOW_DAYS]
    cand = cand.sort_values("_gap").drop_duplicates(["RID", "EXAMDATE"], keep="first")

    keyed = hi.merge(
        cand[["RID", "EXAMDATE", "VISCODE2_norm", "_gap"]].rename(columns={"_gap": "MATCH_GAP_DAYS"}),
        on=["RID", "EXAMDATE"], how="left",
    )
    coverage = keyed["VISCODE2_norm"].notna().mean()
    print(f"Keyed coverage: {coverage:.1%} ({keyed['VISCODE2_norm'].notna().sum()}/{len(keyed)})")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "method_b_autoencoder_hi", "gmm_hi_trajectories_keyed.csv")
    keyed.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
