"""Graph-free link-existence baseline, "raw" variant, retrained with hard
2-hop negatives + a BPR ranking loss (train_graphsage_hard.py's protocol).
Not comparable against train_mlp_baseline_raw.py's random-negative
results -- different, harder task."""
from mlp_baseline_hard_common import main_for_variant

if __name__ == "__main__":
    main_for_variant("raw", default_out="mlp_baseline_raw_link_predictor_hardneg.pt")
