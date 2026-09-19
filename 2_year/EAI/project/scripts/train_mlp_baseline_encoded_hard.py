"""Graph-free link-existence baseline, "encoded" variant, retrained with
hard 2-hop negatives + a BPR ranking loss (train_graphsage_hard.py's
protocol). Not comparable against train_mlp_baseline_encoded.py's
random-negative results -- different, harder task."""
from mlp_baseline_hard_common import main_for_variant

if __name__ == "__main__":
    main_for_variant("encoded", default_out="mlp_baseline_encoded_link_predictor_hardneg.pt")
