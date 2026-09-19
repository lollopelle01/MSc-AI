"""Graph-free link-existence baseline, "raw" variant: the decoder MLP
scores [x_u || x_v] directly from each paper's raw node features -- no
per-node transform, no message passing. Floor of the ablation ladder
(raw -> encoded -> GraphSAGE); see mlp_baseline_common.py."""
from mlp_baseline_common import main_for_variant

if __name__ == "__main__":
    main_for_variant("raw", default_out="mlp_baseline_raw_link_predictor.pt")
