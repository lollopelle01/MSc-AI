"""Graph-free link-existence baseline, "encoded" variant: a per-node MLP
transforms each paper's raw features independently of its neighbours (no
edge_index), then the decoder scores [h_u || h_v]. Middle rung of the
ablation ladder (raw -> encoded -> GraphSAGE); see mlp_baseline_common.py."""
from mlp_baseline_common import main_for_variant

if __name__ == "__main__":
    main_for_variant("encoded", default_out="mlp_baseline_encoded_link_predictor.pt")
