"""Shared data loading for the GNN training scripts. Loads the cached
train/dev/test citation graphs (pairwise node-disjoint, directed) and
builds per-node features: [SPECTER2 embedding || log1p(in-degree) ||
log1p(out-degree) || hop flag]."""
import os
import pickle

import numpy as np
import torch
from torch_geometric.data import Data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "..", "graphs", "cache")
GRAPH_PKL = os.path.join(CACHE_DIR, "final_graphs.pkl")
EMBEDDING_DIM = 768  # SPECTER2


class _NumpyCoreCompatUnpickler(pickle.Unpickler):
    """Rewrites the numpy._core module path to numpy.core so pickles saved
    under numpy>=2 load under numpy<2."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_split_graphs(path: str = GRAPH_PKL) -> dict:
    """Returns {"train": igraph.Graph, "dev": igraph.Graph, "test": igraph.Graph},
    each a directed igraph.Graph, already enriched with node features
    (embedding, year, venue, domain, citation/reference counts, ...)."""
    with open(path, "rb") as f:
        return _NumpyCoreCompatUnpickler(f).load()


def build_structural_node_features(g) -> np.ndarray:
    """[log1p(in-degree), log1p(out-degree), hop flag] per node."""
    indeg = np.array(g.indegree(), dtype=np.float32)
    outdeg = np.array(g.outdegree(), dtype=np.float32)
    hop = np.array([0.0 if h in (None, 0) else 1.0 for h in g.vs["hop"]], dtype=np.float32)
    return np.stack([np.log1p(indeg), np.log1p(outdeg), hop], axis=1)


def build_content_node_features(g) -> np.ndarray:
    """[SPECTER2 embedding] per node, zero vector where missing."""
    zero = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    return np.stack(
        [e if e is not None else zero for e in g.vs["embedding"]], axis=0
    ).astype(np.float32)


def build_node_features(g) -> np.ndarray:
    """Concatenates the SPECTER2 content features with the structural features."""
    content = build_content_node_features(g)
    struct = build_structural_node_features(g)
    return np.concatenate([content, struct], axis=1)


def igraph_to_pyg_data(g) -> Data:
    """Converts one split's directed igraph.Graph into a PyG Data object:
    node features (see build_node_features) + the full directed edge_index
    (every citation edge, regardless of intent label). Parallel edges (a
    paper can cite another in more than one context) are collapsed, since
    link existence only cares whether u -> v holds at all.
    """
    x = torch.tensor(build_node_features(g), dtype=torch.float)
    sources = [e.source for e in g.es]
    targets = [e.target for e in g.es]
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)
    return Data(x=x, edge_index=edge_index, num_nodes=g.vcount())

