"""One-time preprocessing step for the "view citation graph" feature:
extracts a lightweight adjacency index (paper_id, year, in/out edges only,
no embeddings) from graphs/cache/final_graphs.pkl and writes it to
webapp/data/test_graph_index.json, so the Flask app doesn't need to load
the full ~3.6GB pickle at request time. Only the "test" split is indexed,
matching second_hand_citation_test.csv. Run with `python
build_graph_index.py` from this folder."""
import json
import os
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
GRAPH_PKL = os.path.join(PROJECT_ROOT, "graphs", "cache", "final_graphs.pkl")

OUT_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_PATH = os.path.join(OUT_DIR, "test_graph_index.json")


class _NumpyCoreCompatUnpickler(pickle.Unpickler):
    """Same shim as scripts/data_utils.py: rewrites numpy._core to
    numpy.core so a numpy>=2 pickle loads under numpy<2. Duplicated here
    (not imported) so this script doesn't need torch installed."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_split_graphs(path: str = GRAPH_PKL) -> dict:
    with open(path, "rb") as f:
        return _NumpyCoreCompatUnpickler(f).load()


def main():
    print("Loading graphs/cache/final_graphs.pkl (this needs real memory -- "
          "several GB -- and may take a few minutes)...")
    graphs = load_split_graphs()
    g = graphs["test"]
    print(f"Loaded test split: {g.vcount()} nodes, {g.ecount()} edges")

    paper_ids = g.vs["paper_id"]
    years = g.vs["year"]

    # Adjacency as index-based lists first (cheap), then remapped to
    # paper_id strings for the JSON output.
    out_edges = [[] for _ in range(g.vcount())]  # this node cites ->
    in_edges = [[] for _ in range(g.vcount())]   # this node is cited by <-
    seen = set()
    for e in g.es:
        key = (e.source, e.target)
        if key in seen:  # collapse parallel edges (same as igraph_to_pyg_data)
            continue
        seen.add(key)
        out_edges[e.source].append(e.target)
        in_edges[e.target].append(e.source)

    nodes = {}
    for i in range(g.vcount()):
        pid = paper_ids[i]
        nodes[pid] = {
            "year": years[i],
            "out": [paper_ids[j] for j in out_edges[i]],
            "in": [paper_ids[j] for j in in_edges[i]],
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"split": "test", "node_count": g.vcount(), "edge_count": len(seen), "nodes": nodes}, f)

    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Wrote {OUT_PATH} ({size_mb:.1f} MB, {len(nodes)} nodes)")


if __name__ == "__main__":
    main()
