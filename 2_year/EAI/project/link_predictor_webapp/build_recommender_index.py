"""One-time offline step for the link-prediction recommender app: loads the
trained GraphSAGE model + test-split citation graph, computes a 256-dim
embedding per paper, and resolves metadata from the local Semantic Scholar
cache, so the Flask app never needs torch or the full graph at request time.

Produces:
    recommender_embeddings.npz -- embeddings (float32 [N,256], GraphSAGE
        encoder output) and content_embeddings (float16 [N,768], raw
        SPECTER2, pre-graph-mixing -- a purer topical-relevance signal,
        no citation-degree/popularity signal at all), plus paper_ids/
        titles/authors/venues/years.
    recommender_predictor.npz -- the trained LinkPredictor MLP's weights.
    recommender_explainer.npz -- a shallow CART surrogate tree (SKE, see
        L4-Transparency) approximating the model's predicted probability
        from five interpretable features, plus its held-out fidelity.

Needs enough RAM to load final_graphs.pkl (~3.6GB on disk); run with
`python build_recommender_index.py` from this folder. Uses the "test"
split, matching second_hand_citation_webapp's test_graph_index.json, so
paper_ids line up between the two apps."""
import json
import os
import pickle
import random
import sqlite3
import sys
import time

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from data_utils import load_split_graphs, igraph_to_pyg_data, EMBEDDING_DIM  # noqa: E402
from train_graphsage import GraphSAGELinkPredictor  # noqa: E402

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "GNNs", "weights", "graphsage_link_predictor_es_ap_final.pt")
SQLITE_PATH = os.path.join(PROJECT_ROOT, "graphs", "cache", "detail_cache_compact.sqlite3")
OUT_EMB_PATH = os.path.join(BASE_DIR, "data", "recommender_embeddings.npz")
OUT_PREDICTOR_PATH = os.path.join(BASE_DIR, "data", "recommender_predictor.npz")
OUT_EXPLAINER_PATH = os.path.join(BASE_DIR, "data", "recommender_explainer.npz")
OUT_EXAMPLES_PATH = os.path.join(BASE_DIR, "data", "recommender_examples.json")

EXAMPLES_N_QUERY_SAMPLE = 3000  # how many query papers to test for a strong candidate
EXAMPLES_MIN_DEGREE = 2  # skip near-isolated query papers -- they have no structural candidates anyway
EXAMPLES_CANDIDATE_CAP = 300  # cap the 2-hop candidate pool per query, for speed
EXAMPLES_MIN_PROBABILITY = 0.6  # a query only qualifies as a "good example" above this
EXAMPLES_TOP_N = 30  # how many curated examples to keep, ranked by their best candidate's probability
EXAMPLES_SEED = 0

SPLIT = "test"
HIDDEN_DIM = 256  # matches the checkpoint's actual shapes, not the class default of 128
NUM_LAYERS = 2
SQLITE_CHUNK_SIZE = 900  # SQLite has a default limit around 999 bound parameters per query

# Must match app.py's EXPLAINER_FEATURE_NAMES exactly, in the same order --
# the exported tree's "feature" array is a list of column INDICES into
# whatever order build_explainer() builds its X matrix in below, and app.py
# has to reproduce that same order at request time (compute_features()) to
# walk the tree correctly. topical_relevance is last (index 4) so an older
# 4-feature tree still works unmodified if you're re-reading one built
# before this field existed.
EXPLAINER_FEATURE_NAMES = ["cosine_similarity", "shared_neighbor_count", "year_gap", "candidate_degree", "topical_relevance"]
EXPLAINER_N_POS = 4000  # sampled real edges
EXPLAINER_N_NEG = 4000  # sampled random non-edges
EXPLAINER_TREE_MAX_DEPTH = 4
EXPLAINER_MIN_SAMPLES_LEAF = 50
EXPLAINER_SEED = 0


def build_explainer(g, h, content_h, model):
    """Fits a shallow CART decision tree approximating the trained model's
    predicted link probability from five interpretable features (SKE).
    Trained on a sample of real edges (positives) and random non-edges
    (negatives), scored with the real model directly (not the numpy
    re-implementation). content_h is the raw SPECTER2 content embedding,
    used only for topical_relevance -- kept separate from h so the tree can
    distinguish "the model is confident" from "actually the same topic"."""
    rng = random.Random(EXPLAINER_SEED)
    n = g.vcount()
    edges = g.get_edgelist()
    edge_set = set(edges)

    pos_sample = rng.sample(edges, min(EXPLAINER_N_POS, len(edges)))
    neg_sample = []
    while len(neg_sample) < EXPLAINER_N_NEG:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v and (u, v) not in edge_set:
            neg_sample.append((u, v))
    pairs = pos_sample + neg_sample
    rng.shuffle(pairs)
    print(f"  sampled {len(pos_sample)} real edges + {len(neg_sample)} random non-edges for the surrogate")

    years = np.array(g.vs["year"], dtype=float)
    median_year = np.nanmedian(years) if np.any(~np.isnan(years)) else 0.0
    years = np.where(np.isnan(years), median_year, years)
    degrees = np.array(g.degree(mode="all"), dtype=float)

    needed_nodes = {idx for pair in pairs for idx in pair}
    neighbor_sets = {idx: set(g.neighbors(idx, mode="all")) for idx in needed_nodes}

    us = np.array([u for u, v in pairs])
    vs = np.array([v for u, v in pairs])

    norms = np.linalg.norm(h, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    norm_h = h / norms
    cosine_sim = np.sum(norm_h[us] * norm_h[vs], axis=1)
    shared_neighbor_count = np.array([len(neighbor_sets[u] & neighbor_sets[v]) for u, v in pairs], dtype=float)
    year_gap = np.abs(years[us] - years[vs])
    candidate_degree = degrees[vs]

    content_h32 = content_h.astype(np.float32)
    content_norms = np.linalg.norm(content_h32, axis=1, keepdims=True)
    content_norms[content_norms == 0] = 1e-8
    norm_content_h = content_h32 / content_norms
    topical_relevance = np.sum(norm_content_h[us] * norm_content_h[vs], axis=1)

    X = np.column_stack([cosine_sim, shared_neighbor_count, year_gap, candidate_degree, topical_relevance])

    with torch.no_grad():
        h_t = torch.from_numpy(h)
        edge_label_index = torch.stack([torch.from_numpy(us), torch.from_numpy(vs)])
        logits = model.predictor(h_t, edge_label_index).reshape(-1)
        y = torch.sigmoid(logits).numpy()

    n_total = len(pairs)
    n_train = int(n_total * 0.8)
    X_train, X_held = X[:n_train], X[n_train:]
    y_train, y_held = y[:n_train], y[n_train:]

    tree = DecisionTreeRegressor(
        max_depth=EXPLAINER_TREE_MAX_DEPTH,
        min_samples_leaf=EXPLAINER_MIN_SAMPLES_LEAF,
        random_state=EXPLAINER_SEED,
    )
    tree.fit(X_train, y_train)
    fidelity_r2 = r2_score(y_held, tree.predict(X_held))
    print(f"  surrogate decision tree fidelity (R^2 vs real model, held-out): {fidelity_r2:.3f}")

    t = tree.tree_
    np.savez(
        OUT_EXPLAINER_PATH,
        feature=t.feature.astype(np.int64),
        threshold=t.threshold.astype(np.float64),
        children_left=t.children_left.astype(np.int64),
        children_right=t.children_right.astype(np.int64),
        value=t.value.reshape(-1).astype(np.float64),
        feature_names=np.array(EXPLAINER_FEATURE_NAMES, dtype=object),
        fidelity_r2=np.array([fidelity_r2], dtype=np.float64),
    )
    print(f"Saved explainer to {OUT_EXPLAINER_PATH}")


def fetch_metadata(paper_ids):
    """Batched sqlite lookup of title/authors/venue/year/citation-count for
    every paper_id, chunked to stay under sqlite's bound-parameter limit.
    Returns five arrays (titles, authors_str, venues, years,
    citation_counts), index-aligned with paper_ids, with empty/NaN
    placeholders for anything not found. citation_counts is the paper's
    real-world Semantic Scholar count, distinct from this app's own
    citation-graph degree (edges present in the indexed test subgraph)."""
    n = len(paper_ids)
    titles = np.empty(n, dtype=object)
    authors = np.empty(n, dtype=object)
    venues = np.empty(n, dtype=object)
    years = np.full(n, np.nan, dtype=float)
    citation_counts = np.full(n, np.nan, dtype=float)
    titles[:], authors[:], venues[:] = "", "", ""

    if not os.path.exists(SQLITE_PATH):
        print(f"WARNING: {SQLITE_PATH} not found -- all titles will be blank.")
        return titles, authors, venues, years, citation_counts

    conn = sqlite3.connect(SQLITE_PATH)
    id_to_pos = {pid: i for i, pid in enumerate(paper_ids)}
    resolved = 0
    for start in range(0, n, SQLITE_CHUNK_SIZE):
        chunk = paper_ids[start:start + SQLITE_CHUNK_SIZE]
        placeholders = ",".join("?" for _ in chunk)
        cur = conn.execute(f"SELECT paper_id, payload FROM details WHERE paper_id IN ({placeholders})", chunk)
        for pid, blob in cur.fetchall():
            pos = id_to_pos.get(pid)
            if pos is None:
                continue
            try:
                payload = pickle.loads(blob)
            except Exception:
                continue
            if payload.get("title"):
                titles[pos] = payload.get("title") or ""
                authors[pos] = ", ".join(a.get("name", "") for a in (payload.get("authors") or []) if a.get("name"))
                venues[pos] = payload.get("venue") or ""
                y = payload.get("year")
                if y:
                    years[pos] = float(y)
                cc = payload.get("citationCount")
                if cc is not None:
                    citation_counts[pos] = float(cc)
                resolved += 1
        if start % (SQLITE_CHUNK_SIZE * 20) == 0:
            print(f"  metadata lookup: {start + len(chunk)}/{n} checked, {resolved} resolved so far")
    conn.close()
    print(f"Metadata resolved for {resolved}/{n} papers.")
    return titles, authors, venues, years, citation_counts


def build_examples(g, h, model, paper_ids):
    """Curates query papers with a confident, structurally-grounded predicted
    citation link, for the UI's example list. Candidates are drawn only from
    each query's own 2-hop structural neighborhood, both cheaper and more
    likely to score highly than an arbitrary distant candidate."""
    rng = random.Random(EXAMPLES_SEED)
    n = g.vcount()
    eligible = [i for i in range(n) if g.degree(i, mode="all") >= EXAMPLES_MIN_DEGREE]
    sample_nodes = rng.sample(eligible, min(EXAMPLES_N_QUERY_SAMPLE, len(eligible)))
    print(f"  scanning {len(sample_nodes)} candidate query papers for a confident predicted link...")

    found = []
    with torch.no_grad():
        h_t = torch.from_numpy(h)
        for q in sample_nodes:
            direct = set(g.neighbors(q, mode="all"))
            two_hop = set()
            for nb in direct:
                two_hop.update(g.neighbors(nb, mode="all"))
            candidates = list((two_hop - direct) - {q})
            if not candidates:
                continue
            if len(candidates) > EXAMPLES_CANDIDATE_CAP:
                candidates = rng.sample(candidates, EXAMPLES_CANDIDATE_CAP)

            us = torch.full((len(candidates),), q, dtype=torch.long)
            vs = torch.tensor(candidates, dtype=torch.long)
            logits = model.predictor(h_t, torch.stack([us, vs])).reshape(-1)
            probs = torch.sigmoid(logits).numpy()
            best_pos = int(np.argmax(probs))
            best_prob = float(probs[best_pos])
            if best_prob >= EXAMPLES_MIN_PROBABILITY:
                found.append((q, candidates[best_pos], best_prob))

    found.sort(key=lambda r: -r[2])
    top = found[:EXAMPLES_TOP_N]
    print(f"  found {len(found)} query papers with a confident (>={EXAMPLES_MIN_PROBABILITY}) predicted link, keeping top {len(top)}")

    examples = [
        {
            "paper_id": paper_ids[q],
            "best_candidate_id": paper_ids[c],
            "best_predicted_probability": round(p, 4),
        }
        for q, c, p in top
    ]
    with open(OUT_EXAMPLES_PATH, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {len(examples)} curated examples to {OUT_EXAMPLES_PATH}")


def main():
    t0 = time.time()
    print(f"Loading {SPLIT} split from final_graphs.pkl (this can take a while and several GB of RAM)...")
    graphs = load_split_graphs()
    g = graphs[SPLIT]
    print(f"  {g.vcount()} nodes, {g.ecount()} edges ({time.time() - t0:.1f}s)")

    print("Converting to PyG Data...")
    data = igraph_to_pyg_data(g)
    in_dim = data.x.size(-1)
    print(f"  in_dim={in_dim}")

    print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
    model = GraphSAGELinkPredictor(in_dim=in_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print("Running encoder to get node embeddings...")
    with torch.no_grad():
        h = model.encoder(data.x, data.edge_index).numpy().astype(np.float32)
    print(f"  embeddings shape: {h.shape} ({time.time() - t0:.1f}s elapsed)")

    # The raw SPECTER2 content embedding is just the first EMBEDDING_DIM
    # columns of the encoder's own input (build_node_features() concatenates
    # [content || structural] -- see data_utils.py), so no extra graph work
    # is needed to pull it out. float16 keeps the export size reasonable;
    # a relevance cosine similarity doesn't need float32 precision.
    content_h = data.x[:, :EMBEDDING_DIM].numpy().astype(np.float16)
    print(f"  content (SPECTER2) embeddings shape: {content_h.shape}")

    paper_ids = list(g.vs["paper_id"])
    assert len(paper_ids) == h.shape[0], "paper_id count doesn't match embedding row count"

    print("Resolving titles/authors/venue/year/citation-count from the local metadata cache...")
    titles, authors, venues, years, citation_counts = fetch_metadata(paper_ids)

    os.makedirs(os.path.dirname(OUT_EMB_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_EMB_PATH,
        embeddings=h,
        content_embeddings=content_h,
        paper_ids=np.array(paper_ids, dtype=object),
        titles=titles,
        authors=authors,
        venues=venues,
        years=years,
        citation_counts=citation_counts,
    )
    print(f"Saved embeddings + metadata to {OUT_EMB_PATH} ({os.path.getsize(OUT_EMB_PATH) / 1e6:.1f} MB)")

    predictor_sd = model.predictor.state_dict()
    # LinkPredictor.mlp = Sequential(Linear(2*hidden,hidden), ReLU, Dropout, Linear(hidden,1))
    # -> indices 0 and 3 are the two Linear layers.
    W0 = predictor_sd["mlp.0.weight"].numpy().astype(np.float32)  # (hidden, 2*hidden)
    b0 = predictor_sd["mlp.0.bias"].numpy().astype(np.float32)    # (hidden,)
    W3 = predictor_sd["mlp.3.weight"].numpy().astype(np.float32)  # (1, hidden)
    b3 = predictor_sd["mlp.3.bias"].numpy().astype(np.float32)    # (1,)
    np.savez(OUT_PREDICTOR_PATH, W0=W0, b0=b0, W3=W3, b3=b3)
    print(f"Saved predictor weights to {OUT_PREDICTOR_PATH}")

    print("Fitting the decision-tree surrogate explainer (SKE)...")
    build_explainer(g, h, content_h, model)

    print("Curating example papers with a confident predicted link...")
    build_examples(g, h, model, paper_ids)

    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
