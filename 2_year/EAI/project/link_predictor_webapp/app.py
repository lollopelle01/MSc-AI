"""
Link Predictor -- "papers you might want to cite" recommender.

Standalone Flask app (separate from second_hand_citation_webapp/) that surfaces the
trained GraphSAGE link predictor's own recommendations, with explanations, not just
a black-box ranked list.

Runs entirely on precomputed data (see build_recommender_index.py) -- plain numpy at
request time, no torch/final_graphs.pkl dependency.

Run
---
    pip install -r requirements.txt
    python app.py
    # then open http://localhost:5051
"""
import datetime
import json
import os

import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

RECOMMENDER_EMB_PATH = os.path.join(BASE_DIR, "data", "recommender_embeddings.npz")
RECOMMENDER_PREDICTOR_PATH = os.path.join(BASE_DIR, "data", "recommender_predictor.npz")
RECOMMENDER_EXPLAINER_PATH = os.path.join(BASE_DIR, "data", "recommender_explainer.npz")
RECOMMENDER_EXAMPLES_PATH = os.path.join(BASE_DIR, "data", "recommender_examples.json")

# NOTE: order must match build_recommender_index.py's build_explainer() exactly --
# the exported tree's "feature" array is column INDICES, not names. topical_relevance
# was added last (index 4) so a tree exported before it existed keeps working unmodified.
EXPLAINER_FEATURE_NAMES = ["cosine_similarity", "shared_neighbor_count", "year_gap", "candidate_degree", "topical_relevance"]
TREE_DISAGREEMENT_THRESHOLD = 0.15  # if |tree - real prob| exceeds this, the tree isn't trusted for this instance
MIN_TREE_FIDELITY = 0.3  # below this held-out R^2, the tree isn't trusted globally, ever
LIME_N_SAMPLES = 300
LIME_NOISE_STD = 0.15  # relative to the candidate embedding's own norm

# only candidates at/above this probability are drawn as a predicted edge
MIN_CONFIDENT_PREDICTED_PROBABILITY = 0.5
GRAPH_INDEX_PATH = os.path.join(PROJECT_ROOT, "second_hand_citation_webapp", "data", "test_graph_index.json")

SHORTLIST_SIZE = 300  # stage 1: how many candidates the cosine-similarity prefilter keeps
DEFAULT_TOP_K = 15  # stage 2: how many exact-scored candidates are returned by default
SEARCH_RESULT_LIMIT = 20
BUCKET_SIZE = 5  # how many candidates to surface per bucket in /api/explore
GRAPH_MAX_NEIGHBORS_PER_DIRECTION = 12  # real citation edges shown per direction around the query paper

GREATS_FRACTION = 1 / 3  # top fraction of the shortlist by citation-graph degree -> "all-time greats"
# smallest-first date windows for "promising" vs "hidden gem"; see assign_buckets()
RECENT_YEARS_WINDOWS = [3, 5, 7, 10, 15]
CURRENT_YEAR = datetime.date.today().year

app = Flask(__name__)

_recommender = None  # None = not yet attempted, False = attempted and unavailable, dict = loaded
_graph_index = None  # same tri-state pattern
_explainer = None  # same tri-state pattern
_examples = None  # same tri-state pattern


def get_recommender():
    """Lazily loads embeddings + metadata + trained LinkPredictor MLP weights.
    Returns False if the artifacts haven't been built yet."""
    global _recommender
    if _recommender is None:
        if not (os.path.exists(RECOMMENDER_EMB_PATH) and os.path.exists(RECOMMENDER_PREDICTOR_PATH)):
            _recommender = False
        else:
            emb_data = np.load(RECOMMENDER_EMB_PATH, allow_pickle=True)
            embeddings = emb_data["embeddings"]
            paper_ids = emb_data["paper_ids"]
            titles = emb_data["titles"]
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            pred_data = np.load(RECOMMENDER_PREDICTOR_PATH)
            # graceful fallback for indexes built before these fields existed
            if "citation_counts" in emb_data.files:
                citation_counts = emb_data["citation_counts"]
            else:
                citation_counts = np.full(len(paper_ids), np.nan)
            # content_embeddings (raw SPECTER2, pre-GraphSAGE) -- topical_relevance
            # comes back None if the index predates this field
            norm_content_embeddings = None
            if "content_embeddings" in emb_data.files:
                content_embeddings = emb_data["content_embeddings"].astype(np.float32)
                content_norms = np.linalg.norm(content_embeddings, axis=1, keepdims=True)
                content_norms[content_norms == 0] = 1.0
                norm_content_embeddings = content_embeddings / content_norms
            _recommender = {
                "id_to_idx": {pid: i for i, pid in enumerate(paper_ids)},
                "paper_ids": paper_ids,
                "titles": titles,
                "titles_lower": np.char.lower(titles.astype(str)),
                "authors": emb_data["authors"],
                "venues": emb_data["venues"],
                "years": emb_data["years"],
                "citation_counts": citation_counts,
                "embeddings": embeddings,
                "norm_embeddings": embeddings / norms,
                "norm_content_embeddings": norm_content_embeddings,  # None if not built yet
                "W0": pred_data["W0"], "b0": pred_data["b0"],
                "W3": pred_data["W3"], "b3": pred_data["b3"],
            }
    return _recommender


def get_graph_index():
    global _graph_index
    if _graph_index is None:
        if not os.path.exists(GRAPH_INDEX_PATH):
            _graph_index = False
        else:
            with open(GRAPH_INDEX_PATH) as f:
                _graph_index = json.load(f)
    return _graph_index


def get_explainer():
    """Lazily loads the decision-tree surrogate (SKE, see
    build_recommender_index.py's build_explainer()). Returns False if not
    built yet -- the app still works without it, just without the "why"."""
    global _explainer
    if _explainer is None:
        if not os.path.exists(RECOMMENDER_EXPLAINER_PATH):
            _explainer = False
        else:
            data = np.load(RECOMMENDER_EXPLAINER_PATH, allow_pickle=True)
            _explainer = {
                "feature": data["feature"],
                "threshold": data["threshold"],
                "children_left": data["children_left"],
                "children_right": data["children_right"],
                "value": data["value"],
                "feature_names": [str(n) for n in data["feature_names"]],
                "fidelity_r2": float(data["fidelity_r2"][0]),
            }
    return _explainer


def candidate_degrees(idxs, rec, graph_index):
    """Citation-graph degree (in+out) for each candidate index, 0 if the
    paper isn't in the citation-neighborhood index."""
    if not graph_index:
        return np.zeros(len(idxs))
    paper_ids = rec["paper_ids"]
    out = np.zeros(len(idxs))
    for i, idx in enumerate(idxs):
        pid = str(paper_ids[idx])
        node = graph_index["nodes"].get(pid)
        if node is not None:
            out[i] = len(node["out"]) + len(node["in"])
    return out


def assign_buckets(
    degrees,
    years,
    greats_fraction=GREATS_FRACTION,
    recent_years_windows=RECENT_YEARS_WINDOWS,
    current_year=CURRENT_YEAR,
):
    """Splits a shortlist of candidates into three buckets: all_time_greats
    (top fraction by citation-graph degree, ranked within this shortlist --
    the indexed subgraph is too sparse for a fixed global threshold), then
    promising_stars/hidden_gems by real publication date, widening
    recent_years_windows smallest-first until a window finds a candidate.
    Returns a list of bucket names, one per candidate."""
    n = len(degrees)
    buckets = [None] * n
    if n == 0:
        return buckets

    order_by_degree = np.argsort(-degrees, kind="stable")
    n_greats = max(1, round(n * greats_fraction)) if n >= 3 else max(1, n // 3)
    greats_positions = set(order_by_degree[:n_greats].tolist())
    for p in greats_positions:
        buckets[p] = "all_time_greats"

    remaining = [p for p in range(n) if p not in greats_positions]
    if remaining:
        remaining_years = years[remaining]
        filled_years = np.where(np.isnan(remaining_years), -np.inf, remaining_years)

        recent_mask = np.zeros(len(remaining), dtype=bool)
        for window in sorted(recent_years_windows):
            cutoff = current_year - window
            recent_mask = filled_years >= cutoff
            if recent_mask.any():
                break

        for p, is_recent in zip(remaining, recent_mask):
            buckets[p] = "promising_stars" if is_recent else "hidden_gems"

    return buckets


def get_examples():
    """Lazily loads the curated example papers (build_recommender_index.py's
    build_examples()). Returns False if not built yet."""
    global _examples
    if _examples is None:
        if not os.path.exists(RECOMMENDER_EXAMPLES_PATH):
            _examples = False
        else:
            with open(RECOMMENDER_EXAMPLES_PATH) as f:
                _examples = json.load(f)
    return _examples


def compute_features(q_idx, cand_idx, rec, graph_index, shared_neighbor_count):
    """The five interpretable features the tree/LIME explainer are built on:
    GraphSAGE cosine similarity, shared citation-neighbor count, year gap,
    candidate degree, and topical_relevance (raw SPECTER2 cosine similarity,
    no structural signal -- separates "model is confident" from "actually
    about the same thing"). topical_relevance is 0.0 if the index predates
    this field."""
    cosine_sim = float(rec["norm_embeddings"][q_idx] @ rec["norm_embeddings"][cand_idx])
    y_q, y_c = rec["years"][q_idx], rec["years"][cand_idx]
    year_gap = float(abs(y_q - y_c)) if not (np.isnan(y_q) or np.isnan(y_c)) else 0.0
    candidate_degree = 0.0
    pid = str(rec["paper_ids"][cand_idx])
    if graph_index and pid in graph_index["nodes"]:
        gnode = graph_index["nodes"][pid]
        candidate_degree = float(len(gnode["out"]) + len(gnode["in"]))
    topical_relevance = 0.0
    if rec["norm_content_embeddings"] is not None:
        topical_relevance = float(rec["norm_content_embeddings"][q_idx] @ rec["norm_content_embeddings"][cand_idx])
    return np.array([cosine_sim, float(shared_neighbor_count), year_gap, candidate_degree, topical_relevance])


def tree_predict_and_path(features, expl):
    """Walks the exported CART tree with plain numpy (no sklearn at request
    time), returning the tree's predicted probability and a human-readable
    decision path."""
    node = 0
    path = []
    names = expl["feature_names"]
    while expl["children_left"][node] != -1:
        f = int(expl["feature"][node])
        thr = float(expl["threshold"][node])
        name = names[f]
        val = float(features[f])
        if val <= thr:
            path.append({"feature": name, "op": "<=", "threshold": round(thr, 3), "actual": round(val, 3)})
            node = int(expl["children_left"][node])
        else:
            path.append({"feature": name, "op": ">", "threshold": round(thr, 3), "actual": round(val, 3)})
            node = int(expl["children_right"][node])
    return float(expl["value"][node]), path


def lime_explain(h_query, h_candidate, base_features, rec, n_samples=LIME_N_SAMPLES, noise_std=LIME_NOISE_STD, seed=0):
    """From-scratch LIME (Ribeiro et al. 2016): perturb the candidate
    embedding with Gaussian noise, score every perturbation with the REAL
    model (predictor_forward), then fit a local proximity-weighted linear
    surrogate on the same feature space the tree uses. Only cosine_similarity
    moves with the perturbation; the other four features are held at their
    real value -- a near-zero topical_relevance coefficient alongside a
    large cosine_similarity one means the score is driven by embedding
    closeness, not actual topical overlap."""
    rng = np.random.default_rng(seed)
    D = h_candidate.shape[0]
    scale = noise_std * (np.linalg.norm(h_candidate) + 1e-8)
    noise = rng.normal(0, 1, size=(n_samples, D)).astype(np.float32) * scale
    perturbed = h_candidate[None, :] + noise

    perturbed_probs = predictor_forward(h_query, perturbed, rec)

    q_norm = h_query / (np.linalg.norm(h_query) + 1e-8)
    cand_norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
    cand_norms[cand_norms == 0] = 1e-8
    cos_perturbed = (perturbed / cand_norms) @ q_norm

    n = n_samples
    # topical_relevance (base_features[4]) comes from the raw SPECTER2 content
    # embedding, not the perturbed GraphSAGE one -- held fixed like the rest.
    X = np.column_stack([
        cos_perturbed,
        np.full(n, base_features[1]),
        np.full(n, base_features[2]),
        np.full(n, base_features[3]),
        np.full(n, base_features[4]),
    ])
    dist = np.linalg.norm(noise, axis=1)
    weights = np.exp(-(dist ** 2) / (2 * scale ** 2 + 1e-8))

    Xd = np.column_stack([np.ones(n), X])
    XtWX = Xd.T @ (weights[:, None] * Xd)
    XtWy = Xd.T @ (weights * perturbed_probs)
    try:
        coefs = np.linalg.solve(XtWX + 1e-6 * np.eye(Xd.shape[1]), XtWy)
    except np.linalg.LinAlgError:
        coefs, *_ = np.linalg.lstsq(Xd, perturbed_probs, rcond=None)

    pred = Xd @ coefs
    ss_res = float(np.sum(weights * (perturbed_probs - pred) ** 2))
    weighted_mean = float(np.average(perturbed_probs, weights=weights))
    ss_tot = float(np.sum(weights * (perturbed_probs - weighted_mean) ** 2))
    local_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    names = ["intercept"] + EXPLAINER_FEATURE_NAMES
    return {
        "coefficients": {name: round(float(c), 4) for name, c in zip(names, coefs)},
        "local_fidelity_r2": round(local_r2, 3),
    }


def build_explanation(q_idx, cand_idx, h_query, h_candidate, real_probability, shared_neighbor_count, rec, graph_index):
    """Decides per-suggestion whether the global tree surrogate (SKE) is
    trustworthy for this instance, falling back to live LIME if not.
    Returns None if no explainer has been built."""
    expl = get_explainer()
    if not expl:
        return None

    features = compute_features(q_idx, cand_idx, rec, graph_index, shared_neighbor_count)
    tree_pred, path = tree_predict_and_path(features, expl)

    if expl["fidelity_r2"] >= MIN_TREE_FIDELITY and abs(tree_pred - real_probability) <= TREE_DISAGREEMENT_THRESHOLD:
        return {
            "method": "decision_tree",
            "tree_predicted_probability": round(tree_pred, 4),
            "path": path,
            "global_fidelity_r2": round(expl["fidelity_r2"], 3),
        }

    lime = lime_explain(h_query, h_candidate, features, rec)
    return {
        "method": "lime",
        "reason": (
            "the global decision-tree surrogate doesn't fit this particular prediction closely enough "
            f"(tree says {tree_pred:.3f}, model says {real_probability:.3f})"
            if expl["fidelity_r2"] >= MIN_TREE_FIDELITY
            else f"the decision tree's overall fidelity is too low to trust (R^2={expl['fidelity_r2']:.2f})"
        ),
        **lime,
    }


def predictor_forward(h_query, H_candidates, rec):
    """Vectorized numpy re-implementation of the trained LinkPredictor's
    forward pass: concat([h_u, h_v]) -> Linear -> ReLU -> Linear -> sigmoid."""
    K = H_candidates.shape[0]
    h_q_tiled = np.broadcast_to(h_query, (K, h_query.shape[0]))
    x = np.concatenate([h_q_tiled, H_candidates], axis=1)  # (K, 512)
    h1 = np.maximum(0.0, x @ rec["W0"].T + rec["b0"])       # (K, 256)
    logits = (h1 @ rec["W3"].T + rec["b3"]).ravel()          # (K,)
    return 1.0 / (1.0 + np.exp(-logits))


def paper_summary(rec, idx):
    year = rec["years"][idx]
    citation_count = rec["citation_counts"][idx]
    return {
        "paperId": str(rec["paper_ids"][idx]),
        "title": str(rec["titles"][idx]) or None,
        "authors": [a.strip() for a in str(rec["authors"][idx]).split(",") if a.strip()],
        "venue": str(rec["venues"][idx]) or None,
        "year": int(year) if not np.isnan(year) else None,
        "citationCount": int(citation_count) if not np.isnan(citation_count) else None,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def api_meta():
    rec = get_recommender()
    expl = get_explainer()
    return jsonify({
        "available": bool(rec),
        "num_papers": int(len(rec["paper_ids"])) if rec else 0,
        "has_graph_index": bool(get_graph_index()),
        "has_explainer": bool(expl),
        "explainer_fidelity_r2": round(expl["fidelity_r2"], 3) if expl else None,
        "has_examples": bool(get_examples()),
        "model": "GraphSAGE link predictor (test AUC 0.993 / AP 0.986)",
    })


@app.route("/api/examples")
def api_examples():
    """Curated query papers known to have at least one confident predicted
    link -- recommended starting points."""
    rec = get_recommender()
    examples = get_examples()
    if not rec or not examples:
        return jsonify({"available": False, "results": []})

    results = []
    for ex in examples:
        pid = ex["paper_id"]
        cand_pid = ex["best_candidate_id"]
        if pid not in rec["id_to_idx"] or cand_pid not in rec["id_to_idx"]:
            continue
        results.append({
            "paper": paper_summary(rec, rec["id_to_idx"][pid]),
            "best_candidate": paper_summary(rec, rec["id_to_idx"][cand_pid]),
            "best_predicted_probability": ex["best_predicted_probability"],
        })
    return jsonify({"available": True, "results": results})


@app.route("/api/search")
def api_search():
    """Substring search over paper titles (and exact paper_id match)."""
    rec = get_recommender()
    if not rec:
        return jsonify({"available": False, "results": []})

    query = (request.args.get("q") or "").strip()
    if len(query) < 2:
        return jsonify({"available": True, "results": []})

    query_lower = query.lower()
    results = []

    if query in rec["id_to_idx"]:
        results.append(paper_summary(rec, rec["id_to_idx"][query]))

    mask = np.char.find(rec["titles_lower"], query_lower) >= 0
    idxs = np.nonzero(mask)[0][:SEARCH_RESULT_LIMIT]
    for idx in idxs:
        s = paper_summary(rec, idx)
        if s not in results:
            results.append(s)
        if len(results) >= SEARCH_RESULT_LIMIT:
            break

    return jsonify({"available": True, "results": results})


def score_shortlist(node_id, rec, shortlist_size=SHORTLIST_SIZE):
    """Shared stage-1/stage-2 scoring pipeline used by /api/recommend and
    /api/explore: cosine similarity shortlists candidates cheaply, then the
    trained LinkPredictor head re-scores exactly. Returns the raw
    ingredients so callers can post-process differently."""
    graph_index = get_graph_index()
    already_cited = set()
    query_neighbors = set()
    if graph_index and node_id in graph_index["nodes"]:
        gnode = graph_index["nodes"][node_id]
        already_cited = set(gnode["out"])
        query_neighbors = set(gnode["out"]) | set(gnode["in"])

    q_idx = rec["id_to_idx"][node_id]
    h_query = rec["embeddings"][q_idx]
    q_norm = rec["norm_embeddings"][q_idx]

    sims = rec["norm_embeddings"] @ q_norm  # (N,)
    exclude = already_cited | {node_id}
    shortlist_n = min(shortlist_size + len(exclude) + 1, len(sims))
    candidate_idxs = np.argpartition(-sims, shortlist_n - 1)[:shortlist_n]
    candidate_idxs = candidate_idxs[np.argsort(-sims[candidate_idxs])]

    paper_ids = rec["paper_ids"]
    filtered_idxs = []
    for idx in candidate_idxs:
        pid = paper_ids[idx]
        if pid in exclude:
            continue
        filtered_idxs.append(idx)
        if len(filtered_idxs) >= shortlist_size:
            break
    filtered_idxs = np.array(filtered_idxs, dtype=int)

    H_shortlist = rec["embeddings"][filtered_idxs]
    probs = predictor_forward(h_query, H_shortlist, rec)

    # Topical relevance: raw SPECTER2 content-embedding cosine similarity, not
    # the GraphSAGE space `sims` is in -- no structural/popularity signal, so
    # it's a purer check on whether a suggestion is actually on-topic. None if
    # the index predates this field.
    topical_relevance = None
    if rec["norm_content_embeddings"] is not None:
        q_content_norm = rec["norm_content_embeddings"][q_idx]
        topical_relevance = rec["norm_content_embeddings"][filtered_idxs] @ q_content_norm

    return {
        "graph_index": graph_index,
        "query_neighbors": query_neighbors,
        "q_idx": q_idx,
        "h_query": h_query,
        "sims": sims,
        "filtered_idxs": filtered_idxs,
        "probs": probs,
        "topical_relevance": topical_relevance,  # (len(filtered_idxs),) or None
    }


def shared_neighbors_for(pid, query_neighbors, graph_index, rec, limit=8):
    if not (graph_index and pid in graph_index["nodes"]):
        return [], 0
    cand_node = graph_index["nodes"][pid]
    cand_neighbors = set(cand_node["out"]) | set(cand_node["in"])
    shared_ids = list(query_neighbors & cand_neighbors)
    resolved = [
        paper_summary(rec, rec["id_to_idx"][sid])
        for sid in shared_ids[:limit] if sid in rec["id_to_idx"]
    ]
    return resolved, len(shared_ids)


@app.route("/api/recommend")
def api_recommend():
    """For a given paper, candidate papers it doesn't already cite, ranked
    by predicted probability, each with a "why" explanation (decision-tree
    surrogate, or LIME when the tree doesn't fit well)."""
    rec = get_recommender()
    if not rec:
        return jsonify({
            "available": False,
            "message": (
                "No recommender index found at link_predictor_webapp/data/recommender_embeddings.npz. "
                "Run `python build_recommender_index.py` from this folder first "
                "(on a machine with enough RAM/torch to load final_graphs.pkl and the "
                "trained GraphSAGE checkpoint)."
            ),
        }), 200

    node_id = request.args.get("node", default="", type=str)
    top_k = min(50, max(1, request.args.get("top_k", default=DEFAULT_TOP_K, type=int)))

    if node_id not in rec["id_to_idx"]:
        return jsonify({"available": True, "error": f"paper_id {node_id!r} not found in the recommender index"}), 404

    s = score_shortlist(node_id, rec)
    graph_index, query_neighbors = s["graph_index"], s["query_neighbors"]
    q_idx, h_query, sims = s["q_idx"], s["h_query"], s["sims"]
    filtered_idxs, probs, topical_relevance = s["filtered_idxs"], s["probs"], s["topical_relevance"]
    paper_ids = rec["paper_ids"]

    order = np.argsort(-probs)[:top_k]

    results = []
    for rank, pos in enumerate(order, start=1):
        idx = filtered_idxs[pos]
        pid = str(paper_ids[idx])
        shared, full_shared_count = shared_neighbors_for(pid, query_neighbors, graph_index, rec)
        real_probability = float(probs[pos])
        explanation = build_explanation(
            q_idx, idx, h_query, rec["embeddings"][idx], real_probability, full_shared_count, rec, graph_index,
        )

        results.append({
            "rank": rank,
            "paper": paper_summary(rec, idx),
            "predicted_probability": round(real_probability, 4),
            "cosine_similarity": round(float(sims[idx]), 4),
            "topical_relevance": round(float(topical_relevance[pos]), 4) if topical_relevance is not None else None,
            "shared_neighbors": shared,
            "shared_neighbor_count": full_shared_count,
            "explanation": explanation,
        })

    return jsonify({
        "available": True,
        "center": node_id,
        "center_paper": paper_summary(rec, q_idx),
        "results": results,
        "shortlist_size": len(filtered_idxs),
        "model": "GraphSAGE (test AUC 0.993 / AP 0.986)",
    })


@app.route("/api/explore")
def api_explore():
    """Combined view: a citation subgraph around the query paper (real
    edges) with the model's predicted links drawn in, plus candidates split
    into three lenses (all_time_greats / promising_stars / hidden_gems) to
    mitigate always surfacing the same most-cited papers. Bucket membership
    comes from citation degree/age; within-bucket ranking uses
    topical_relevance (see bucket_rank_key below)."""
    rec = get_recommender()
    if not rec:
        return jsonify({
            "available": False,
            "message": (
                "No recommender index found at link_predictor_webapp/data/recommender_embeddings.npz. "
                "Run `python build_recommender_index.py` from this folder first."
            ),
        }), 200

    node_id = request.args.get("node", default="", type=str)
    per_bucket = min(15, max(1, request.args.get("per_bucket", default=BUCKET_SIZE, type=int)))

    if node_id not in rec["id_to_idx"]:
        return jsonify({"available": True, "error": f"paper_id {node_id!r} not found in the recommender index"}), 404

    s = score_shortlist(node_id, rec, shortlist_size=max(SHORTLIST_SIZE, per_bucket * 20))
    graph_index, query_neighbors = s["graph_index"], s["query_neighbors"]
    q_idx, h_query, sims = s["q_idx"], s["h_query"], s["sims"]
    filtered_idxs, probs, topical_relevance = s["filtered_idxs"], s["probs"], s["topical_relevance"]
    paper_ids = rec["paper_ids"]

    years_by_pos = rec["years"][filtered_idxs]
    degrees = candidate_degrees(filtered_idxs, rec, graph_index)
    bucket_names = assign_buckets(degrees, years_by_pos)

    bucketed = {"all_time_greats": [], "promising_stars": [], "hidden_gems": []}
    for pos, name in enumerate(bucket_names):
        bucketed[name].append(pos)

    def build_item(pos, rank):
        idx = filtered_idxs[pos]
        pid = str(paper_ids[idx])
        shared, full_shared_count = shared_neighbors_for(pid, query_neighbors, graph_index, rec)
        real_probability = float(probs[pos])
        explanation = build_explanation(
            q_idx, idx, h_query, rec["embeddings"][idx], real_probability, full_shared_count, rec, graph_index,
        )
        return {
            "rank": rank,
            "paper": paper_summary(rec, idx),
            "predicted_probability": round(real_probability, 4),
            "cosine_similarity": round(float(sims[idx]), 4),
            # raw SPECTER2 content-embedding cosine similarity, no structural signal
            "topical_relevance": round(float(topical_relevance[pos]), 4) if topical_relevance is not None else None,
            # citation-graph degree assign_buckets() ranks on; distinct from paper.citationCount
            "graph_degree": int(degrees[pos]),
            "shared_neighbors": shared,
            "shared_neighbor_count": full_shared_count,
            "explanation": explanation,
        }

    # Rank within each bucket by topical_relevance rather than
    # predicted_probability, which saturates near the top of a shortlist and
    # barely discriminates by topic. predicted_probability is the
    # tie-breaker (and the sole key if topical_relevance is unavailable).
    # Bucket membership itself is unaffected -- assign_buckets() already
    # decided that upstream from degree/year.
    def bucket_rank_key(p):
        if topical_relevance is not None:
            return (-topical_relevance[p], -probs[p])
        return (-probs[p],)

    buckets_out = {}
    graph_predicted_nodes = []
    pos_to_bucket = {}
    for name, positions in bucketed.items():
        positions.sort(key=bucket_rank_key)
        top_positions = positions[:per_bucket]
        buckets_out[name] = [build_item(p, r) for r, p in enumerate(top_positions, start=1)]
        graph_predicted_nodes.extend(top_positions)
        for p in top_positions:
            pos_to_bucket[p] = name

    # Real citation subgraph around the query paper (1-hop)
    real_nodes = {}
    real_edges = []
    if graph_index and node_id in graph_index["nodes"]:
        gnode = graph_index["nodes"][node_id]
        for pid in gnode["out"][:GRAPH_MAX_NEIGHBORS_PER_DIRECTION]:
            real_nodes[pid] = "cited"
            real_edges.append({"source": node_id, "target": pid, "type": "real"})
        for pid in gnode["in"][:GRAPH_MAX_NEIGHBORS_PER_DIRECTION]:
            real_nodes[pid] = "citing"
            real_edges.append({"source": pid, "target": node_id, "type": "real"})

    graph_nodes = {node_id: {"role": "center", **paper_summary(rec, q_idx)}}
    for pid, role in real_nodes.items():
        if pid in rec["id_to_idx"]:
            graph_nodes[pid] = {"role": role, **paper_summary(rec, rec["id_to_idx"][pid])}
        else:
            graph_nodes[pid] = {"role": role, "paperId": pid, "title": None, "authors": [], "venue": None, "year": None}

    confident_positions = [p for p in graph_predicted_nodes if probs[p] >= MIN_CONFIDENT_PREDICTED_PROBABILITY]
    predicted_edges = []
    for pos in confident_positions:
        idx = filtered_idxs[pos]
        pid = str(paper_ids[idx])
        # bucket-specific role so the frontend can color-code predicted nodes;
        # a real edge always wins if a node is somehow both
        if pid not in graph_nodes:
            role = f"predicted_{pos_to_bucket.get(pos, 'all_time_greats')}"
            graph_nodes[pid] = {"role": role, **paper_summary(rec, idx)}
        predicted_edges.append({
            "source": node_id,
            "target": pid,
            "type": "predicted",
            "bucket": pos_to_bucket.get(pos),
            "predicted_probability": round(float(probs[pos]), 4),
        })

    return jsonify({
        "available": True,
        "center": node_id,
        "center_paper": paper_summary(rec, q_idx),
        "graph": {
            "nodes": list(graph_nodes.values()),
            "edges": real_edges + predicted_edges,
        },
        "buckets": buckets_out,
        "bucket_sizes": {name: len(items) for name, items in bucketed.items()},
        "has_confident_predictions": len(confident_positions) > 0,
        "min_confident_probability": MIN_CONFIDENT_PREDICTED_PROBABILITY,
        "model": "GraphSAGE (test AUC 0.993 / AP 0.986)",
    })


if __name__ == "__main__":
    # 5051: 5050 is second_hand_citation_webapp/, 5000 is claimed by macOS AirPlay
    app.run(debug=True, port=5051)
