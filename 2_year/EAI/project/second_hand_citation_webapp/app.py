"""Second-Hand Citation Detector -- local web app. Serves the results of
scripts/second_hand_citation.py (candidate "second-hand" citation triads:
u -> w -> x, where u also cites x directly and u's citing sentence for x
reads suspiciously like w's) as a browsable, filterable, explained list.
Paper metadata is resolved on demand from the local Semantic Scholar
detail cache (graphs/cache/detail_cache_compact.sqlite3). Run with
`python app.py` after `pip install -r requirements.txt`."""
import json
import os
import pickle
import sqlite3
from functools import lru_cache

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

CSV_PATH = os.path.join(PROJECT_ROOT, "GNNs", "results", "second_hand_citation_test.csv")
SQLITE_PATH = os.path.join(PROJECT_ROOT, "graphs", "cache", "detail_cache_compact.sqlite3")
GRAPH_INDEX_PATH = os.path.join(BASE_DIR, "data", "test_graph_index.json")
FALLBACK_CACHE_PATH = os.path.join(BASE_DIR, "data", "title_fallback_cache.json")

TRIAD_CONTEXT_NEIGHBORS_PER_DIRECTION = 3  # extra (non-triad) neighbors each of U/W/X may contribute, per direction
TRIAD_MAX_TOTAL_NODES = 18  # keeps the view focused on the triad itself, not a general neighborhood explorer

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_df = None
_sqlite_conn = None
_graph_index = None  # None = not yet attempted, False = attempted and unavailable, dict = loaded
_fallback_cache = None  # paper_id -> {title, authors, venue, year}, from fetch_missing_titles.py


def get_fallback_cache():
    global _fallback_cache
    if _fallback_cache is None:
        if os.path.exists(FALLBACK_CACHE_PATH):
            with open(FALLBACK_CACHE_PATH) as f:
                _fallback_cache = json.load(f)
        else:
            _fallback_cache = {}
    return _fallback_cache


def get_df():
    global _df
    if _df is None:
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(
                f"Results CSV not found at {CSV_PATH}. Run "
                f"`python scripts/second_hand_citation.py --split test` first "
                f"from the project root to generate it."
            )
        df = pd.read_csv(CSV_PATH)
        # dates come back as strings from CSV; keep them as-is (already ISO)
        _df = df
    return _df


def get_sqlite_conn():
    global _sqlite_conn
    if _sqlite_conn is None:
        if not os.path.exists(SQLITE_PATH):
            _sqlite_conn = False  # sentinel: no metadata source available
        else:
            _sqlite_conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    return _sqlite_conn


@lru_cache(maxsize=8192)
def get_paper_meta(paper_id: str):
    """Looks up title/authors/venue/year for a paper_id: first the sqlite
    detail cache (core SciCite corpus), then title_fallback_cache.json
    (graph-neighbor papers backfilled by fetch_missing_titles.py). Falls
    back to a stub if neither source has it."""
    stub = {
        "paperId": paper_id,
        "title": None,
        "authors": [],
        "venue": None,
        "year": None,
    }

    conn = get_sqlite_conn()
    if conn:
        cur = conn.execute("SELECT payload FROM details WHERE paper_id = ?", (paper_id,))
        row = cur.fetchone()
        if row is not None:
            try:
                payload = pickle.loads(row[0])
                if payload.get("title"):
                    return {
                        "paperId": paper_id,
                        "title": payload.get("title"),
                        "authors": [a.get("name") for a in (payload.get("authors") or []) if a.get("name")],
                        "venue": payload.get("venue") or None,
                        "year": payload.get("year"),
                    }
            except Exception as e:
                # A failure here (e.g. numpy version mismatch unpickling the
                # payload) used to fall silently through to the stub.
                app.logger.warning(
                    "get_paper_meta: failed to unpickle cached payload for %s: %r",
                    paper_id, e,
                )

    fallback = get_fallback_cache().get(paper_id)
    if fallback and fallback.get("title"):
        return {"paperId": paper_id, **fallback}

    return stub


def get_graph_index():
    """Lazily loads the structural index built by build_graph_index.py.
    Returns False if it hasn't been generated yet (optional feature)."""
    global _graph_index
    if _graph_index is None:
        if not os.path.exists(GRAPH_INDEX_PATH):
            _graph_index = False
        else:
            with open(GRAPH_INDEX_PATH) as f:
                _graph_index = json.load(f)
    return _graph_index


def truncate(text, n=280):
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def shared_authors(meta_a, meta_b):
    """Author names common to both papers, loose-matched (lowercased,
    whitespace-trimmed) since the metadata cache stores whatever name
    string Semantic Scholar returned, not a disambiguated author ID -- this
    will miss genuine overlaps where the name is formatted differently
    across the two papers (e.g. "T. Nijboer" vs "Tanja C. W. Nijboer") and,
    much more rarely, could over-match two different people who happen to
    share a common name. It's a real but imperfect signal, not proof either
    way -- see the report's discussion of why this matters: a strong
    citing-sentence echo between two papers by the same author(s) is much
    more plausibly self-reuse of their own prior phrasing than a genuine
    second-hand pickup through an unrelated intermediate paper, so this is
    surfaced to let the user judge that for themselves rather than
    presenting every flagged triad as equally suggestive of the latter."""
    def norm_names(authors):
        return {a.strip().lower() for a in (authors or []) if a and a.strip()}
    overlap = norm_names(meta_a.get("authors")) & norm_names(meta_b.get("authors"))
    return sorted(overlap)


def row_to_record(r):
    u_meta = get_paper_meta(r["u"])
    w_meta = get_paper_meta(r["w"])
    return {
        "rank": int(r["rank"]),
        "percentile": round(float(r["percentile"]), 4),
        "similarity": round(float(r["similarity"]), 4),
        "background_mean": round(float(r["background_mean"]), 4),
        "background_n": int(r["background_n"]),
        "u": u_meta,
        "w": w_meta,
        "x": get_paper_meta(r["x"]),
        "year_u": int(r["year_u"]) if pd.notna(r["year_u"]) else None,
        "year_w": int(r["year_w"]) if pd.notna(r["year_w"]) else None,
        "year_x": int(r["year_x"]) if pd.notna(r["year_x"]) else None,
        "date_u": r["date_u"] if pd.notna(r["date_u"]) else None,
        "date_w": r["date_w"] if pd.notna(r["date_w"]) else None,
        "date_x": r["date_x"] if pd.notna(r["date_x"]) else None,
        "u_x_text": truncate(r["u_x_text"]),
        "w_x_text": truncate(r["w_x_text"]),
        # U and W are the two papers whose citing sentences are actually
        # being compared (both cite X) -- if they share an author, the
        # textual echo this tool flags is at least as plausibly the same
        # person(s) reusing their own prior phrasing as it is a genuine
        # second-hand pickup. See shared_authors() docstring.
        "shared_authors": shared_authors(u_meta, w_meta),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def api_meta():
    df = get_df()
    return jsonify({
        "total": int(len(df)),
        "min_percentile": float(df["percentile"].min()),
        "max_percentile": float(df["percentile"].max()),
        "has_metadata_source": bool(get_sqlite_conn()),
    })


@app.route("/api/citations")
def api_citations():
    df = get_df()

    min_percentile = request.args.get("min_percentile", default=0.0, type=float)
    query = (request.args.get("q") or "").strip().lower()
    sort = request.args.get("sort", default="percentile")
    page = max(1, request.args.get("page", default=1, type=int))
    page_size = min(100, max(1, request.args.get("page_size", default=15, type=int)))

    filtered = df[df["percentile"] >= min_percentile]

    sort_cols = {
        "percentile": ["percentile", "similarity"],
        "similarity": ["similarity"],
        "date_desc": ["year_u"],
        "date_asc": ["year_x"],
    }.get(sort, ["percentile", "similarity"])
    ascending = sort == "date_asc"
    filtered = filtered.sort_values(sort_cols, ascending=ascending)

    # Metadata is resolved before text search since it's needed anyway.
    records = [row_to_record(r) for _, r in filtered.iterrows()]

    if query:
        def matches(rec):
            hay = " ".join([
                str(rec["u"]["title"] or ""), str(rec["w"]["title"] or ""), str(rec["x"]["title"] or ""),
                rec["u_x_text"], rec["w_x_text"],
            ]).lower()
            return query in hay
        records = [r for r in records if matches(r)]

    total = len(records)
    start = (page - 1) * page_size
    end = start + page_size
    page_records = records[start:end]

    return jsonify({
        "total": total,
        "page": page,
        "page_size": page_size,
        "results": page_records,
    })


@app.route("/api/triad_graph")
def api_triad_graph():
    """One focused graph per flagged triad: always contains the three
    triad papers U, W, X (tagged by role) with the triad edges u->w, w->x,
    u->x marked "triad". If the graph index is available, a small capped
    amount of extra citation context is layered in around them, tagged
    "context" (see TRIAD_CONTEXT_NEIGHBORS_PER_DIRECTION, TRIAD_MAX_TOTAL_NODES)."""
    u = request.args.get("u", default="", type=str)
    w = request.args.get("w", default="", type=str)
    x = request.args.get("x", default="", type=str)
    if not (u and w and x):
        return jsonify({"available": True, "error": "u, w, and x paper_id query params are all required"}), 400

    included_roles = {x: "x", w: "w", u: "u"}
    edges = []
    edge_set = set()

    def add_edge(src, tgt, kind):
        key = (src, tgt)
        if key in edge_set:
            return
        edge_set.add(key)
        edges.append({"source": src, "target": tgt, "type": kind})

    # The triad itself: always drawn, regardless of graph-index availability.
    add_edge(u, w, "triad")
    add_edge(w, x, "triad")
    add_edge(u, x, "triad")

    index = get_graph_index()
    node_cap_hit = False
    if index:
        nodes_by_id = index["nodes"]

        def add_context_from(pid):
            nonlocal node_cap_hit
            node = nodes_by_id.get(pid)
            if node is None:
                return
            for nid in node["out"][:TRIAD_CONTEXT_NEIGHBORS_PER_DIRECTION]:
                if nid not in included_roles and len(included_roles) >= TRIAD_MAX_TOTAL_NODES:
                    node_cap_hit = True
                    continue
                add_edge(pid, nid, "context")
                included_roles.setdefault(nid, "context")
            for nid in node["in"][:TRIAD_CONTEXT_NEIGHBORS_PER_DIRECTION]:
                if nid not in included_roles and len(included_roles) >= TRIAD_MAX_TOTAL_NODES:
                    node_cap_hit = True
                    continue
                add_edge(nid, pid, "context")
                included_roles.setdefault(nid, "context")

        for pid in (u, w, x):
            add_context_from(pid)

        # Cross-links between included nodes, so context renders as a
        # small web rather than disjoint stars.
        for pid in list(included_roles.keys()):
            node = nodes_by_id.get(pid)
            if node is None:
                continue
            for nid in node["out"]:
                if nid in included_roles:
                    add_edge(pid, nid, "context")

    result_nodes = []
    for pid, role in included_roles.items():
        # get_paper_meta is @lru_cache-decorated and returns the same dict
        # object each call -- copy before mutating with the "role" field.
        meta = dict(get_paper_meta(pid))
        meta["role"] = role
        result_nodes.append(meta)

    return jsonify({
        "available": True,
        "graph_index_available": bool(index),
        "u": u,
        "w": w,
        "x": x,
        "nodes": result_nodes,
        "edges": edges,
        "node_cap_hit": node_cap_hit,
    })


if __name__ == "__main__":
    # Port 5050, not 5000: on macOS, 5000 is often claimed by AirPlay Receiver.
    app.run(debug=True, port=5050)
