"""Detects candidate second-hand citations via citing-sentence similarity
on closed triads (u, w, x): u cites w, w cites x, u also cites x directly.
Each triad's citing-sentence similarity (u->x vs w->x, SciBERT mean-pooled
embeddings) is scored as a percentile against a background of similarity
between other citers of x, so a high score means u's and w's citing
sentences for x echo each other more than unrelated citations of x do.

    pip install transformers
    python scripts/second_hand_citation.py --split test --top-k 20
"""
import argparse
import itertools
import os
import random
from collections import defaultdict
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common.atomic_cache import load_cache, save_cache
from common.text_embedding import mean_pool
from data_utils import CACHE_DIR, load_split_graphs

SCIBERT_MODEL_NAME = "allenai/scibert_scivocab_uncased"
EMBED_DIM = 768
EDGE_TEXT_CACHE = os.path.join(CACHE_DIR, "scibert_edge_embeddings.pkl")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GNNs", "results")


def parse_pub_date(s):
    """Parses an ISO `publication_date` string, returning None if missing/invalid."""
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def build_edge_lookup(g):
    """Maps (u, v) -> the `unique_id`/`string` of ONE real edge u -> v
    (the first one found, if there are parallel edges from multiple
    citing contexts)."""
    lookup = {}
    for e in g.es:
        key = (e.source, e.target)
        if key not in lookup:
            lookup[key] = (e["unique_id"], e["string"] or "")
    return lookup


def embed_needed_sentences(uid_text_pairs, cache, batch_size, device):
    """Embeds every (uid, text) not already in cache, updates cache in
    place, and returns it. Loads SciBERT lazily -- only if there's
    actually something new to embed."""
    todo = [(uid, text) for uid, text in uid_text_pairs if uid not in cache]
    if not todo:
        return cache

    print(f"Embedding {len(todo)} new citing sentences with {SCIBERT_MODEL_NAME}...")
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(SCIBERT_MODEL_NAME).to(device)
    model.eval()

    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        texts = [t if t.strip() else "[EMPTY]" for _, t in batch]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        vecs = mean_pool(out.last_hidden_state, enc["attention_mask"]).cpu().numpy()
        for (uid, text), vec in zip(batch, vecs):
            cache[uid] = np.zeros(EMBED_DIM, dtype=np.float32) if not text.strip() else vec.astype(np.float32)

    save_cache(EDGE_TEXT_CACHE, cache)
    return cache


def find_closed_triads(g, max_triads, seed):
    """Yields (u, w, x): u -> w, w -> x, u -> x all real edges. Iterates
    (w, x) edges in shuffled order so a cap on max_triads doesn't bias
    toward low-index papers."""
    rng = random.Random(seed)
    edge_set = set(g.get_edgelist())
    wx_edges = list(edge_set)
    rng.shuffle(wx_edges)

    found = 0
    for w, x in wx_edges:
        if found >= max_triads:
            break
        # set(): g is a multigraph; predecessors() yields one entry per
        # parallel edge, which would duplicate (u, w, x) otherwise.
        for u in set(g.predecessors(w)):
            if u != w and u != x and (u, x) in edge_set:
                yield u, w, x
                found += 1
                if found >= max_triads:
                    break


def main():
    parser = argparse.ArgumentParser(
        description="Detect candidate second-hand citations via citing-sentence "
                    "similarity on closed triads (u->w->x with u->x also real)."
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--max-triads", type=int, default=5000)
    parser.add_argument("--min-citers", type=int, default=3,
                         help="a target x needs at least this many citers (u, w, +1 more) to have a background")
    parser.add_argument("--max-background-per-target", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = torch.device(args.device)

    print(f"Loading {args.split} split...")
    graphs = load_split_graphs()
    g = graphs[args.split]

    print("Finding closed triads (u -> w -> x, with u -> x also real)...")
    triads = list(find_closed_triads(g, args.max_triads, args.seed))
    print(f"Found {len(triads)} closed triads")

    citers_by_target = defaultdict(set)
    for src, dst in g.get_edgelist():
        citers_by_target[dst].add(src)

    edge_lookup = build_edge_lookup(g)

    # Pairs needing embeddings: triad edges + each target's background citer edges.
    needed_pairs = set()
    triad_backgrounds = {}
    for u, w, x in triads:
        needed_pairs.add((u, x))
        needed_pairs.add((w, x))

        citers = citers_by_target[x]
        if len(citers) < args.min_citers:
            continue
        other_citers = [c for c in citers if c not in (u, w)]
        pairs = (list(itertools.combinations(other_citers, 2))
                 + [(c, w) for c in other_citers] + [(c, u) for c in other_citers])
        if len(pairs) > args.max_background_per_target:
            pairs = rng.sample(pairs, args.max_background_per_target)
        triad_backgrounds[(u, w, x)] = pairs
        for c1, c2 in pairs:
            needed_pairs.add((c1, x))
            needed_pairs.add((c2, x))

    uid_text_pairs = [edge_lookup[(u, v)] for (u, v) in needed_pairs if (u, v) in edge_lookup]
    print(f"{len(needed_pairs)} distinct citing-sentence embeddings needed "
          f"({len(uid_text_pairs)} resolved to a real edge)")

    cache = load_cache(EDGE_TEXT_CACHE)
    cache = embed_needed_sentences(uid_text_pairs, cache, args.batch_size, device)

    def emb(u, v):
        if (u, v) not in edge_lookup:
            return None
        uid, text = edge_lookup[(u, v)]
        # Missing text embeds as a zero vector, which would falsely inflate
        # similarity with other missing-text pairs -- skip instead.
        if not text or not text.strip():
            return None
        return cache.get(uid)

    def sim(u1, v1, u2, v2):
        a, b = emb(u1, v1), emb(u2, v2)
        if a is None or b is None:
            return None
        a_t, b_t = torch.from_numpy(a), torch.from_numpy(b)
        return F.cosine_similarity(a_t.unsqueeze(0), b_t.unsqueeze(0)).item()

    results = []
    skipped, skipped_temporal = 0, 0
    for u, w, x in triads:
        # u->w->x only tells a coherent "picked up through w" story if
        # x predates w and w predates u.
        year_u, year_w, year_x = g.vs[u]["year"], g.vs[w]["year"], g.vs[x]["year"]
        if year_u is None or year_w is None or year_x is None:
            skipped_temporal += 1
            continue
        if not (year_x <= year_w <= year_u):
            skipped_temporal += 1
            continue

        # Refine with day-precision dates where parseable, for same-year triads.
        date_u = parse_pub_date(g.vs[u]["publication_date"])
        date_w = parse_pub_date(g.vs[w]["publication_date"])
        date_x = parse_pub_date(g.vs[x]["publication_date"])
        if date_x is not None and date_w is not None and date_x > date_w:
            skipped_temporal += 1
            continue
        if date_w is not None and date_u is not None and date_w > date_u:
            skipped_temporal += 1
            continue

        if (u, w, x) not in triad_backgrounds:
            skipped += 1
            continue

        target_sim = sim(u, x, w, x)
        if target_sim is None:
            skipped += 1
            continue

        background = [s for c1, c2 in triad_backgrounds[(u, w, x)]
                      if (s := sim(c1, x, c2, x)) is not None]
        if not background:
            skipped += 1
            continue

        percentile = float(np.mean(np.array(background) <= target_sim))
        results.append({
            "u": g.vs[u]["paper_id"], "w": g.vs[w]["paper_id"], "x": g.vs[x]["paper_id"],
            "year_u": year_u, "year_w": year_w, "year_x": year_x,
            "date_u": date_u, "date_w": date_w, "date_x": date_x,
            "similarity": target_sim, "background_mean": float(np.mean(background)),
            "background_n": len(background), "percentile": percentile,
            "u_x_text": edge_lookup[(u, x)][1], "w_x_text": edge_lookup[(w, x)][1],
        })

    print(f"Scored {len(results)} triads ({skipped} skipped: no background or missing text, "
          f"{skipped_temporal} skipped: missing year, year(x) <= year(w) <= year(u) not satisfied, "
          f"or publication_date shows x/w/u out of order within a shared year)")

    if not results:
        return

    results.sort(key=lambda r: (r["percentile"], r["similarity"]), reverse=True)
    percentiles = np.array([r["percentile"] for r in results])
    print(f"\nPercentile distribution: mean={percentiles.mean():.3f}  "
          f"median={np.median(percentiles):.3f}  "
          f"frac >= 0.95: {(percentiles >= 0.95).mean():.3f}")

    def truncate(text, n=70):
        text = " ".join(text.split())
        return text if len(text) <= n else text[:n - 1] + "…"

    df = pd.DataFrame([{
        "rank": i + 1,
        "percentile": round(r["percentile"], 3),
        "similarity": round(r["similarity"], 4),
        "bg_mean": round(r["background_mean"], 4),
        "n_bg": r["background_n"],
        "u": r["u"][:12], "w": r["w"][:12], "x": r["x"][:12],
        "date_u": r["date_u"], "date_w": r["date_w"], "date_x": r["date_x"],
        "u->x text": truncate(r["u_x_text"]),
        "w->x text": truncate(r["w_x_text"]),
    } for i, r in enumerate(results)])

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Untruncated version of the console `df` above, written to disk.
    full_df = pd.DataFrame([{
        "rank": i + 1,
        "percentile": r["percentile"],
        "similarity": r["similarity"],
        "background_mean": r["background_mean"],
        "background_n": r["background_n"],
        "u": r["u"], "w": r["w"], "x": r["x"],
        "year_u": r["year_u"], "year_w": r["year_w"], "year_x": r["year_x"],
        "date_u": r["date_u"], "date_w": r["date_w"], "date_x": r["date_x"],
        "u_x_text": r["u_x_text"], "w_x_text": r["w_x_text"],
    } for i, r in enumerate(results)])
    results_csv = os.path.join(RESULTS_DIR, f"second_hand_citation_{args.split}.csv")
    full_df.to_csv(results_csv, index=False)
    print(f"\nFull results ({len(full_df)} rows, untruncated) written to {results_csv}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(percentiles, bins=30, range=(0, 1))
    ax.axvline(0.95, color="red", linestyle="--", label="0.95 threshold")
    ax.set_xlabel("percentile (similarity vs. background)")
    ax.set_ylabel("count")
    ax.set_title(f"Second-hand citation candidate triads -- {args.split} split (n={len(results)})")
    ax.legend()
    plot_path = os.path.join(RESULTS_DIR, f"second_hand_citation_{args.split}_percentiles.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Percentile distribution plot written to {plot_path}")

    print(f"\nAll {len(df)} scored triads, ordered by percentile (u -> x, plausibly derived from w -> x):\n")
    pd.set_option("display.max_colwidth", None)
    print(df.to_string(index=False))

    print(f"\nCiting-sentence text for the top {args.top_k} (the 'local explanation' for each flagged pair):\n")
    for i, r in enumerate(results[:args.top_k]):
        print(f"  #{i + 1}  u={r['u']}  w={r['w']}  x={r['x']}  percentile={r['percentile']:.3f}")
        print(f"    u->x: {r['u_x_text'][:160]!r}")
        print(f"    w->x: {r['w_x_text'][:160]!r}")


if __name__ == "__main__":
    main()
