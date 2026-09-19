"""Shared hard-negative candidate generation for the citation link-
existence models. A "hard" negative is an open 2-hop triad: u cites w, w
cites v, but u -> v is not a real edge -- structurally plausible unlike a
random unrelated pair. Candidates also enforce year(v) <= year(u) (u
can't cite a paper published after itself); missing years aren't filtered.

Two flavors:
  - build_hard_negative_candidates: an UNANCHORED candidate pool, for
    post-hoc evaluation.
  - precompute_two_hop_targets + sample_matched_hard_negatives: an
    ANCHORED sampler tied to a specific source u, for training with a
    pairwise ranking loss where each positive (u, v) needs a matched
    negative (u, v') sharing the same anchor.
"""
import random


def build_hard_negative_candidates(g, max_total=20000, max_per_hub=50, seed=0):
    """Finds open 2-hop triads (u, v): u -> w and w -> v both exist, but
    u -> v does not. Caps how many candidates a single intermediate paper w
    can contribute (max_per_hub) so a handful of hub papers with huge
    in/out-degree can't dominate the sample or blow up runtime, and stops
    once max_total candidates are collected. Node visit order is shuffled
    so the result isn't biased toward low-index papers."""
    rng = random.Random(seed)
    edge_set = set(g.get_edgelist())
    years = g.vs["year"]
    order = list(range(g.vcount()))
    rng.shuffle(order)

    def chrono_ok(u, v):
        yu, yv = years[u], years[v]
        return yu is None or yv is None or yv <= yu

    candidates = set()
    for w in order:
        if len(candidates) >= max_total:
            break
        preds = g.predecessors(w)
        succs = g.successors(w)
        if not preds or not succs:
            continue

        if len(preds) * len(succs) > max_per_hub * 20:
            # Sample pairs directly instead of the full cross product.
            picked, attempts = set(), 0
            while len(picked) < max_per_hub and attempts < max_per_hub * 10:
                u, v = rng.choice(preds), rng.choice(succs)
                attempts += 1
                if u != v and (u, v) not in edge_set and chrono_ok(u, v):
                    picked.add((u, v))
            pairs = list(picked)
        else:
            pairs = [(u, v) for u in preds for v in succs
                     if u != v and (u, v) not in edge_set and chrono_ok(u, v)]
            if len(pairs) > max_per_hub:
                pairs = rng.sample(pairs, max_per_hub)

        for p in pairs:
            candidates.add(p)
            if len(candidates) >= max_total:
                break

    return list(candidates)


def precompute_two_hop_targets(g, cap_per_node=50, seed=0):
    """For every node u, finds 2-hop-reachable papers (u -> w -> v) not
    already cited directly by u -- per-anchor hard-negative candidates.
    Returns {u: [v, v, ...]}; nodes with no candidate are absent (callers
    fall back to a random negative, see sample_matched_hard_negatives)."""
    rng = random.Random(seed)
    n = g.vcount()
    years = g.vs["year"]
    direct_succ = [set(s) for s in g.get_adjlist(mode="out")]
    two_hop = g.neighborhood(vertices=None, order=2, mode="out")

    targets = {}
    for u in range(n):
        yu = years[u]
        candidates = [
            v for v in two_hop[u]
            if v != u and v not in direct_succ[u]
            and (yu is None or years[v] is None or years[v] <= yu)
        ]
        if not candidates:
            continue
        if len(candidates) > cap_per_node:
            candidates = rng.sample(candidates, cap_per_node)
        targets[u] = candidates
    return targets


def sample_matched_hard_negatives(sources, two_hop_targets, num_nodes, rng):
    """For a list of source node indices (the u side of a batch of
    positive (u, v) edges), draws one matched negative target per source:
    a random pick from two_hop_targets[u] if available, otherwise a
    uniformly-random node (fallback for anchors with no 2-hop candidate,
    e.g. leaves). Returns (targets, n_hard, n_fallback)."""
    out = []
    n_hard = 0
    for u in sources:
        u = int(u)
        cands = two_hop_targets.get(u)
        if cands:
            out.append(rng.choice(cands))
            n_hard += 1
        else:
            out.append(rng.randrange(num_nodes))
    return out, n_hard, len(sources) - n_hard
