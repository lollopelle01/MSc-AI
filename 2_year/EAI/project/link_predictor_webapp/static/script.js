const els = {
  searchBox: document.getElementById("searchBox"),
  searchResults: document.getElementById("searchResults"),
  statusLine: document.getElementById("statusLine"),
  examplesPanel: document.getElementById("examplesPanel"),
  examplesList: document.getElementById("examplesList"),
  exploreLayout: document.getElementById("exploreLayout"),
  centerPaper: document.getElementById("centerPaper"),
  graphCanvas: document.getElementById("graphCanvas"),
  buckets: {
    all_time_greats: document.getElementById("bucket-all_time_greats"),
    promising_stars: document.getElementById("bucket-promising_stars"),
    hidden_gems: document.getElementById("bucket-hidden_gems"),
  },
};

let network = null;
let explainCounter = 0;
const LIME_N_SAMPLES_LABEL = "~300"; // matches LIME_N_SAMPLES in app.py, for the explainer text only

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : String(str);
  return div.innerHTML;
}

function fmtAuthors(authors) {
  if (!authors || authors.length === 0) return "";
  if (authors.length <= 2) return authors.join(", ");
  return `${authors[0]} et al.`;
}

function paperMetaLine(p) {
  const citations = p.citationCount != null ? `${p.citationCount.toLocaleString()} citations` : null;
  return [fmtAuthors(p.authors), p.year, p.venue, citations].filter(Boolean).join(" · ");
}

function bucketProfileLine(item) {
  // Year and total citations are shown just above, in .paper-meta -- this
  // line adds the ONE number that actually drives bucket placement
  // (assign_buckets() ranks by this, not by the real-world citation
  // count), so it's checkable, not just asserted.
  return `${item.graph_degree} edge${item.graph_degree === 1 ? "" : "s"} in this app's indexed citation subgraph (the number bucket placement is actually ranked on)`;
}

function probabilityBadgeClass(p) {
  if (p >= 0.8) return "badge-high";
  if (p >= 0.5) return "badge-mid";
  return "badge-low";
}

// Thresholds here are a starting heuristic, not calibrated against a
// labeled dataset -- raw SPECTER2 cosine similarity runs on a different
// scale than the GraphSAGE-embedding cosine shown next to it (no
// structural/popularity signal mixed in), so "high" here means something
// different than "high" for cos. Worth recalibrating once you've spent
// time looking at real values across several query papers.
function topicalRelevanceBadgeClass(t) {
  if (t === null || t === undefined) return "badge-unknown";
  if (t >= 0.5) return "badge-high";
  if (t >= 0.25) return "badge-mid";
  return "badge-low";
}

// The specific pattern worth flagging: the model is confident (high
// predicted_probability) but the raw content-embedding similarity is low --
// i.e. this suggestion is likely being driven by something other than
// topical closeness (candidate_degree/popularity, per the SKE explanation),
// exactly the failure mode found in the report's walkthrough example.
function isLikelyRelevanceMismatch(item) {
  return item.topical_relevance !== null && item.topical_relevance !== undefined
    && item.predicted_probability >= 0.8 && item.topical_relevance < 0.25;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

async function runSearch(query) {
  if (query.trim().length < 2) {
    els.searchResults.innerHTML = "";
    return;
  }
  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const data = await res.json();
    if (!data.available) {
      els.searchResults.innerHTML = "";
      els.statusLine.textContent = "Recommender index not built yet — see the README for build_recommender_index.py.";
      return;
    }
    if (data.results.length === 0) {
      els.searchResults.innerHTML = `<div class="empty-state">No matching papers.</div>`;
      return;
    }
    els.searchResults.innerHTML = data.results.map((p) => `
      <button class="search-result-item" data-node="${escapeHtml(p.paperId)}">
        <div class="search-result-title">${escapeHtml(p.title || `(untitled: ${p.paperId.slice(0, 10)}…)`)}</div>
        <div class="search-result-meta">${escapeHtml(paperMetaLine(p))}</div>
      </button>
    `).join("");
  } catch (err) {
    els.statusLine.textContent = `Search failed: ${err.message}`;
  }
}

els.searchBox.addEventListener("input", debounce((e) => runSearch(e.target.value), 300));

els.searchResults.addEventListener("click", (e) => {
  const btn = e.target.closest(".search-result-item");
  if (!btn) return;
  explore(btn.dataset.node);
});

// ---------------------------------------------------------------------------
// Curated examples (landing page)
// ---------------------------------------------------------------------------

async function loadExamples() {
  try {
    const res = await fetch("/api/examples");
    const data = await res.json();
    if (!data.available || data.results.length === 0) {
      els.examplesPanel.hidden = true;
      return;
    }
    els.examplesList.innerHTML = data.results.map((ex) => `
      <button class="example-item" data-node="${escapeHtml(ex.paper.paperId)}">
        <div class="example-item-title">${escapeHtml(ex.paper.title || ex.paper.paperId.slice(0, 10) + "…")}</div>
        <div class="example-item-arrow">→ predicted P(link) ${ex.best_predicted_probability.toFixed(2)} →</div>
        <div class="example-item-candidate">${escapeHtml(ex.best_candidate.title || ex.best_candidate.paperId.slice(0, 10) + "…")}</div>
      </button>
    `).join("");
    els.examplesPanel.hidden = false;
  } catch (err) {
    els.examplesPanel.hidden = true;
  }
}

els.examplesList.addEventListener("click", (e) => {
  const btn = e.target.closest(".example-item");
  if (!btn) return;
  explore(btn.dataset.node);
});

// ---------------------------------------------------------------------------
// Explanation rendering (decision-tree path or LIME coefficients)
// ---------------------------------------------------------------------------

function renderSharedNeighbors(item) {
  if (item.shared_neighbor_count === 0) {
    return `<div class="shared-block shared-block-empty">No papers in the indexed subgraph are cited by (or cite) both this paper and the one you're exploring — this suggestion comes purely from the model's learned embedding, not a shared citation.</div>`;
  }
  const shown = item.shared_neighbors.length;
  const chips = item.shared_neighbors.map((s) =>
    `<span class="shared-chip" title="${escapeHtml(s.paperId)}">${escapeHtml(s.title || s.paperId.slice(0, 8) + "…")}</span>`
  ).join("");
  const moreNote = item.shared_neighbor_count > shown
    ? ` (+${item.shared_neighbor_count - shown} more not shown)`
    : "";
  return `<div class="shared-block"><span class="shared-label">Shared citation neighborhood${moreNote}:</span>${chips}</div>`;
}

function renderExplanation(item) {
  const exp = item.explanation;
  const sharedHtml = renderSharedNeighbors(item);
  if (!exp) {
    return `<div class="explain-panel">${sharedHtml}<div style="margin-top:0.5rem;">No decision-tree/LIME explainer built yet — run build_recommender_index.py's decision-tree step.</div></div>`;
  }
  if (exp.method === "decision_tree") {
    const steps = exp.path.map((s) =>
      `<div class="explain-path-step">${escapeHtml(s.feature)} ${s.op} ${s.threshold} <span style="color:var(--ink-soft)">(actual: ${s.actual})</span></div>`
    ).join("");
    return `
      <div class="explain-panel">
        ${sharedHtml}
        <span class="explain-method-tag">decision tree (SKE)</span>
        <div>Tree's own estimate: <strong>${exp.tree_predicted_probability.toFixed(3)}</strong> · global fidelity R² = ${exp.global_fidelity_r2}</div>
        ${steps || "<div class=\"explain-path-step\">(root is a leaf — no split needed)</div>"}
      </div>
    `;
  }
  if (exp.method === "lime") {
    const maxAbs = Math.max(...Object.values(exp.coefficients).map((v) => Math.abs(v)), 1e-6);
    const coefRows = Object.entries(exp.coefficients).map(([name, val]) => {
      const width = Math.min(100, (Math.abs(val) / maxAbs) * 100);
      const color = val >= 0 ? "var(--accent)" : "var(--danger, #a13a2b)";
      const direction = val >= 0 ? "pushes the score up" : "pushes the score down";
      return `<div class="explain-coef"><span>${escapeHtml(name)} <span style="color:var(--ink-soft)">(${direction})</span></span>
        <span>${val.toFixed(4)} <span class="explain-coef-bar" style="width:${width}px; background:${color}"></span></span></div>`;
    }).join("");
    return `
      <div class="explain-panel">
        ${sharedHtml}
        <span class="explain-method-tag">LIME (local surrogate)</span>
        <p class="explain-intro">
          LIME explains one single prediction, not the model overall: it nudges this candidate's
          real embedding slightly (${LIME_N_SAMPLES_LABEL} random small variations), asks the
          <em>real</em> trained model for its score on each variation, then fits a simple straight-line
          model to how the score moved. The numbers below are that line's slope per feature — how
          much the real model's score tends to shift when that feature moves, <em>right around this
          specific prediction</em>. Only <code>cosine_similarity</code> actually varies under this
          perturbation (the other three are structural/metadata, not derived from the embedding), so a
          near-zero coefficient there just means the model isn't very sensitive to embedding-space
          wiggle for this candidate.
        </p>
        <div style="margin-bottom:0.3rem;">${escapeHtml(exp.reason)}</div>
        <div>How well this local line fits (R²) = ${exp.local_fidelity_r2}</div>
        ${coefRows}
      </div>
    `;
  }
  return `<div class="explain-panel">${sharedHtml}</div>`;
}

function renderItemCard(item, cssClass) {
  const p = item.paper;
  const title = p.title || `(untitled: ${p.paperId.slice(0, 10)}…)`;
  explainCounter += 1;
  const explainId = `explain-${explainCounter}`;
  const hasRelevance = item.topical_relevance !== null && item.topical_relevance !== undefined;
  const relevanceTag = hasRelevance
    ? `<span class="badge ${topicalRelevanceBadgeClass(item.topical_relevance)}">topic sim ${item.topical_relevance.toFixed(3)}</span>`
    : `<span class="badge badge-unknown" title="Rebuild the recommender index to enable this">topic sim n/a</span>`;
  const mismatchWarning = isLikelyRelevanceMismatch(item)
    ? `<div class="relevance-warning">High confidence, but low topical relevance to the query paper — likely driven by citation-graph popularity rather than topic. Worth checking before relying on this suggestion.</div>`
    : "";
  return `
    <article class="${cssClass}" data-node="${escapeHtml(p.paperId)}" data-title="${escapeHtml(title)}">
      <div class="bucket-item-head">
        <span class="rank-tag">#${item.rank}</span>
        <span class="badge ${probabilityBadgeClass(item.predicted_probability)}">P(link) ${item.predicted_probability.toFixed(3)}</span>
        <span class="cosine-tag">cos ${item.cosine_similarity.toFixed(3)}</span>
        ${relevanceTag}
      </div>
      <div class="bucket-item-title">${escapeHtml(title)}</div>
      <div class="paper-meta">${escapeHtml(paperMetaLine(p))}</div>
      <div class="bucket-profile-line">${escapeHtml(bucketProfileLine(item))}</div>
      ${mismatchWarning}
      <button class="explain-toggle" type="button" data-target="${explainId}">Explain this prediction</button>
      <span class="card-hint">click card to explore →</span>
      <div class="explain-panel" id="${explainId}" hidden>${renderExplanation(item)}</div>
    </article>
  `;
}

document.addEventListener("click", (e) => {
  const toggle = e.target.closest(".explain-toggle");
  if (!toggle) return;
  e.stopPropagation();
  const panel = document.getElementById(toggle.dataset.target);
  if (panel) panel.hidden = !panel.hidden;
});

// ---------------------------------------------------------------------------
// Graph rendering (real citation edges + predicted links, one merged view)
// ---------------------------------------------------------------------------

// Predicted nodes/edges are colored by which bucket they landed in, so the
// graph and the side panels read as the same categorization, not two
// separate views. Real citation edges keep their own distinct colors
// (unrelated to bucketing, since they already exist).
const bucketColor = {
  all_time_greats: "#b5762a",   // amber -- matches --warn
  promising_stars: "#2f6f4f",   // green -- matches --good/--accent
  hidden_gems: "#7d5ba6",       // plum -- distinct from anything else on the page
};
const roleColor = {
  center: "#8a3324",   // rust
  citing: "#3a6ea5",   // blue
  cited: "#5a8ba0",    // teal
  predicted_all_time_greats: bucketColor.all_time_greats,
  predicted_promising_stars: bucketColor.promising_stars,
  predicted_hidden_gems: bucketColor.hidden_gems,
};
const roleSize = { center: 24, citing: 14, cited: 14 };
const DEFAULT_PREDICTED_SIZE = 12;

function nodeLabel(n) {
  const title = n.title || `(no title: ${n.paperId.slice(0, 8)}…)`;
  const short = title.length > 36 ? title.slice(0, 35) + "…" : title;
  return `${short}\n${n.year || "?"}`;
}

function renderGraph(graph, centerId) {
  const visNodes = graph.nodes.map((n) => ({
    id: n.paperId,
    label: nodeLabel(n),
    title: n.title || n.paperId,
    color: roleColor[n.role] || "#999",
    shape: "dot",
    size: roleSize[n.role] || DEFAULT_PREDICTED_SIZE,
    font: { size: n.role === "center" ? 13 : 10, color: "#1c1b19" },
  }));
  const visEdges = graph.edges.map((e) => {
    const edgeColor = e.type === "predicted" ? (bucketColor[e.bucket] || "#999") : "#8a3324";
    return {
      from: e.source,
      to: e.target,
      arrows: "to",
      dashes: e.type === "predicted",
      color: { color: edgeColor, opacity: e.type === "predicted" ? 0.6 : 0.85 },
      title: e.type === "predicted" ? `predicted link (${e.bucket}) · P(link) = ${e.predicted_probability}` : "real citation",
    };
  });

  const options = {
    physics: {
      stabilization: { iterations: 150, fit: true },
      barnesHut: { gravitationalConstant: -6000, springLength: 120 },
    },
    interaction: { hover: true, tooltipDelay: 100 },
    edges: { smooth: { type: "continuous" } },
  };

  if (network) network.destroy();
  network = new vis.Network(els.graphCanvas, { nodes: visNodes, edges: visEdges }, options);
  network.once("stabilizationIterationsDone", () => {
    network.fit({ animation: { duration: 300, easingFunction: "easeInOutQuad" } });
  });
  network.on("click", (params) => {
    if (params.nodes.length > 0) {
      explore(params.nodes[0]);
    }
  });
}

// ---------------------------------------------------------------------------
// Main explore flow
// ---------------------------------------------------------------------------

async function explore(nodeId) {
  els.searchResults.innerHTML = "";
  els.searchBox.value = "";
  els.examplesPanel.hidden = true;
  els.statusLine.textContent = "Scoring candidates and building the subgraph…";

  try {
    const res = await fetch(`/api/explore?node=${encodeURIComponent(nodeId)}`);
    const data = await res.json();

    if (!data.available) {
      els.statusLine.textContent = data.message;
      els.exploreLayout.hidden = true;
      return;
    }
    if (data.error) {
      els.statusLine.textContent = data.error;
      els.exploreLayout.hidden = true;
      return;
    }

    els.exploreLayout.hidden = false;
    els.centerPaper.innerHTML = `
      <div class="center-paper">
        <div class="center-paper-label">Exploring</div>
        <div class="center-paper-title">${escapeHtml(data.center_paper.title || `(untitled: ${data.center_paper.paperId.slice(0, 10)}…)`)}</div>
        <div class="center-paper-meta">${escapeHtml(paperMetaLine(data.center_paper))}</div>
      </div>
    `;

    renderGraph(data.graph, data.center);

    for (const [name, el] of Object.entries(els.buckets)) {
      const items = data.buckets[name] || [];
      el.innerHTML = items.length
        ? items.map((item) => renderItemCard(item, "bucket-item")).join("")
        : `<div class="empty-state">No candidates fell into this bucket.</div>`;
    }

    document.querySelectorAll(".bucket-item").forEach((card) => {
      card.addEventListener("click", (e) => {
        if (e.target.closest(".explain-toggle")) return;
        explore(card.dataset.node);
      });
    });

    const confidenceNote = data.has_confident_predictions
      ? ""
      : ` No candidate cleared the confidence bar (P(link) ≥ ${data.min_confident_probability}) for a predicted edge — this paper's local neighborhood is sparse. The bucket rankings below are still real, just all lower-confidence; try one of the example papers for a clearer demonstration.`;
    els.statusLine.textContent = `${data.model} · ${data.graph.nodes.length} nodes in the subgraph. Click any node or bucket item to recenter.${confidenceNote}`;
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (err) {
    els.statusLine.textContent = `Failed to load: ${err.message}`;
  }
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

(async function init() {
  try {
    const res = await fetch("/api/meta");
    const data = await res.json();
    if (!data.available) {
      els.statusLine.textContent = "Recommender index not built yet. Run `python build_recommender_index.py` from this folder first, then reload.";
    } else {
      let msg = `${data.num_papers.toLocaleString()} papers indexed. Search above to get started.`;
      if (!data.has_graph_index) {
        msg += " (Note: no citation-neighborhood index found — real edges/shared neighborhood/buckets will be limited. Run second_hand_citation_webapp/build_graph_index.py.)";
      }
      if (!data.has_explainer) {
        msg += " (Note: no explainer found — run build_recommender_index.py's decision-tree step for the 'Why?' panels.)";
      } else {
        msg += ` Explainer fidelity R² = ${data.explainer_fidelity_r2}.`;
      }
      els.statusLine.textContent = msg;
      if (data.available) loadExamples();
    }
  } catch (err) {
    els.statusLine.textContent = `Failed to reach the server: ${err.message}. Is the Flask server running?`;
  }
})();
