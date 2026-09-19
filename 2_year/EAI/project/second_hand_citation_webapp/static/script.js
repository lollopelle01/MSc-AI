const state = {
  minPercentile: 0.8,
  sort: "percentile",
  query: "",
  page: 1,
  pageSize: 15,
};

const els = {
  minPercentile: document.getElementById("minPercentile"),
  minPercentileValue: document.getElementById("minPercentileValue"),
  sortSelect: document.getElementById("sortSelect"),
  searchBox: document.getElementById("searchBox"),
  statusLine: document.getElementById("statusLine"),
  results: document.getElementById("results"),
  pagination: document.getElementById("pagination"),
  graphModal: document.getElementById("graphModal"),
  graphModalTitle: document.getElementById("graphModalTitle"),
  graphModalSubtitle: document.getElementById("graphModalSubtitle"),
  graphModalClose: document.getElementById("graphModalClose"),
  graphCanvas: document.getElementById("graphCanvas"),
  graphStatus: document.getElementById("graphStatus"),
};

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function percentileBadgeClass(p) {
  if (p >= 0.95) return "badge-high";
  if (p >= 0.85) return "badge-mid";
  return "badge-low";
}

function fmtAuthors(authors) {
  if (!authors || authors.length === 0) return "";
  if (authors.length <= 2) return authors.join(", ");
  return `${authors[0]} et al.`;
}

function paperBox(role, paper, year) {
  const title = paper.title || `Untitled (id: ${paper.paperId.slice(0, 10)}…)`;
  const meta = [fmtAuthors(paper.authors), year, paper.venue].filter(Boolean).join(" · ");
  return `
    <div class="paper-box">
      <div class="paper-role">${role}</div>
      <div class="paper-title">${escapeHtml(title)}</div>
      <div class="paper-meta">${escapeHtml(meta)}</div>
    </div>
  `;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : String(str);
  return div.innerHTML;
}

function renderCard(rec) {
  const badgeClass = percentileBadgeClass(rec.percentile);
  const hasSharedAuthors = rec.shared_authors && rec.shared_authors.length > 0;
  const sharedAuthorsBadge = hasSharedAuthors
    ? `<span class="badge badge-shared-authors" title="U and W share an author -- see note below">shared author${rec.shared_authors.length > 1 ? "s" : ""}</span>`
    : "";
  const sharedAuthorsNote = hasSharedAuthors
    ? `<div class="shared-authors-note">
        U and W share ${rec.shared_authors.length > 1 ? "authors" : "an author"}: ${escapeHtml(rec.shared_authors.join(", "))}.
        This textual echo may be the same author(s) reusing their own prior phrasing rather than a genuine
        second-hand pickup through an unrelated intermediate paper -- worth weighing differently than a
        same-author-free flag.
      </div>`
    : "";
  return `
    <article class="card">
      <div class="card-head">
        <span class="rank-tag">Candidate #${rec.rank}</span>
        <span class="badge ${badgeClass}">percentile ${rec.percentile.toFixed(3)}</span>
        ${sharedAuthorsBadge}
      </div>

      <div class="triad">
        ${paperBox("Original source (X)", rec.x, rec.year_x)}
        <div class="arrow">→</div>
        ${paperBox("Intermediate (W)", rec.w, rec.year_w)}
        <div class="arrow">→</div>
        ${paperBox("Citer (U)", rec.u, rec.year_u)}
      </div>

      <div class="explanation">
        <div class="explanation-label">Why this was flagged — citing-sentence comparison</div>
        <div class="sentence-pair">
          <div class="sentence-block">
            <div class="who">U's sentence citing X</div>
            ${escapeHtml(rec.u_x_text)}
          </div>
          <div class="sentence-block">
            <div class="who">W's sentence citing X</div>
            ${escapeHtml(rec.w_x_text)}
          </div>
        </div>
        <div class="stats-row">
          <span>similarity(U→X, W→X): ${rec.similarity.toFixed(4)}</span>
          <span>background mean: ${rec.background_mean.toFixed(4)} (n=${rec.background_n})</span>
          <span>order: X (${rec.date_x || rec.year_x}) → W (${rec.date_w || rec.year_w}) → U (${rec.date_u || rec.year_u})</span>
        </div>
        ${sharedAuthorsNote}
      </div>

      <div class="card-actions">
        <button class="view-graph-btn"
          data-u="${rec.u.paperId}" data-u-title="${escapeHtml(rec.u.title || rec.u.paperId)}"
          data-w="${rec.w.paperId}" data-w-title="${escapeHtml(rec.w.title || rec.w.paperId)}"
          data-x="${rec.x.paperId}" data-x-title="${escapeHtml(rec.x.title || rec.x.paperId)}"
        >View triad graph</button>
      </div>
    </article>
  `;
}

async function loadCitations() {
  els.statusLine.textContent = "Loading…";
  const params = new URLSearchParams({
    min_percentile: state.minPercentile,
    sort: state.sort,
    q: state.query,
    page: state.page,
    page_size: state.pageSize,
  });

  try {
    const res = await fetch(`/api/citations?${params.toString()}`);
    if (!res.ok) throw new Error(`API error ${res.status}`);
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    els.statusLine.textContent = `Failed to load results: ${err.message}. Is the Flask server running?`;
    els.results.innerHTML = "";
    els.pagination.innerHTML = "";
  }
}

function renderResults(data) {
  const { total, page, page_size, results } = data;
  const totalPages = Math.max(1, Math.ceil(total / page_size));

  if (total === 0) {
    els.statusLine.textContent = "0 candidates match the current filters.";
    els.results.innerHTML = `<div class="empty-state">No flagged triads match this percentile threshold / search. Try lowering the minimum percentile.</div>`;
    els.pagination.innerHTML = "";
    return;
  }

  const start = (page - 1) * page_size + 1;
  const end = Math.min(page * page_size, total);
  els.statusLine.textContent = `Showing ${start}–${end} of ${total} flagged candidates`;

  els.results.innerHTML = results.map(renderCard).join("");
  renderPagination(page, totalPages);
}

function renderPagination(page, totalPages) {
  if (totalPages <= 1) {
    els.pagination.innerHTML = "";
    return;
  }
  const buttons = [];
  buttons.push(`<button ${page === 1 ? "disabled" : ""} data-page="${page - 1}">‹ Prev</button>`);

  const windowStart = Math.max(1, page - 2);
  const windowEnd = Math.min(totalPages, page + 2);
  if (windowStart > 1) buttons.push(`<button data-page="1">1</button>`, `<span>…</span>`);
  for (let p = windowStart; p <= windowEnd; p++) {
    buttons.push(`<button class="${p === page ? "active" : ""}" data-page="${p}">${p}</button>`);
  }
  if (windowEnd < totalPages) buttons.push(`<span>…</span>`, `<button data-page="${totalPages}">${totalPages}</button>`);

  buttons.push(`<button ${page === totalPages ? "disabled" : ""} data-page="${page + 1}">Next ›</button>`);
  els.pagination.innerHTML = buttons.join("");

  els.pagination.querySelectorAll("button[data-page]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.page = parseInt(btn.dataset.page, 10);
      loadCitations();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  });
}

els.minPercentile.addEventListener("input", () => {
  els.minPercentileValue.textContent = parseFloat(els.minPercentile.value).toFixed(2);
});
els.minPercentile.addEventListener("change", () => {
  state.minPercentile = parseFloat(els.minPercentile.value);
  state.page = 1;
  loadCitations();
});

els.sortSelect.addEventListener("change", () => {
  state.sort = els.sortSelect.value;
  state.page = 1;
  loadCitations();
});

els.searchBox.addEventListener("input", debounce(() => {
  state.query = els.searchBox.value;
  state.page = 1;
  loadCitations();
}, 350));

// ---------------------------------------------------------------------------
// Graph view -- one focused graph per flagged triad, not a general
// per-paper neighborhood explorer. U/W/X are always the same three nodes,
// colored and labeled by their role in the triad, with the triad's three
// edges highlighted so the second-hand-citation pattern is visible at a
// glance; a small amount of extra citation context can appear around them,
// but it's clearly secondary (smaller, greyed out) rather than the point.
// ---------------------------------------------------------------------------

let network = null;

const roleColor = {
  u: "#8a3324",   // citer -- the paper that may have picked X up second-hand
  w: "#2f6f4f",   // intermediate -- the paper that could be the source of the echo
  x: "#3a6ea5",   // original source -- the paper actually being cited
  context: "#c9c3b4",
};
const roleLabel = {
  u: "U · citer",
  w: "W · intermediate",
  x: "X · original source",
};
const roleSize = {
  u: 20,
  w: 20,
  x: 20,
  context: 9,
};

function nodeLabel(n) {
  const title = n.title || `(no title cached: ${n.paperId.slice(0, 8)}…)`;
  const short = title.length > 36 ? title.slice(0, 35) + "…" : title;
  const prefix = roleLabel[n.role] ? `[${n.role.toUpperCase()}] ` : "";
  return `${prefix}${short}\n${n.year || "?"}`;
}

async function openTriadGraph({ u, uTitle, w, wTitle, x, xTitle }) {
  els.graphModal.hidden = false;
  els.graphModalTitle.textContent = "Second-hand citation triad";
  els.graphModalSubtitle.textContent = "Loading triad graph…";
  els.graphStatus.textContent = "";
  els.graphCanvas.innerHTML = "";

  try {
    const params = new URLSearchParams({ u, w, x });
    const res = await fetch(`/api/triad_graph?${params.toString()}`);
    const data = await res.json();

    if (data.error) {
      els.graphModalSubtitle.textContent = "Could not build triad graph";
      els.graphStatus.textContent = data.error;
      return;
    }

    const visNodes = data.nodes.map((n) => ({
      id: n.paperId,
      label: nodeLabel(n),
      title: n.title || n.paperId,
      color: roleColor[n.role] || "#999",
      shape: "dot",
      size: roleSize[n.role] || 9,
      font: { size: n.role === "context" ? 9 : 12, color: "#1c1b19" },
    }));
    const visEdges = data.edges.map((e) => ({
      from: e.source,
      to: e.target,
      arrows: "to",
      color: { color: e.type === "triad" ? "#c2703a" : "#e2ded4" },
      width: e.type === "triad" ? 3 : 1,
    }));

    const container = els.graphCanvas;
    const options = {
      physics: {
        stabilization: { iterations: 150, fit: true },
        barnesHut: { gravitationalConstant: -5000, springLength: 110 },
      },
      interaction: { hover: true, tooltipDelay: 100 },
      edges: { smooth: { type: "continuous" } },
    };

    if (network) network.destroy();
    network = new vis.Network(container, { nodes: visNodes, edges: visEdges }, options);

    network.once("stabilizationIterationsDone", () => {
      network.fit({ animation: { duration: 300, easingFunction: "easeInOutQuad" } });
    });

    els.graphModalTitle.textContent = `U: ${uTitle}  →  W: ${wTitle}  →  X: ${xTitle}`;
    els.graphModalSubtitle.textContent =
      `${data.nodes.length} nodes in view` +
      (data.node_cap_hit ? " (hit the context node cap — extra neighbors beyond the triad are limited)" : "") +
      (data.graph_index_available ? "" : " · citation-graph index not built, so only the triad itself is shown (no extra context)");
    els.graphStatus.textContent =
      "Thick orange edges are the flagged triad itself (U→W, W→X, U→X). Thin grey edges are extra citation context around it (real citations, not predictions).";
  } catch (err) {
    els.graphModalSubtitle.textContent = "Failed to load graph";
    els.graphStatus.textContent = err.message;
  }
}

function closeGraph() {
  els.graphModal.hidden = true;
  if (network) {
    network.destroy();
    network = null;
  }
}

els.results.addEventListener("click", (e) => {
  const btn = e.target.closest(".view-graph-btn");
  if (!btn) return;
  openTriadGraph({
    u: btn.dataset.u, uTitle: btn.dataset.uTitle,
    w: btn.dataset.w, wTitle: btn.dataset.wTitle,
    x: btn.dataset.x, xTitle: btn.dataset.xTitle,
  });
});

els.graphModalClose.addEventListener("click", closeGraph);
els.graphModal.addEventListener("click", (e) => {
  if (e.target === els.graphModal) closeGraph();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !els.graphModal.hidden) closeGraph();
});

loadCitations();
