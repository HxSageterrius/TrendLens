/**
 * TrendLens — Frontend Application Logic
 * Integrates with FastAPI endpoints to render real-time charts,
 * clusters, catalog, and image similarity search.
 */

let trendChartInstance = null;
let currentSelectedSampleId = null;
let currentUploadFile = null;

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initForecastControls();
  initSimilarityUploader();
  initCatalogFilters();

  // Load initial data
  loadHealthAndHud();
  loadForecastData("Streetwear", 12);
  loadClusters();
  loadCatalog();
  loadSamplePills();
});

// Navigation Tabs
function initNavigation() {
  const navBtns = document.querySelectorAll(".nav-btn");
  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      navBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      const targetTabId = btn.dataset.tab;
      document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.remove("active");
      });
      const activePane = document.getElementById(targetTabId);
      if (activePane) activePane.classList.add("active");
    });
  });
}

// Health and HUD
async function loadHealthAndHud() {
  try {
    const [healthRes, summaryRes] = await Promise.all([
      fetch("/api/health").then(r => r.json()),
      fetch("/api/trends/summary").then(r => r.json())
    ]);

    const statusEl = document.getElementById("pipeline-status-indicator");
    if (healthRes.status === "healthy") {
      statusEl.textContent = `Online • ${healthRes.catalog_size} Items`;
    }

    if (Array.isArray(summaryRes) && summaryRes.length > 0) {
      document.getElementById("hud-active-trends").textContent = `${summaryRes.length} Styles`;
      const topRising = summaryRes[0];
      const topEl = document.getElementById("hud-top-trend");
      topEl.textContent = `Top: ${topRising.style} (${topRising.growth_rate_pct >= 0 ? '+' : ''}${topRising.growth_rate_pct}%)`;

      // Render leaderboard
      renderLeaderboard(summaryRes);
    }
  } catch (err) {
    console.warn("HUD fetch notice:", err);
  }
}

// Trend Forecasting
function initForecastControls() {
  const styleSelect = document.getElementById("style-select");
  const horizonSlider = document.getElementById("horizon-range");
  const horizonVal = document.getElementById("horizon-val");

  styleSelect.addEventListener("change", () => {
    loadForecastData(styleSelect.value, parseInt(horizonSlider.value, 10));
  });

  horizonSlider.addEventListener("input", () => {
    horizonVal.textContent = horizonSlider.value;
  });

  horizonSlider.addEventListener("change", () => {
    loadForecastData(styleSelect.value, parseInt(horizonSlider.value, 10));
  });
}

async function loadForecastData(style, horizon) {
  try {
    const res = await fetch(`/api/trends/forecast?style=${encodeURIComponent(style)}&horizon=${horizon}`);
    if (!res.ok) throw new Error("Forecast failed to load");
    const data = await res.json();

    updateTrajectoryBanner(data);
    renderTrendChart(data);
  } catch (err) {
    console.error("Forecast error:", err);
  }
}

function updateTrajectoryBanner(data) {
  const badge = document.getElementById("traj-classification");
  badge.textContent = data.trajectory;
  badge.className = `traj-badge badge-${data.badge_color || 'emerald'}`;

  const growthEl = document.getElementById("traj-growth");
  growthEl.textContent = `${data.growth_rate_pct >= 0 ? '+' : ''}${data.growth_rate_pct}%`;
  growthEl.style.color = data.growth_rate_pct >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)';

  document.getElementById("traj-current").textContent = data.current_score;
  document.getElementById("traj-projected").textContent = data.projected_score;
}

function renderTrendChart(data) {
  const ctx = document.getElementById("trendChart").getContext("2d");

  const histDates = data.historical_points.map(p => p.date);
  const histValues = data.historical_points.map(p => p.value);

  const fcDates = data.forecast_points.map(p => p.date);
  const fcValues = data.forecast_points.map(p => p.yhat);
  const fcLower = data.forecast_points.map(p => p.yhat_lower);
  const fcUpper = data.forecast_points.map(p => p.yhat_upper);

  const allLabels = [...histDates, ...fcDates];

  // Align historical dataset (pad nulls for future dates)
  const histPadded = [...histValues, ...new Array(fcDates.length).fill(null)];

  // Align forecast dataset (pad nulls for past dates except connecting point)
  const fcPadded = [...new Array(histDates.length - 1).fill(null), histValues[histValues.length - 1], ...fcValues];
  const upperPadded = [...new Array(histDates.length - 1).fill(null), histValues[histValues.length - 1], ...fcUpper];
  const lowerPadded = [...new Array(histDates.length - 1).fill(null), histValues[histValues.length - 1], ...fcLower];

  if (trendChartInstance) {
    trendChartInstance.destroy();
  }

  trendChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: allLabels,
      datasets: [
        {
          label: "Upper Confidence (90%)",
          data: upperPadded,
          borderColor: "transparent",
          backgroundColor: "rgba(139, 92, 246, 0.12)",
          fill: "+1",
          pointRadius: 0,
          tension: 0.3
        },
        {
          label: "Lower Confidence (90%)",
          data: lowerPadded,
          borderColor: "transparent",
          backgroundColor: "transparent",
          pointRadius: 0,
          tension: 0.3
        },
        {
          label: "Historical Volume",
          data: histPadded,
          borderColor: "#06b6d4",
          backgroundColor: "rgba(6, 182, 212, 0.15)",
          borderWidth: 2.5,
          tension: 0.25,
          pointRadius: 0,
          pointHoverRadius: 5,
        },
        {
          label: "Projected Trend (Forecast)",
          data: fcPadded,
          borderColor: "#a855f7",
          borderWidth: 3,
          borderDash: [6, 4],
          tension: 0.3,
          pointRadius: 2,
          pointBackgroundColor: "#a855f7"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      plugins: {
        legend: {
          position: "top",
          labels: {
            color: "#94a3b8",
            filter: item => !item.text.includes("Confidence")
          }
        },
        tooltip: {
          backgroundColor: "rgba(15, 23, 42, 0.9)",
          titleColor: "#f8fafc",
          bodyColor: "#94a3b8",
          borderColor: "rgba(255, 255, 255, 0.1)",
          borderWidth: 1
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: {
            color: "#64748b",
            maxTicksLimit: 10
          }
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.04)" },
          ticks: { color: "#64748b" },
          title: {
            display: true,
            text: "Trend Engagement Volume",
            color: "#64748b"
          }
        }
      }
    }
  });
}

function renderLeaderboard(stylesList) {
  const tbody = document.getElementById("leaderboard-body");
  if (!tbody) return;

  tbody.innerHTML = stylesList.map(item => `
    <tr>
      <td><strong>${item.style}</strong></td>
      <td>${item.current_score}</td>
      <td>${item.projected_score}</td>
      <td style="color: ${item.growth_rate_pct >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'}; font-weight: 600;">
        ${item.growth_rate_pct >= 0 ? '+' : ''}${item.growth_rate_pct}%
      </td>
      <td>
        <span class="traj-badge badge-${item.badge_color}">${item.trajectory}</span>
      </td>
    </tr>
  `).join("");
}

// Visual Clusters
async function loadClusters() {
  try {
    const res = await fetch("/api/clusters");
    if (!res.ok) return;
    const clusters = await res.json();

    const container = document.getElementById("clusters-grid");
    if (!container) return;

    container.innerHTML = clusters.map(c => {
      const exemplar = c.exemplar_item || {};
      const imgPath = exemplar.file_path ? `/${exemplar.file_path}` : "/web/placeholder.jpg";
      const styles = Object.keys(c.dominant_styles || {}).join(", ") || "General";
      const categories = Object.keys(c.dominant_categories || {}).join(", ") || "Items";

      return `
        <div class="cluster-card">
          <img class="cluster-card-img" src="${imgPath}" alt="${c.cluster_name}" onerror="this.src='/data/fashion_trends/images/FSH_0001_streetwear_shirt.jpg'">
          <h3>${c.cluster_name}</h3>
          <p style="font-size: 0.85rem; color: var(--text-dim); margin-bottom: 8px;">
            ${c.item_count} catalog garments grouped
          </p>
          <div class="cluster-tags">
            <span class="tag-badge">Styles: ${styles}</span>
            <span class="tag-badge">Types: ${categories}</span>
          </div>
          <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 10px;">
            <strong>Centroid Exemplar:</strong> ${exemplar.title || 'N/A'}
          </p>
        </div>
      `;
    }).join("");
  } catch (err) {
    console.error("Cluster load error:", err);
  }
}

// Catalog Explorer
function initCatalogFilters() {
  const catFilter = document.getElementById("catalog-category-filter");
  const styleFilter = document.getElementById("catalog-style-filter");

  catFilter.addEventListener("change", loadCatalog);
  styleFilter.addEventListener("change", loadCatalog);
}

async function loadCatalog() {
  try {
    const cat = document.getElementById("catalog-category-filter")?.value || "";
    const style = document.getElementById("catalog-style-filter")?.value || "";

    let url = "/api/items?";
    if (cat) url += `category=${encodeURIComponent(cat)}&`;
    if (style) url += `style=${encodeURIComponent(style)}&`;

    const res = await fetch(url);
    if (!res.ok) return;
    const items = await res.json();

    const grid = document.getElementById("catalog-grid");
    if (!grid) return;

    grid.innerHTML = items.map(item => `
      <div class="catalog-card">
        <img src="/${item.file_path}" alt="${item.title}" onerror="this.src='/data/fashion_trends/images/FSH_0001_streetwear_shirt.jpg'">
        <div class="catalog-info">
          <h4>${item.title}</h4>
          <span class="tag-badge">${item.style} • ${item.category}</span>
          <div class="catalog-meta">
            <span>❤️ ${item.likes?.toLocaleString() || 0}</span>
            <span>💬 ${item.comments || 0}</span>
          </div>
        </div>
      </div>
    `).join("");
  } catch (err) {
    console.error("Catalog load error:", err);
  }
}

// Visual Similarity Search
function initSimilarityUploader() {
  const uploadZone = document.getElementById("upload-zone");
  const fileInput = document.getElementById("image-file-input");
  const uploadPrompt = document.getElementById("upload-prompt");
  const previewContainer = document.getElementById("upload-preview-container");
  const previewImg = document.getElementById("upload-preview-img");
  const btnClear = document.getElementById("btn-clear-upload");
  const btnMatch = document.getElementById("btn-run-match");

  uploadZone.addEventListener("click", (e) => {
    if (e.target !== btnClear) fileInput.click();
  });

  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = "var(--accent-cyan)";
  });

  uploadZone.addEventListener("dragleave", () => {
    uploadZone.style.borderColor = "var(--border-glass)";
  });

  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = "var(--border-glass)";
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      handleSelectedFile(fileInput.files[0]);
    }
  });

  btnClear.addEventListener("click", (e) => {
    e.stopPropagation();
    currentUploadFile = null;
    fileInput.value = "";
    previewContainer.classList.add("hidden");
    uploadPrompt.classList.remove("hidden");
    btnMatch.disabled = !currentSelectedSampleId;
  });

  btnMatch.addEventListener("click", () => {
    runSimilarityMatch();
  });
}

function handleSelectedFile(file) {
  currentUploadFile = file;
  currentSelectedSampleId = null;

  document.querySelectorAll(".sample-pill-btn").forEach(p => p.classList.remove("active"));

  const reader = new FileReader();
  reader.onload = (e) => {
    const previewImg = document.getElementById("upload-preview-img");
    previewImg.src = e.target.result;
    document.getElementById("upload-preview-container").classList.remove("hidden");
    document.getElementById("upload-prompt").classList.add("hidden");
    document.getElementById("btn-run-match").disabled = false;
  };
  reader.readAsDataURL(file);
}

async function loadSamplePills() {
  try {
    const res = await fetch("/api/items?limit=4");
    if (!res.ok) return;
    const items = await res.json();

    const pillsContainer = document.getElementById("sample-pills");
    if (!pillsContainer) return;

    pillsContainer.innerHTML = items.map(item => `
      <button type="button" class="sample-pill-btn" data-id="${item.item_id}" data-img="/${item.file_path}">
        <strong>${item.title}</strong> (${item.style})
      </button>
    `).join("");

    pillsContainer.querySelectorAll(".sample-pill-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        pillsContainer.querySelectorAll(".sample-pill-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        currentSelectedSampleId = btn.dataset.id;
        currentUploadFile = null;

        // Show preview
        const previewImg = document.getElementById("upload-preview-img");
        previewImg.src = btn.dataset.img;
        document.getElementById("upload-preview-container").classList.remove("hidden");
        document.getElementById("upload-prompt").classList.add("hidden");
        document.getElementById("btn-run-match").disabled = false;
      });
    });
  } catch (err) {
    console.error("Samples error:", err);
  }
}

async function runSimilarityMatch() {
  const matchStatus = document.getElementById("match-status-label");
  const matchesGrid = document.getElementById("matches-grid");

  matchStatus.textContent = "Computing ResNet feature vectors & cosine ranking...";
  matchesGrid.innerHTML = `<div class="empty-state"><p>Analyzing visual features...</p></div>`;

  try {
    let res;
    if (currentUploadFile) {
      const formData = new FormData();
      formData.append("file", currentUploadFile);
      res = await fetch("/api/similarity/match?top_k=6", {
        method: "POST",
        body: formData
      });
    } else if (currentSelectedSampleId) {
      res = await fetch(`/api/similarity/match?item_id=${encodeURIComponent(currentSelectedSampleId)}&top_k=6`, {
        method: "POST"
      });
    } else {
      return;
    }

    if (!res.ok) throw new Error("Similarity match request failed");
    const result = await res.json();
    renderMatches(result.matches);
    matchStatus.textContent = `Found ${result.matches.length} closest visual matches`;
  } catch (err) {
    matchesGrid.innerHTML = `<div class="empty-state"><p style="color: var(--accent-rose)">Match failed: ${err.message}</p></div>`;
    matchStatus.textContent = "Error occurred";
  }
}

function renderMatches(matches) {
  const grid = document.getElementById("matches-grid");
  if (!matches || matches.length === 0) {
    grid.innerHTML = `<div class="empty-state"><p>No visual matches found.</p></div>`;
    return;
  }

  grid.innerHTML = matches.map(m => `
    <div class="match-card">
      <div class="match-badge">${m.match_percentage}% Match</div>
      <img src="/${m.file_path}" alt="${m.title}" onerror="this.src='/data/fashion_trends/images/FSH_0001_streetwear_shirt.jpg'">
      <div class="match-info">
        <h4>${m.title}</h4>
        <p>${m.style} • ${m.category}</p>
        <div style="font-size: 0.75rem; color: var(--text-dim); margin-top: 6px;">
          Rank #${m.rank} • ❤️ ${m.likes?.toLocaleString() || 0}
        </div>
      </div>
    </div>
  `).join("");
}
