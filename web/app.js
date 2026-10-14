/**
 * TrendLens — Frontend Application Logic
 * Integrates with FastAPI endpoints to render real-time charts,
 * clusters, catalog, and image similarity search for both Fashion and Automotive genres.
 */

let trendChartInstance = null;
let currentSelectedSampleId = null;
let currentUploadFile = null;
let currentGenre = "fashion";

document.addEventListener("DOMContentLoaded", () => {
  initGenreSwitcher();
  initNavigation();
  initForecastControls();
  initSimilarityUploader();
  initCatalogFilters();

  // Load initial data for fashion (default)
  loadHealthAndHud();
  loadForecastData("Streetwear", 12);
  loadClusters();
  loadCatalog();
  loadSamplePills();
});

// Genre Switcher
function initGenreSwitcher() {
  const fashionBtn = document.getElementById("genre-fashion-btn");
  const autoBtn = document.getElementById("genre-auto-btn");
  if (!fashionBtn || !autoBtn) return;

  fashionBtn.addEventListener("click", () => switchGenre("fashion"));
  autoBtn.addEventListener("click", () => switchGenre("automotive"));
}

function switchGenre(genre) {
  if (currentGenre === genre) return;
  currentGenre = genre;

  // Toggle button styling
  const fashionBtn = document.getElementById("genre-fashion-btn");
  const autoBtn = document.getElementById("genre-auto-btn");
  if (fashionBtn) fashionBtn.classList.toggle("active", genre === "fashion");
  if (autoBtn) autoBtn.classList.toggle("active", genre === "automotive");

  // DOM elements to update
  const subtitle = document.getElementById("brand-subtitle");
  const headerTitle = document.getElementById("header-title");
  const headerSub = document.getElementById("header-subtitle");
  const simSub = document.getElementById("similarity-subtitle");
  const btnMatchLabel = document.getElementById("btn-run-match-label");
  const catalogNav = document.querySelector('.nav-btn[data-tab="catalog-tab"]');
  const catalogTitle = document.querySelector('#catalog-tab .toolbar-title h2');
  const catalogSub = document.querySelector('#catalog-tab .toolbar-title p');
  const autoNotice = document.getElementById("auto-forecast-notice");
  const styleSelectGroup = document.getElementById("style-select")?.closest(".control-group");
  const horizonGroup = document.getElementById("horizon-range")?.closest(".control-group");

  if (genre === "automotive") {
    if (subtitle) subtitle.textContent = "Automotive AI & Clustering";
    if (headerTitle) headerTitle.textContent = "Automotive Visual Intelligence";
    if (headerSub) headerSub.textContent = "Deep ResNet-18 Feature Clustering & Fine-Grained Similarity Search (Stanford Cars POC)";
    if (simSub) simSub.textContent = "Upload a car image or select a test vehicle to find Top-K visual matches";
    if (btnMatchLabel) btnMatchLabel.textContent = "Find Similar Vehicles";
    if (catalogNav) {
      catalogNav.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg> Vehicle Catalog`;
    }
    if (catalogTitle) catalogTitle.textContent = "Automotive Catalog Explorer";
    if (catalogSub) catalogSub.textContent = "Browse ingested vehicles, body types, and specifications";

    // Show automotive notice and hide forecast controls
    if (autoNotice) autoNotice.classList.remove("hidden");
    if (styleSelectGroup) styleSelectGroup.style.display = "none";
    if (horizonGroup) horizonGroup.style.display = "none";

    // Set trajectory banner for automotive
    const trajBadge = document.getElementById("traj-classification");
    if (trajBadge) {
      trajBadge.textContent = "Awaiting Automotive Sales / Search Data";
      trajBadge.className = "traj-badge badge-sky";
    }
    const trajGrowth = document.getElementById("traj-growth");
    if (trajGrowth) {
      trajGrowth.textContent = "N/A";
      trajGrowth.style.color = "var(--text-dim)";
    }
    const trajCurr = document.getElementById("traj-current");
    if (trajCurr) trajCurr.textContent = "N/A";
    const trajProj = document.getElementById("traj-projected");
    if (trajProj) trajProj.textContent = "N/A";

    // Clear chart
    if (trendChartInstance) {
      trendChartInstance.destroy();
      trendChartInstance = null;
    }

    // Leaderboard notice
    const tbody = document.getElementById("leaderboard-body");
    if (tbody) {
      tbody.innerHTML = `<tr><td colspan="5" class="text-center" style="padding: 24px; color: var(--text-dim);">
        <strong style="color: var(--accent-sky);">Automotive Forecasting Disabled:</strong> Awaiting Automotive Sales / Search Data.<br>
        Stanford Cars model_year labels encode production year only and must not be treated as a popularity or trend signal.
      </td></tr>`;
    }

    populateAutomotiveFilters();
  } else {
    // Fashion genre
    if (subtitle) subtitle.textContent = "Fashion AI & Forecasting";
    if (headerTitle) headerTitle.textContent = "Fashion Visual Intelligence";
    if (headerSub) headerSub.textContent = "End-to-End Deep Feature Learning, Cluster Aggregation & Trajectory Forecasting";
    if (simSub) simSub.textContent = "Upload a fashion item image or select a sample query to find Top-K visual matches";
    if (btnMatchLabel) btnMatchLabel.textContent = "Find Similar Garments";
    if (catalogNav) {
      catalogNav.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg> Fashion Catalog`;
    }
    if (catalogTitle) catalogTitle.textContent = "Fashion Catalog Explorer";
    if (catalogSub) catalogSub.textContent = "Browse ingested garments, visual tags, and engagement metrics";

    if (autoNotice) autoNotice.classList.add("hidden");
    if (styleSelectGroup) styleSelectGroup.style.display = "";
    if (horizonGroup) horizonGroup.style.display = "";

    populateFashionFilters();
    loadForecastData("Streetwear", 12);
  }

  // Clear upload/preview state
  clearSimilaritySelection();

  // Reload genre-specific datasets
  loadHealthAndHud();
  loadClusters();
  loadCatalog();
  loadSamplePills();
}

function populateAutomotiveFilters() {
  const catFilter = document.getElementById("catalog-category-filter");
  const styleFilter = document.getElementById("catalog-style-filter");

  if (catFilter) {
    catFilter.innerHTML = `
      <option value="">All Body Types</option>
      <option value="sedan">Sedan</option>
      <option value="suv">SUV</option>
      <option value="coupe">Coupe</option>
      <option value="truck">Truck / Pickup</option>
      <option value="convertible">Convertible</option>
      <option value="wagon">Wagon / Minivan</option>
    `;
    catFilter.value = "";
  }

  if (styleFilter) {
    styleFilter.style.display = "none";
  }
}

function populateFashionFilters() {
  const catFilter = document.getElementById("catalog-category-filter");
  const styleFilter = document.getElementById("catalog-style-filter");

  if (catFilter) {
    catFilter.innerHTML = `
      <option value="">All Categories</option>
      <option value="shirt">Shirts & Tops</option>
      <option value="trousers">Trousers & Pants</option>
      <option value="jacket">Jackets & Outerwear</option>
      <option value="dress">Dresses</option>
      <option value="skirt">Skirts</option>
    `;
    catFilter.value = "";
  }

  if (styleFilter) {
    styleFilter.style.display = "";
    styleFilter.innerHTML = `
      <option value="">All Styles</option>
      <option value="Streetwear">Streetwear</option>
      <option value="Y2K">Y2K</option>
      <option value="Minimalist">Minimalist</option>
      <option value="Denim">Denim</option>
      <option value="Cottagecore">Cottagecore</option>
      <option value="Athleisure">Athleisure</option>
    `;
    styleFilter.value = "";
  }
}

function clearSimilaritySelection() {
  currentUploadFile = null;
  currentSelectedSampleId = null;
  const fileInput = document.getElementById("image-file-input");
  if (fileInput) fileInput.value = "";
  const preview = document.getElementById("upload-preview-container");
  if (preview) preview.classList.add("hidden");
  const prompt = document.getElementById("upload-prompt");
  if (prompt) prompt.classList.remove("hidden");
  const btnMatch = document.getElementById("btn-run-match");
  if (btnMatch) btnMatch.disabled = true;

  const matchStatus = document.getElementById("match-status-label");
  if (matchStatus) matchStatus.textContent = "Awaiting search query...";
  const matchesGrid = document.getElementById("matches-grid");
  if (matchesGrid) {
    matchesGrid.innerHTML = `<div class="empty-state"><p>Upload an image or pick a test sample to calculate deep cosine similarity rankings.</p></div>`;
  }
}

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
      fetch(`/api/trends/summary?genre=${currentGenre}`).then(r => r.json())
    ]);

    const statusEl = document.getElementById("pipeline-status-indicator");
    if (healthRes.status === "healthy") {
      if (currentGenre === "automotive") {
        statusEl.textContent = `Online • ${healthRes.automotive_catalog_size} Vehicles`;
      } else {
        statusEl.textContent = `Online • ${healthRes.catalog_size} Items`;
      }
    }

    if (currentGenre === "automotive") {
      document.getElementById("hud-active-trends").textContent = "Awaiting Data";
      document.getElementById("hud-top-trend").textContent = "Sales Data Pending";
      document.getElementById("hud-clusters").textContent = "K=3 Groups";
      const silEl = document.getElementById("hud-silhouette");
      if (silEl) silEl.textContent = "Optimal K: 3";
    } else if (Array.isArray(summaryRes) && summaryRes.length > 0 && !summaryRes[0]?.forecasting_disabled) {
      document.getElementById("hud-active-trends").textContent = `${summaryRes.length} Styles`;
      const topRising = summaryRes[0];
      const topEl = document.getElementById("hud-top-trend");
      topEl.textContent = `Top: ${topRising.style} (${topRising.growth_rate_pct >= 0 ? '+' : ''}${topRising.growth_rate_pct}%)`;
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

  if (styleSelect) {
    styleSelect.addEventListener("change", () => {
      if (currentGenre === "fashion") {
        loadForecastData(styleSelect.value, parseInt(horizonSlider.value, 10));
      }
    });
  }

  if (horizonSlider) {
    horizonSlider.addEventListener("input", () => {
      if (horizonVal) horizonVal.textContent = horizonSlider.value;
    });

    horizonSlider.addEventListener("change", () => {
      if (currentGenre === "fashion") {
        loadForecastData(styleSelect.value, parseInt(horizonSlider.value, 10));
      }
    });
  }
}

async function loadForecastData(style, horizon) {
  try {
    const res = await fetch(`/api/trends/forecast?genre=${currentGenre}&style=${encodeURIComponent(style)}&horizon=${horizon}`);
    if (!res.ok) throw new Error("Forecast failed to load");
    const data = await res.json();

    if (data.forecasting_disabled) {
      updateTrajectoryBanner({
        trajectory: "Awaiting Automotive Sales / Search Data",
        badge_color: "sky",
        growth_rate_pct: 0,
        current_score: "N/A",
        projected_score: "N/A"
      });
      if (trendChartInstance) {
        trendChartInstance.destroy();
        trendChartInstance = null;
      }
      return;
    }

    updateTrajectoryBanner(data);
    renderTrendChart(data);
  } catch (err) {
    console.error("Forecast error:", err);
  }
}

function updateTrajectoryBanner(data) {
  const badge = document.getElementById("traj-classification");
  if (badge) {
    badge.textContent = data.trajectory;
    badge.className = `traj-badge badge-${data.badge_color || 'emerald'}`;
  }

  const growthEl = document.getElementById("traj-growth");
  if (growthEl) {
    if (typeof data.growth_rate_pct === 'number' && !data.forecasting_disabled) {
      growthEl.textContent = `${data.growth_rate_pct >= 0 ? '+' : ''}${data.growth_rate_pct}%`;
      growthEl.style.color = data.growth_rate_pct >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)';
    } else {
      growthEl.textContent = "N/A";
      growthEl.style.color = "var(--text-dim)";
    }
  }

  const currEl = document.getElementById("traj-current");
  if (currEl) currEl.textContent = data.current_score ?? "N/A";

  const projEl = document.getElementById("traj-projected");
  if (projEl) projEl.textContent = data.projected_score ?? "N/A";
}

function renderTrendChart(data) {
  const canvas = document.getElementById("trendChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

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
    const res = await fetch(`/api/clusters?genre=${currentGenre}`);
    if (!res.ok) return;
    const clusters = await res.json();

    const container = document.getElementById("clusters-grid");
    if (!container) return;

    const fallbackImg = currentGenre === "automotive"
      ? "/data/automotive/smoke_test/images/AUTO_0001_audi_s4_sedan_2012.jpg"
      : "/data/fashion_trends/images/FSH_0001_streetwear_shirt.jpg";

    const itemLabel = currentGenre === "automotive" ? "vehicles" : "garments";

    container.innerHTML = clusters.map(c => {
      const exemplar = c.exemplar_item || {};
      const imgPath = exemplar.file_path ? `/${exemplar.file_path}` : fallbackImg;
      const styles = Object.keys(c.dominant_styles || {}).join(", ") || "General";
      const categories = Object.keys(c.dominant_categories || {}).join(", ") || "Types";

      return `
        <div class="cluster-card">
          <img class="cluster-card-img" src="${imgPath}" alt="${c.cluster_name}" onerror="this.src='${fallbackImg}'">
          <h3>${c.cluster_name}</h3>
          <p style="font-size: 0.85rem; color: var(--text-dim); margin-bottom: 8px;">
            ${c.item_count} catalog ${itemLabel} grouped
          </p>
          <div class="cluster-tags">
            <span class="tag-badge">${currentGenre === 'automotive' ? 'Body Types' : 'Styles'}: ${styles}</span>
            <span class="tag-badge">Categories: ${categories}</span>
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

  if (catFilter) catFilter.addEventListener("change", loadCatalog);
  if (styleFilter) styleFilter.addEventListener("change", loadCatalog);
}

async function loadCatalog() {
  try {
    const cat = document.getElementById("catalog-category-filter")?.value || "";
    const style = document.getElementById("catalog-style-filter")?.value || "";

    let url = `/api/items?genre=${currentGenre}&`;
    if (cat) url += `category=${encodeURIComponent(cat)}&`;
    if (style && currentGenre === "fashion") url += `style=${encodeURIComponent(style)}&`;

    const res = await fetch(url);
    if (!res.ok) return;
    const items = await res.json();

    const grid = document.getElementById("catalog-grid");
    if (!grid) return;

    const fallbackImg = currentGenre === "automotive"
      ? "/data/automotive/smoke_test/images/AUTO_0001_audi_s4_sedan_2012.jpg"
      : "/data/fashion_trends/images/FSH_0001_streetwear_shirt.jpg";

    grid.innerHTML = items.map(item => `
      <div class="catalog-card">
        <img src="/${item.file_path}" alt="${item.title}" onerror="this.src='${fallbackImg}'">
        <div class="catalog-info">
          <h4>${item.title}</h4>
          <span class="tag-badge">${item.style || item.category} • ${item.category}</span>
          <div class="catalog-meta">
            ${item.model_year ? `<span>Year: ${item.model_year}</span>` : `<span>❤️ ${item.likes?.toLocaleString() || 0}</span>`}
            ${item.make ? `<span>Make: ${item.make}</span>` : `<span>💬 ${item.comments || 0}</span>`}
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
  const btnClear = document.getElementById("btn-clear-upload");
  const btnMatch = document.getElementById("btn-run-match");

  if (!uploadZone || !fileInput) return;

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

  if (btnClear) {
    btnClear.addEventListener("click", (e) => {
      e.stopPropagation();
      clearSimilaritySelection();
    });
  }

  if (btnMatch) {
    btnMatch.addEventListener("click", () => {
      runSimilarityMatch();
    });
  }
}

function handleSelectedFile(file) {
  currentUploadFile = file;
  currentSelectedSampleId = null;

  document.querySelectorAll(".sample-pill-btn").forEach(p => p.classList.remove("active"));

  const reader = new FileReader();
  reader.onload = (e) => {
    const previewImg = document.getElementById("upload-preview-img");
    if (previewImg) previewImg.src = e.target.result;
    document.getElementById("upload-preview-container")?.classList.remove("hidden");
    document.getElementById("upload-prompt")?.classList.add("hidden");
    const btnMatch = document.getElementById("btn-run-match");
    if (btnMatch) btnMatch.disabled = false;
  };
  reader.readAsDataURL(file);
}

async function loadSamplePills() {
  try {
    const res = await fetch(`/api/items?genre=${currentGenre}&limit=4`);
    if (!res.ok) return;
    const items = await res.json();

    const pillsContainer = document.getElementById("sample-pills");
    if (!pillsContainer) return;

    pillsContainer.innerHTML = items.map(item => `
      <button type="button" class="sample-pill-btn" data-id="${item.item_id}" data-img="/${item.file_path}">
        <strong>${item.title}</strong> (${item.style || item.category})
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
        if (previewImg) previewImg.src = btn.dataset.img;
        document.getElementById("upload-preview-container")?.classList.remove("hidden");
        document.getElementById("upload-prompt")?.classList.add("hidden");
        const btnMatch = document.getElementById("btn-run-match");
        if (btnMatch) btnMatch.disabled = false;
      });
    });
  } catch (err) {
    console.error("Samples error:", err);
  }
}

async function runSimilarityMatch() {
  const matchStatus = document.getElementById("match-status-label");
  const matchesGrid = document.getElementById("matches-grid");

  if (matchStatus) matchStatus.textContent = "Computing ResNet-18 feature vectors & cosine ranking...";
  if (matchesGrid) matchesGrid.innerHTML = `<div class="empty-state"><p>Analyzing visual features...</p></div>`;

  try {
    let res;
    if (currentUploadFile) {
      const formData = new FormData();
      formData.append("file", currentUploadFile);
      res = await fetch(`/api/similarity/match?genre=${currentGenre}&top_k=6`, {
        method: "POST",
        body: formData
      });
    } else if (currentSelectedSampleId) {
      res = await fetch(`/api/similarity/match?genre=${currentGenre}&item_id=${encodeURIComponent(currentSelectedSampleId)}&top_k=6`, {
        method: "POST"
      });
    } else {
      return;
    }

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || "Similarity match request failed");
    }
    const result = await res.json();
    renderMatches(result.matches);
    if (matchStatus) matchStatus.textContent = `Found ${result.matches.length} closest visual matches (${currentGenre})`;
  } catch (err) {
    if (matchesGrid) matchesGrid.innerHTML = `<div class="empty-state"><p style="color: var(--accent-rose)">Match failed: ${err.message}</p></div>`;
    if (matchStatus) matchStatus.textContent = "Error occurred";
  }
}

function renderMatches(matches) {
  const grid = document.getElementById("matches-grid");
  if (!grid) return;
  if (!matches || matches.length === 0) {
    grid.innerHTML = `<div class="empty-state"><p>No visual matches found.</p></div>`;
    return;
  }

  const fallbackImg = currentGenre === "automotive"
    ? "/data/automotive/smoke_test/images/AUTO_0001_audi_s4_sedan_2012.jpg"
    : "/data/fashion_trends/images/FSH_0001_streetwear_shirt.jpg";

  grid.innerHTML = matches.map(m => `
    <div class="match-card">
      <div class="match-badge">${m.match_percentage}% Match</div>
      <img src="/${m.file_path}" alt="${m.title}" onerror="this.src='${fallbackImg}'">
      <div class="match-info">
        <h4>${m.title}</h4>
        <p>${m.style || m.category} • ${m.category}</p>
        <div style="font-size: 0.75rem; color: var(--text-dim); margin-top: 6px;">
          Rank #${m.rank} ${m.model_year ? `• Year: ${m.model_year}` : (m.likes ? `• ❤️ ${m.likes.toLocaleString()}` : '')}
        </div>
      </div>
    </div>
  `).join("");
}
