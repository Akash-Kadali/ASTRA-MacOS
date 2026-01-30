/* ============================================================
   ASTRA • dashboard.js (v2.1.0 — Analytics & History Dashboard)
   ------------------------------------------------------------
   Features:
   • Fetches /api/dashboard (summary + trend + history) with timeout
   • Falls back to /api/dashboard/summary automatically
   • Renders the Dashboard UI (ids from dashboard.html)
   • Uses unified localStorage history (hirex_history) as fallback
   • Caches API data for 12h; works offline from cache
   • Optional refresh/status elements if present
   • Safe even if parts of the DOM/Chart.js are missing
   Author: Sri Akash Kadali
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  const APP = "ASTRA";
  const APP_VERSION = "v2.1.0";

  /* ------------------------------------------------------------
     🔧 DOM References (optional-safe)
  ------------------------------------------------------------ */
  const $ = (id) => document.getElementById(id);

  // Metric counters
  const elOpt   = $("opt_count");
  const elCL    = $("cl_count");
  const elHM    = $("hm_count");
  const elMM    = $("mm_count");
  const elTalk  = $("talk_count");

  // Chart canvases
  const cvs = {
    optimizations: $("chart_optimizations"),
    coverletters:  $("chart_coverletters"),
    superhuman:    $("chart_superhuman"),
    mastermind:    $("chart_mastermind"),
    talk:          $("chart_talk"),
  };

  // Optional elements
  const refreshBtn = $("dashboard-refresh");
  const statusBar  = $("dashboard-status");

  /* ------------------------------------------------------------
     🧠 Utilities
  ------------------------------------------------------------ */
  const RT = (window.ASTRA ?? window.HIREX) || {};
  const debug = (msg, data) => (typeof RT.debugLog === "function" ? RT.debugLog(msg, data) : void 0);
  const toast = (msg, t = 3000) => (RT.toast ? RT.toast(msg, t) : console.info(msg));

  const getApiBase = () => {
    try { if (typeof RT.getApiBase === "function") return RT.getApiBase(); } catch {}
    if (["127.0.0.1", "0.0.0.0", "localhost"].includes(location.hostname)) return "http://127.0.0.1:8000";
    return location.origin;
  };
  const apiBase = getApiBase();

  const dayLabels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const dayIdx = (d) => (d.getDay() + 6) % 7; // Monday=0

  const CACHE_TTL_MS = 12 * 60 * 60 * 1000;
  const CACHE_KEY_NEW = "astra_dashboard_cache_v2";
  const CACHE_KEY_OLD = "hirex_dashboard_cache_v2"; // migrate if present

  const loadCache = () => {
    const read = (k) => { try { return JSON.parse(localStorage.getItem(k) || "null"); } catch { return null; } };
    let obj = read(CACHE_KEY_NEW) || read(CACHE_KEY_OLD);
    if (!obj || !obj._cached_at) return null;
    if (Date.now() - obj._cached_at > CACHE_TTL_MS) {
      localStorage.removeItem(CACHE_KEY_NEW);
      localStorage.removeItem(CACHE_KEY_OLD);
      return null;
    }
    // Defensive: shape
    obj.summary ||= {};
    obj.trend ||= {};
    obj.history = Array.isArray(obj.history) ? obj.history : [];
    return obj;
  };

  const saveCache = (data) => {
    try {
      localStorage.setItem(
        CACHE_KEY_NEW,
        JSON.stringify({ ...data, _cached_at: Date.now(), version: APP_VERSION })
      );
    } catch (e) {
      console.warn("[ASTRA] Dashboard cache save failed:", e);
    }
  };

  const getLocalHistory = () => {
    try { return JSON.parse(localStorage.getItem("hirex_history") || "[]"); }
    catch { return []; }
  };

  // timeout-friendly fetch
  async function fetchJSON(url, ms = 10000) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), ms);
    try {
      const r = await fetch(url, { signal: ctrl.signal, cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } finally {
      clearTimeout(t);
    }
  }

  /* ------------------------------------------------------------
     📦 Normalize & Fallback shaping
  ------------------------------------------------------------ */
  const TYPE_ALIASES = {
    optimization: new Set(["optimization", "opt", "resume", "resume_optimization", "optimize", "optimized"]),
    coverletter:  new Set(["coverletter", "cover_letter", "cover-letter", "cl", "coverletter_generated"]),
    humanize:     new Set(["humanize", "superhuman", "hm"]),
    mastermind:   new Set(["mastermind", "mm", "chatgpt"]),
    talk:         new Set(["talk", "qa", "assistant", "talk_to_hirex"]),
  };

  const pickType = (t = "") => {
    const key = String(t || "").toLowerCase();
    for (const [norm, set] of Object.entries(TYPE_ALIASES)) {
      if (set.has(key)) return norm;
    }
    // Heuristic substrings for backend "event" names
    if (/optimiz/.test(key)) return "optimization";
    if (/cover.?letter/.test(key)) return "coverletter";
    if (/superhuman|humanize/.test(key)) return "humanize";
    if (/mastermind/.test(key)) return "mastermind";
    if (/talk/.test(key)) return "talk";
    return key || "unknown";
  };

  // Ensure history items expose a "type" and "timestamp"
  const normalizeHistory = (arr = []) =>
    (Array.isArray(arr) ? arr : []).map((r) => ({
      ...r,
      type: r?.type || r?.event || "unknown",
      timestamp: r?.timestamp || r?.time || r?.ts || new Date().toISOString(),
    }));

  // Build a Mon..Sun series for a given normalized type
  const seriesFromHistory = (history, normType) => {
    const out = Array(7).fill(0);
    history.forEach((r) => {
      const t = pickType(r.type);
      if (t !== normType) return;
      const d = new Date(r.timestamp);
      if (isNaN(d)) return;
      out[dayIdx(d)] += 1;
    });
    return out;
  };

  const countsFromHistory = (history) => {
    const c = { optimization:0, coverletter:0, humanize:0, mastermind:0, talk:0 };
    history.forEach((r) => {
      const t = pickType(r.type);
      if (t in c) c[t] += 1;
    });
    return c;
  };

  /* ------------------------------------------------------------
     📊 Renderers (Metrics + Charts)
  ------------------------------------------------------------ */
  const activeCharts = {};

  const ensureChart = (canvas, cfgBuilder) => {
    if (!canvas || !window.Chart) return null;
    const existing = (Chart.getChart && Chart.getChart(canvas)) || activeCharts[canvas.id];
    try { existing?.destroy?.(); } catch {}
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    let cfg;
    try { cfg = cfgBuilder(ctx); } catch { return null; }
    try {
      const chart = new Chart(ctx, cfg);
      activeCharts[canvas.id] = chart;
      return chart;
    } catch {
      return null;
    }
  };

  const gradient = (ctx, c1, c2) => {
    try {
      const g = ctx.createLinearGradient(0, 0, 0, 300);
      g.addColorStop(0, c1); g.addColorStop(1, c2);
      return g;
    } catch {
      return c1;
    }
  };

  const baseOptions = {
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: "#bbb" }, grid: { color: "rgba(255,255,255,0.05)" } },
      y: { ticks: { color: "#888" }, grid: { color: "rgba(255,255,255,0.05)" } },
    },
  };

  // Helper to read number from multiple possible keys (API/back-compat)
  const pickNum = (obj, keys, fallback = 0) => {
    for (const k of keys) {
      const v = obj?.[k];
      if (typeof v === "number" && !Number.isNaN(v)) return v;
      if (typeof v === "string" && v.trim() !== "" && !Number.isNaN(Number(v))) return Number(v);
    }
    return fallback;
  };

  const renderMetrics = (summary = {}, fallbackHistory = []) => {
    // API (v2 backend) fields:
    //  optimize_runs, coverletters, superhuman_calls, talk_queries, mastermind_chats
    const histCounts = countsFromHistory(normalizeHistory(fallbackHistory));

    const opt  = pickNum(summary, ["optimize_runs", "opt_count", "opt", "optimizations"], histCounts.optimization);
    const cl   = pickNum(summary, ["coverletters", "cl_count"],                       histCounts.coverletter);
    const hm   = pickNum(summary, ["superhuman_calls", "hm_count", "humanize"],       histCounts.humanize);
    const mm   = pickNum(summary, ["mastermind_chats", "mm_count"],                   histCounts.mastermind);
    const talk = pickNum(summary, ["talk_queries", "talk_count"],                     histCounts.talk);

    if (elOpt)  elOpt.textContent  = String(opt);
    if (elCL)   elCL.textContent   = String(cl);
    if (elHM)   elHM.textContent   = String(hm);
    if (elMM)   elMM.textContent   = String(mm);
    if (elTalk) elTalk.textContent = String(talk);
  };

  const coerce7 = (arr) => {
    const a = Array.isArray(arr) ? arr.map((x) => Number(x) || 0) : Array(7).fill(0);
    if (a.length === 7) return a;
    const out = Array(7).fill(0);
    for (let i = 0; i < Math.min(7, a.length); i++) out[i] = a[i];
    return out;
  };

  // Accept either API-provided trend arrays or compute from history
  const renderCharts = (trend = {}, history = []) => {
    const haveTrend = (k) => Array.isArray(trend?.[k]) && trend[k].length;

    const makeLineCfg = (label, arr, c1, c2) => (ctx) => ({
      type: "line",
      data: { labels: dayLabels, datasets: [{
        label, data: coerce7(arr), borderColor: c1, backgroundColor: gradient(ctx, c1, c2),
        fill: true, tension: 0.4, pointRadius: 3, pointHoverRadius: 5,
      }]},
      options: baseOptions,
    });

    const makeBarCfg = (label, arr, c1, c2) => (ctx) => ({
      type: "bar",
      data: { labels: dayLabels, datasets: [{
        label, data: coerce7(arr), borderColor: c1, backgroundColor: gradient(ctx, c1, c2),
      }]},
      options: baseOptions,
    });

    const normHist = normalizeHistory(history);

    const sOpt  = haveTrend("optimizations") ? trend.optimizations : seriesFromHistory(normHist, "optimization");
    const sCL   = haveTrend("coverletters")  ? trend.coverletters  : seriesFromHistory(normHist, "coverletter");
    const sHM   = haveTrend("superhuman")    ? trend.superhuman    : seriesFromHistory(normHist, "humanize");
    const sMM   = haveTrend("mastermind")    ? trend.mastermind    : seriesFromHistory(normHist, "mastermind");
    const sTalk = haveTrend("talk")          ? trend.talk          : seriesFromHistory(normHist, "talk");

    if (!window.Chart) return; // graceful if Chart.js missing

    cvs.optimizations && ensureChart(cvs.optimizations, makeLineCfg("Optimizations", sOpt,  "rgba(91,208,255,0.85)", "rgba(159,120,255,0.30)"));
    cvs.coverletters  && ensureChart(cvs.coverletters,  makeBarCfg ("Cover Letters", sCL, "rgba(255,184,77,0.85)", "rgba(255,107,107,0.30)"));
    cvs.superhuman    && ensureChart(cvs.superhuman,    makeLineCfg("Humanize",     sHM, "rgba(121,255,207,0.85)", "rgba(91,208,255,0.30)"));
    cvs.mastermind    && ensureChart(cvs.mastermind,    makeBarCfg ("MasterMind",   sMM, "rgba(173,91,255,0.85)", "rgba(91,208,255,0.30)"));
    cvs.talk          && ensureChart(cvs.talk,          makeLineCfg("Talk",         sTalk,"rgba(255,107,107,0.85)","rgba(255,184,77,0.30)"));
  };

  /* ------------------------------------------------------------
     🛰️ Fetch + Fallback
  ------------------------------------------------------------ */
  function shapeApiData(raw) {
    // Accept multiple server shapes; prefer explicit fields
    const summary = raw.summary || raw.totals || raw.counts || raw.metrics || {};
    const trend   = raw.trend   || raw.series  || {};
    const history = Array.isArray(raw.history) ? raw.history : [];
    return { summary, trend, history };
  }

  async function fetchDashboard() {
    if (statusBar) statusBar.textContent = "Fetching…";
    const urls = [
      `${apiBase}/api/dashboard`,
      `${apiBase}/api/dashboard/summary`
    ];

    for (const url of urls) {
      try {
        const data = await fetchJSON(url, 10000);
        const shaped = shapeApiData(data);
        // Defensive shaping
        shaped.summary ||= {};
        shaped.trend   ||= {};
        shaped.history = Array.isArray(shaped.history) ? shaped.history : [];
        saveCache(shaped);
        if (statusBar) statusBar.textContent = `Live ✓  (${apiBase.replace(/^https?:\/\//,"")})`;
        debug("Dashboard fetch OK", { endpoint: url, history: shaped.history.length });
        return shaped;
      } catch (e) {
        debug("Dashboard endpoint failed", { endpoint: url, error: e?.message });
      }
    }

    // If both endpoints failed:
    console.warn("[ASTRA] Dashboard fetch error — using cache/local");
    toast("⚠️ Backend unreachable — showing cached data if available.");
    if (statusBar) statusBar.textContent = "Offline (cached)";

    const cached = loadCache();
    if (cached) return cached;

    // Final fallback: synthesize from localStorage history
    const history = normalizeHistory(getLocalHistory());
    const counts = countsFromHistory(history);
    const synth = {
      summary: {
        optimize_runs:    counts.optimization,
        coverletters:     counts.coverletter,
        superhuman_calls: counts.humanize,
        mastermind_chats: counts.mastermind,
        talk_queries:     counts.talk,
      },
      trend: {
        optimizations: seriesFromHistory(history, "optimization"),
        coverletters:  seriesFromHistory(history, "coverletter"),
        superhuman:    seriesFromHistory(history, "humanize"),
        mastermind:    seriesFromHistory(history, "mastermind"),
        talk:          seriesFromHistory(history, "talk"),
      },
      history,
    };
    return synth;
  }

  /* ------------------------------------------------------------
     🔄 Refresh (optional button)
  ------------------------------------------------------------ */
  refreshBtn?.addEventListener("click", async () => {
    refreshBtn.disabled = true;
    toast("🔄 Refreshing dashboard…");
    try {
      const data = await fetchDashboard();
      renderMetrics(data.summary || {}, data.history || []);
      renderCharts(data.trend || {}, data.history || []);
    } finally {
      refreshBtn.disabled = false;
    }
  });

  /* ------------------------------------------------------------
     🚀 Init: render from cache (if any), then fetch fresh
  ------------------------------------------------------------ */
  (async () => {
    const cached = loadCache();
    if (cached) {
      renderMetrics(cached.summary || {}, cached.history || []);
      renderCharts(cached.trend || {}, cached.history || []);
      if (statusBar) statusBar.textContent = "Loaded from cache";
    }

    const fresh = await fetchDashboard();
    renderMetrics(fresh.summary || {}, fresh.history || []);
    renderCharts(fresh.trend || {}, fresh.history || []);
  })();

  /* ------------------------------------------------------------
     🎨 Event Sync (optional)
  ------------------------------------------------------------ */
  window.addEventListener("hirex:theme-change", (e) =>
    debug("Dashboard theme changed", { theme: e.detail?.theme })
  );
  window.addEventListener("hirex:humanize-change", (e) =>
    debug("Dashboard humanize toggled", { on: e.detail?.on })
  );

  /* ------------------------------------------------------------
     🧹 Cleanup
  ------------------------------------------------------------ */
  window.addEventListener("beforeunload", () => {
    Object.values(activeCharts).forEach((ch) => ch?.destroy?.());
  });

  /* ------------------------------------------------------------
     ✅ Init Log
  ------------------------------------------------------------ */
  console.log(
    `%c📊 ${APP} dashboard.js initialized — ${APP_VERSION}`,
    "background:#5bd0ff;color:#00121e;padding:4px 8px;border-radius:4px;font-weight:bold;"
  );
  debug("DASHBOARD PAGE LOADED", { app: APP, version: APP_VERSION, apiBase });
});
