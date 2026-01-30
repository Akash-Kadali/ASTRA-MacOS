/* ============================================================
   HIREX • preview.js (v2.1.2 — Memory-Enhanced Resume Viewer)
   ------------------------------------------------------------
   Fixes vs 2.1.0:
   • Clicking a "Recent Job" now ALWAYS loads that job’s assets.
   • Purges stale localStorage (PDF/TeX) before saving a new context.
   • Reads nested fields: humanized.tex/pdf_base64, optimized.tex/pdf_base64.
   • If a field is missing in a context, the old cache is cleared.
   • Auto-loads latest server context only when nothing local exists.
   • Properly revokes blob URLs on unload.
   Author: Sri Akash Kadali
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  const APP_VERSION = "v2.1.2";

  /* ------------------------------------------------------------
     🔧 DOM Elements
  ------------------------------------------------------------ */
  const texOutput      = document.getElementById("tex-output");
  const pdfContainer   = document.getElementById("pdf-container");
  const btnDownloadTex = document.getElementById("download-tex");
  const btnCopyTex     = document.getElementById("copy-tex");
  const fitCircle      = document.getElementById("fitCircle");
  const fitTierEl      = document.getElementById("fit-tier");
  const fitRoundsEl    = document.getElementById("fit-rounds");
  const historyList    = document.getElementById("history-list");

  /* ------------------------------------------------------------
     🧠 Runtime helpers
  ------------------------------------------------------------ */
  const RT = (window.ASTRA ?? window.HIREX) || {};
  const toast = (msg, t = 3000) => (RT.toast ? RT.toast(msg, t) : alert(msg));
  const debug = (msg, data) => RT.debugLog?.(msg, data);

  const getApiBase = () => {
    try { if (typeof RT.getApiBase === "function") return RT.getApiBase(); } catch {}
    if (["127.0.0.1", "localhost"].includes(location.hostname)) return "http://127.0.0.1:8000";
    return location.origin;
  };
  const apiBase = getApiBase();

  const getTS = () => new Date().toISOString().replace(/[:.]/g, "-");
  const sanitize = (s) => String(s || "file").replace(/[\\/:*?"<>|]+/g, "_").trim() || "file";

  // Base64 helpers: handle url-safe payloads and padding
  const normalizeB64 = (b64 = "") => {
    let x = (b64 || "").trim();
    const i = x.indexOf("base64,");
    if (i >= 0) x = x.slice(i + 7);
    x = x.replace(/[\r\n\s]/g, "").replace(/-/g, "+").replace(/_/g, "/");
    const pad = x.length % 4;
    return pad ? x + "=".repeat(4 - pad) : x;
  };
  const safeAtob = (b64) => { try { return atob(normalizeB64(b64)); } catch { return ""; } };
  const b64ToBlob = (b64, mime = "application/pdf") => {
    const bin = safeAtob(b64);
    if (!bin) return null;
    const len = bin.length;
    const out = new Uint8Array(len);
    for (let i = 0; i < len; i++) out[i] = bin.charCodeAt(i) & 0xff;
    return new Blob([out], { type: mime });
  };
  const downloadFile = (name, blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a   = Object.assign(document.createElement("a"), { href: url, download: name });
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 700);
  };
  const downloadText = (name, text) => downloadFile(name, new Blob([text], { type: "text/plain" }));

  /* ------------------------------------------------------------
     💾 Local Cache Snapshot
  ------------------------------------------------------------ */
  let texString         = localStorage.getItem("hirex_tex") || "";
  let pdfB64            = localStorage.getItem("hirex_pdf") || "";
  let pdfB64Humanized   = localStorage.getItem("hirex_pdf_humanized") || "";
  let companyRaw        = localStorage.getItem("hirex_company") || "Company";
  let roleRaw           = localStorage.getItem("hirex_role") || "Role";
  const cacheVersion    = localStorage.getItem("hirex_version") || APP_VERSION;

  let company = sanitize(companyRaw).replace(/\s+/g, "_");
  let role    = sanitize(roleRaw).replace(/\s+/g, "_");

  // Remove stale resume cache before saving a new selection
  const purgeResumeCache = () => {
    [
      "hirex_tex",
      "hirex_resume_plain",
      "hirex_resume_text",
      "hirex_pdf",
      "hirex_pdf_humanized",
      // optional fit cache—clear to avoid misleading carry-over
      "hirex_fit_score",
      "hirex_rating_history",
      "hirex_coverage_ratio",
    ].forEach((k) => localStorage.removeItem(k));
  };

  /* ------------------------------------------------------------
     🧭 History Loader (local) + Backend Contexts (dedup newest)
  ------------------------------------------------------------ */
  const keyFrom = (obj) => {
    const k = (obj?.key) ? String(obj.key)
      : `${obj?.company || ""}__${obj?.role || ""}`;
    return k.toLowerCase().replace(/\s+/g, "_");
  };

  const loadLocalHistory = () => {
    try { return JSON.parse(localStorage.getItem("hirex_history") || "[]"); }
    catch { return []; }
  };
  const history = loadLocalHistory();

  const dedupeLatest = (arr) => {
    const seen = new Set();
    const out = [];
    for (let i = arr.length - 1; i >= 0; i--) {
      const it = arr[i];
      const k = keyFrom(it);
      if (seen.has(k)) continue;
      seen.add(k);
      out.push({ ...it, __origIndex: i, __key: k });
    }
    return out; // newest-first
  };

  const renderHistoryListLocal = () => {
    if (!historyList) return;
    if (!Array.isArray(history) || !history.length) return false;

    const latest = dedupeLatest(history);
    historyList.innerHTML = latest.map((h) => `
      <li data-source="local" data-index="${h.__origIndex}" data-key="${h.__key}" class="history-item">
        <div class="history-entry">
          <strong>${sanitize(h.company || "—")}</strong><br/>
          <small>${sanitize(h.role || "—")}</small>
        </div>
      </li>
    `).join("");
    return true;
  };

  const loadLocalEntry = (index) => {
    const entry = history[index];
    if (!entry) return;

    // Purge first so missing fields don't retain old blobs/tex
    purgeResumeCache();

    localStorage.setItem("hirex_company", entry.company || "Company");
    localStorage.setItem("hirex_role", entry.role || "Role");

    // Older records may hold inline content; newer rely on server memory.
    if (entry.tex) localStorage.setItem("hirex_tex", entry.tex);
    if (entry.pdf) localStorage.setItem("hirex_pdf", entry.pdf);
    if (entry.pdf_humanized) localStorage.setItem("hirex_pdf_humanized", entry.pdf_humanized);

    // Persist minimal selection for other pages
    localStorage.setItem("hirex_selected_context", JSON.stringify({
      key: keyFrom(entry),
      company: entry.company || "",
      role: entry.role || "",
      jd_text: entry.jd_text || entry.jd || "",
      resume_tex: entry.tex || "",
      pdf_base64: entry.pdf || "",
      pdf_base64_humanized: entry.pdf_humanized || "",
    }));

    localStorage.setItem("hirex_version", APP_VERSION);
    toast(`📂 Loaded ${entry.company || "—"} — ${entry.role || "—"}`);
    window.location.reload();
  };

  // --- Backend context API (aligned to /api/context v2.x) ---
  const fetchContextList = async (limit = 50) => {
    try {
      const res = await fetch(`${apiBase}/api/context/list?limit=${limit}`, { credentials: "same-origin" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const items = Array.isArray(data.items) ? data.items : [];
      // Dedupe by key (prefer first as most recent)
      const seen = new Set();
      const uniq = [];
      for (const it of items) {
        const k = keyFrom(it);
        if (seen.has(k)) continue;
        seen.add(k);
        uniq.push({ ...it, __key: k });
      }
      return uniq;
    } catch (e) {
      console.warn("[HIREX] /api/context/list failed:", e);
      return [];
    }
  };

  const fetchContextByKey = async (key) => {
    try {
      const res = await fetch(`${apiBase}/api/context/get?key=${encodeURIComponent(key)}`, {
        credentials: "same-origin",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      console.warn("[HIREX] /api/context/get?key failed:", e);
      return null;
    }
  };

  const renderHistoryListBackend = async () => {
    if (!historyList) return;
    const items = await fetchContextList(50);
    if (!items.length) return false;

    const frag = document.createDocumentFragment();
    items.forEach((it) => {
      const li = document.createElement("li");
      li.className = "history-item";
      li.dataset.source = "backend";
      li.dataset.key = it.__key || keyFrom(it);
      li.innerHTML = `
        <div class="history-entry">
          <strong>${sanitize(it.company || "—")}</strong><br/>
          <small>${sanitize(it.role || "—")}</small>
        </div>`;
      frag.appendChild(li);
    });

    if (historyList.children.length) {
      const sep = document.createElement("li");
      sep.className = "history-sep";
      sep.innerHTML = `<div class="history-entry muted">— Server Memory —</div>`;
      historyList.appendChild(sep);
    }

    historyList.appendChild(frag);
    return true;
  };

  // Persist a minimal selected context for cross-pages (Talk, etc.)
  const persistSelectedContext = (ctx) => {
    const selected = {
      key: keyFrom(ctx),
      company: ctx.company || ctx.company_name || "",
      role: ctx.role || "",
      jd_text: ctx.jd_text || ctx.jd || "",
      resume_tex:
        (ctx.humanized?.tex) ||
        ctx.humanized_tex ||
        (ctx.optimized?.tex) ||
        ctx.resume_tex ||
        ctx.tex_string ||
        "",
      pdf_base64: ctx.pdf_base64 || ctx.optimized?.pdf_base64 || "",
      pdf_base64_humanized: ctx.pdf_base64_humanized || ctx.humanized?.pdf_base64 || "",
    };
    localStorage.setItem("hirex_selected_context", JSON.stringify(selected));
    return selected;
  };

  const loadBackendEntry = async (key) => {
    const ctx = await fetchContextByKey(key);
    if (!ctx) return toast("⚠️ Unable to load context from server.");

    // Purge first so missing fields don't retain old blobs/tex
    purgeResumeCache();

    // Normalize & save across app
    const companyName = ctx.company || ctx.company_name || "Company";
    const roleName    = ctx.role || "Role";

    const tex =
      (ctx.humanized?.tex) ||
      ctx.humanized_tex ||
      (ctx.optimized?.tex) ||
      ctx.resume_tex ||
      ctx.tex_string ||
      "";

    const pdfOpt = ctx.pdf_base64 || ctx.optimized?.pdf_base64 || "";
    const pdfHum = ctx.pdf_base64_humanized || ctx.humanized?.pdf_base64 || "";

    localStorage.setItem("hirex_company", companyName);
    localStorage.setItem("hirex_role", roleName);

    if (tex) localStorage.setItem("hirex_tex", tex); // keep cleared if absent
    if (pdfOpt) localStorage.setItem("hirex_pdf", pdfOpt);
    if (pdfHum) localStorage.setItem("hirex_pdf_humanized", pdfHum);

    // Fit score + history (coverage rounds) if available
    if (ctx.fit_score != null) {
      localStorage.setItem("hirex_fit_score", String(ctx.fit_score));
    } else if (ctx.coverage_ratio != null) {
      localStorage.setItem("hirex_fit_score", String(Math.round(Number(ctx.coverage_ratio) * 100)));
      localStorage.setItem("hirex_coverage_ratio", String(ctx.coverage_ratio));
    }
    if (Array.isArray(ctx.coverage_history)) {
      try { localStorage.setItem("hirex_rating_history", JSON.stringify(ctx.coverage_history)); } catch {}
    }

    if (ctx.jd_text) localStorage.setItem("hirex_jd_text", ctx.jd_text);

    // Persist selection for other pages
    persistSelectedContext(ctx);
    localStorage.setItem("hirex_version", APP_VERSION);

    toast(`☁️ Loaded ${companyName} — ${roleName} from server memory`);
    window.location.reload();
  };

  if (historyList) {
    const hadLocal = renderHistoryListLocal();
    renderHistoryListBackend().then((hadBackend) => {
      if (!hadLocal && !hadBackend) {
        historyList.innerHTML =
          "<li style='color:#888;padding:.5rem;'>No saved resumes yet.</li>";
      }
    });

    historyList.addEventListener("click", (e) => {
      const li = e.target.closest("li.history-item");
      if (!li) return;
      if (li.dataset.source === "local") {
        loadLocalEntry(Number(li.dataset.index));
      } else if (li.dataset.source === "backend") {
        loadBackendEntry(li.dataset.key);
      }
    });
  }

  /* ------------------------------------------------------------
     ⚠️ Version Notice
  ------------------------------------------------------------ */
  if (cacheVersion !== APP_VERSION) {
    console.warn(`[HIREX] Cache version mismatch: ${cacheVersion} ≠ ${APP_VERSION}`);
    toast("⚠️ Cache from a different version detected — re-optimize recommended.");
  }

  /* ------------------------------------------------------------
     🎯 JD Fit Gauge (align with main.js keys)
  ------------------------------------------------------------ */
  const computeFit = () => {
    let ratingScore = (() => {
      const raw = localStorage.getItem("hirex_fit_score");
      if (raw == null || raw === "n/a") return NaN;
      const n = Number(raw);
      return Number.isFinite(n) ? Math.round(n) : NaN;
    })();

    let ratingRounds = 0;
    try {
      const hist = JSON.parse(localStorage.getItem("hirex_rating_history") || "[]");
      if (Array.isArray(hist)) ratingRounds = hist.length;
      if (!Number.isFinite(ratingScore) && hist.length) {
        const last = hist.at(-1);
        if (typeof last?.coverage === "number") ratingScore = Math.round(last.coverage * 100);
      }
      if (!Number.isFinite(ratingScore)) {
        const cov = Number(localStorage.getItem("hirex_coverage_ratio"));
        if (Number.isFinite(cov)) ratingScore = Math.round(cov * 100);
      }
    } catch {}
    return { ratingScore, ratingRounds };
  };

  const renderFit = () => {
    const { ratingScore, ratingRounds } = computeFit();
    if (!fitCircle) return;

    const hasScore = Number.isFinite(ratingScore) && ratingScore >= 0;
    const tier = hasScore
      ? ratingScore >= 90 ? "Excellent"
      : ratingScore >= 75 ? "Strong"
      : ratingScore >= 60 ? "Moderate"
      : "Low"
      : "Awaiting Analysis…";

    fitCircle.dataset.score = hasScore ? String(ratingScore) : "--";
    fitCircle.style.borderColor = hasScore
      ? (ratingScore >= 90 ? "#6effa0"
        : ratingScore >= 75 ? "#5bd0ff"
        : ratingScore >= 60 ? "#ffc35b"
        : "#ff6b6b")
      : "rgba(255,255,255,0.25)";

    if (fitTierEl)   fitTierEl.textContent   = tier;
    if (fitRoundsEl) fitRoundsEl.textContent = ratingRounds || "--";
  };

  renderFit();

  /* ------------------------------------------------------------
     📜 Render LaTeX
  ------------------------------------------------------------ */
  if (texOutput) {
    texOutput.style.background  = "rgba(10,16,32,0.85)";
    texOutput.style.color       = "#dfe7ff";
    texOutput.style.whiteSpace  = "pre-wrap";
    texOutput.textContent = (texString || "").trim()
      ? texString
      : "% ⚠️ No optimized LaTeX found.\n% Please re-run optimization from Home or select a saved job.";
  }

  /* ------------------------------------------------------------
     📋 Copy / Download LaTeX
  ------------------------------------------------------------ */
  btnCopyTex?.addEventListener("click", async () => {
    if (!(texString || "").trim()) return toast("⚠️ No LaTeX to copy!");
    try {
      await navigator.clipboard.writeText(texString);
      toast("✅ LaTeX copied to clipboard!");
    } catch {
      toast("⚠️ Clipboard permission denied.");
    }
  });

  btnDownloadTex?.addEventListener("click", () => {
    if (!(texString || "").trim()) return toast("⚠️ No LaTeX to download!");
    const name = sanitize(`HIREX_Resume_${company}_${role}_${getTS()}.tex`);
    downloadText(name, texString);
    toast("⬇️ Downloading LaTeX file…");
  });

  /* ------------------------------------------------------------
     📄 PDF Renderer
  ------------------------------------------------------------ */
  const objectUrls = [];
  const makePdfUrl = (b64) => {
    const blob = b64ToBlob(b64);
    if (!blob) return "";
    const url = URL.createObjectURL(blob);
    objectUrls.push(url);
    return url;
  };

  const createPdfCard = (title, b64, suffix = "") => {
    const url = makePdfUrl(b64);
    if (!url) return "";
    const filename = sanitize(`HIREX_Resume_${company}_${role}${suffix}_${getTS()}.pdf`);
    return `
      <div class="pdf-card anim fade">
        <h3>${title}</h3>
        <div class="pdf-frame">
          <iframe src="${url}#view=FitH" loading="lazy" title="${title}"></iframe>
        </div>
        <div class="pdf-download">
          <button class="cta-primary" data-url="${url}" data-filename="${filename}">
            ⬇️ Download PDF
          </button>
        </div>
      </div>`;
  };

  const renderPdfs = () => {
    let html = "";
    if ((pdfB64 || "").trim())           html += createPdfCard("Optimized Resume", pdfB64);
    if ((pdfB64Humanized || "").trim())  html += createPdfCard("Humanized Resume (Tone-Refined)", pdfB64Humanized, "_Humanized");

    if (!html) {
      html = `<p class="muted" style="text-align:center;margin-top:2rem;">
        ⚠️ No PDF cached — optimize your resume first or pick a saved item on the left.
      </p>`;
    }

    if (pdfContainer) pdfContainer.innerHTML = html;
  };

  renderPdfs();

  pdfContainer?.addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-filename]");
    if (!btn) return;
    const { filename, url } = btn.dataset;
    try {
      const blob = await fetch(url).then((r) => r.blob());
      downloadFile(filename, blob);
    } catch {
      toast("❌ Failed to download PDF.");
    }
  });

  /* ------------------------------------------------------------
     ✨ Humanize Mode Highlight
  ------------------------------------------------------------ */
  const highlightActiveMode = (on) => {
    const cards = pdfContainer?.querySelectorAll(".pdf-card") || [];
    cards.forEach((c) => c.classList.remove("preferred"));
    cards.forEach((c) => {
      const isHuman = /Humanized/i.test(c.querySelector("h3")?.textContent || "");
      const prefer  = on ? isHuman : !isHuman;
      if (prefer) c.classList.add("preferred");
    });
  };

  const humanizeOn = (() => {
    if (typeof RT.getHumanizeState === "function") {
      try { return !!RT.getHumanizeState(); } catch {}
    }
    const storedBool = localStorage.getItem("hirex_use_humanize"); // "true"/"false"
    if (storedBool === "true" || storedBool === "false") return storedBool === "true";
    return localStorage.getItem("hirex-use-humanize") === "on";  // legacy "on"/"off"
  })();

  highlightActiveMode(humanizeOn);
  window.addEventListener("hirex:humanize-change", (e) =>
    highlightActiveMode(!!e.detail?.on)
  );

  /* ------------------------------------------------------------
     ☁️ Auto-load most recent server memory if nothing local
     ——— Uses ?latest=1 only when no local PDFs/TeX exist
  ------------------------------------------------------------ */
  (async () => {
    const nothingLocal =
      !(texString || "").trim() &&
      !(pdfB64 || "").trim() &&
      !(pdfB64Humanized || "").trim();

    if (!nothingLocal) return;

    try {
      const res = await fetch(`${apiBase}/api/context/get?latest=1`, { credentials: "same-origin" });
      if (res.ok) {
        const ctx = await res.json();

        // Build normalized assets first
        const texN =
          (ctx?.humanized?.tex) ||
          ctx?.humanized_tex ||
          (ctx?.optimized?.tex) ||
          ctx?.resume_tex ||
          ctx?.tex_string ||
          "";

        const pdfOptN = ctx?.pdf_base64 || ctx?.optimized?.pdf_base64 || "";
        const pdfHumN = ctx?.pdf_base64_humanized || ctx?.humanized?.pdf_base64 || "";

        const hasAssets = !!(texN || pdfOptN || pdfHumN);
        if (ctx && hasAssets) {
          // Purge first to avoid stale carry-over
          purgeResumeCache();

          companyRaw = ctx.company || ctx.company_name || companyRaw;
          roleRaw    = ctx.role || roleRaw;
          company    = sanitize(companyRaw).replace(/\s+/g, "_");
          role       = sanitize(roleRaw).replace(/\s+/g, "_");

          texString       = texN;
          pdfB64          = pdfOptN;
          pdfB64Humanized = pdfHumN;

          localStorage.setItem("hirex_company", companyRaw);
          localStorage.setItem("hirex_role", roleRaw);
          if (texString) localStorage.setItem("hirex_tex", texString);
          if (pdfB64) localStorage.setItem("hirex_pdf", pdfB64);
          if (pdfB64Humanized) localStorage.setItem("hirex_pdf_humanized", pdfB64Humanized);

          // Fit score + history
          if (ctx.fit_score != null) {
            localStorage.setItem("hirex_fit_score", String(ctx.fit_score));
          } else if (ctx.coverage_ratio != null) {
            localStorage.setItem("hirex_fit_score", String(Math.round(Number(ctx.coverage_ratio) * 100)));
            localStorage.setItem("hirex_coverage_ratio", String(ctx.coverage_ratio));
          }
          if (Array.isArray(ctx.coverage_history)) {
            try { localStorage.setItem("hirex_rating_history", JSON.stringify(ctx.coverage_history)); } catch {}
          }

          // Persist selection (helpful for Talk page)
          localStorage.setItem("hirex_selected_context", JSON.stringify({
            key: keyFrom(ctx),
            company: companyRaw,
            role: roleRaw,
            jd_text: ctx.jd_text || "",
            resume_tex: texString,
            pdf_base64: pdfB64,
            pdf_base64_humanized: pdfB64Humanized,
          }));

          localStorage.setItem("hirex_version", APP_VERSION);

          // refresh UI parts
          if (texOutput) texOutput.textContent = texString || texOutput.textContent;
          renderPdfs();
          renderFit();
          highlightActiveMode(humanizeOn);
          toast("☁️ Loaded latest resume from server memory.");
        }
      }
    } catch (e) {
      console.warn("[HIREX] Could not fetch latest context:", e);
    }
  })();

  /* ------------------------------------------------------------
     🧹 Cleanup + Init Log
  ------------------------------------------------------------ */
  window.addEventListener("beforeunload", () =>
    objectUrls.forEach((u) => URL.revokeObjectURL(u))
  );

  console.log(
    `%c📄 HIREX preview.js initialized — ${APP_VERSION}`,
    "background:#5bd0ff;color:#fff;padding:4px 8px;border-radius:4px;font-weight:bold;"
  );
  debug("PREVIEW PAGE LOADED", {
    version: APP_VERSION,
    company: companyRaw,
    role: roleRaw,
    historyCount: history.length,
    hasTex: !!(texString || "").trim(),
    hasPdf: !!(pdfB64 || "").trim(),
    hasPdfHumanized: !!(pdfB64Humanized || "").trim(),
  });
});
