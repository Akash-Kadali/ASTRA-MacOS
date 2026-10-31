/* ============================================================
   HIREX • coverletter.js (v1.0.0 — JD-Specific Cover Letter)
   ------------------------------------------------------------
   • Reads selected JD+Resume context from localStorage
   • Fetches previous jobs from backend /api/context/list
   • Submits FormData to /api/coverletter (FastAPI contract)
   • Optional "Humanize BODY" via global toggle/state
   • Renders returned LaTeX + PDF (tabs + iframe)
   • PDF/LaTeX toolbar (copy / download)
   • Timeout (3m) + Cancel + robust errors
   • Safe base64 decode, no duplicate listeners
   • Stores server memory id/path for quick reopen
   • Auto-populates <select> from backend + localStorage
   • Restores last cover letter on load via /api/context/get?latest=true
   Author: Sri Akash Kadali
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  const APP_NAME = "HIREX";
  const APP_VERSION = "v1.0.0";

  /* ------------------------------------------------------------
     🌐 Elements (optional-safe)
  ------------------------------------------------------------ */
  const selectEl    = document.getElementById("historySelect");
  const genBtn      = document.getElementById("generate_btn");
  const statusBadge = document.getElementById("meta_badge");

  const panePdf     = document.getElementById("pane_pdf");
  const pdfFrame    = document.getElementById("pdf_frame");

  const paneTex     = document.getElementById("pane_tex");
  const texOut      = document.getElementById("cl-tex-output");

  const paneBody    = document.getElementById("pane_body");
  const bodyOut     = document.getElementById("body_output");

  /* ------------------------------------------------------------
     🧠 Runtime helpers
  ------------------------------------------------------------ */
  const RT = (window.HIREX ?? window.ASTRA) || {};
  const debug = (msg, data) => (typeof RT.debugLog === "function" ? RT.debugLog(msg, data) : void 0);

  const getApiBase = () => {
    try { if (typeof RT.getApiBase === "function") return RT.getApiBase(); } catch {}
    if (["127.0.0.1", "localhost"].includes(location.hostname)) return "http://127.0.0.1:8000";
    return location.origin;
  };
  const apiBase = getApiBase();

  const toast = (msg, t = 3000) => (RT.toast ? RT.toast(msg, t) : console.log(`[HIREX Toast] ${msg}`));
  const nowStamp = () => new Date().toISOString().replace(/[:.]/g, "-");
  const sanitize = (name) => String(name || "file").replace(/[\\/:*?"<>|]+/g, "_").trim() || "file";

  const safeAtob = (b64) => {
    try {
      const s = String(b64 || "").trim();
      const base = s.includes("base64,") ? s.split("base64,")[1] : s;
      // Handle URL-safe base64 + stray whitespace/newlines
      const norm = base.replace(/-/g, "+").replace(/_/g, "/").replace(/\s+/g, "");
      return atob(norm);
    } catch {
      return "";
    }
  };

  const b64ToBlob = (b64, mime = "application/pdf") => {
    const bin = safeAtob(b64);
    if (!bin) return null;
    return new Blob([Uint8Array.from(bin, (c) => c.charCodeAt(0))], { type: mime });
  };

  const downloadText = (name, text) => {
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement("a"), { href: url, download: name });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 500);
  };

  const downloadBlob = (name, blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const a = Object.assign(document.createElement("a"), { href: url, download: name });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 500);
  };

  // Default ON when unset (matches backend default True)
  const getHumanizeState = () => {
    try { if (typeof RT.getHumanizeState === "function") return !!RT.getHumanizeState(); } catch {}
    // support both new and legacy keys
    const v = localStorage.getItem("hirex_use_humanize");
    if (v === null) return true;                            // ← default ON
    if (v === "true" || v === "false") return v === "true";
    return (localStorage.getItem("hirex-use-humanize") ?? "on") === "on";
  };

  const getTone = () => {
    try { if (typeof RT.getTone === "function") return RT.getTone() || "balanced"; } catch {}
    const el = document.getElementById("toneSelect");
    return (el && el.value) || "balanced";
  };

  const getLengthPref = () => {
    try { if (typeof RT.getCoverLetterLength === "function") return RT.getCoverLetterLength() || "standard"; } catch {}
    const el = document.getElementById("lengthSelect");
    return (el && el.value) || "standard";
  };

  /* ------------------------------------------------------------
     📦 Context helpers
  ------------------------------------------------------------ */
  const readHistory = () => {
    try { return JSON.parse(localStorage.getItem("hirex_history") || "[]"); }
    catch { return []; }
  };

  const pickFromHistory = (index) => {
    const history = readHistory();
    return history?.[index] || null;
  };

  const getFallbackCtx = () => ({
    jd:        localStorage.getItem("hirex_jd_text") || "",
    resumeTex: localStorage.getItem("hirex_tex") || "",
    company:   localStorage.getItem("hirex_company") || "Company",
    role:      localStorage.getItem("hirex_role") || "Role",
  });

  const getSelectedContext = () => {
    const idx = Number(selectEl?.value ?? 0);
    const item = pickFromHistory(idx);
    const fallback = getFallbackCtx();

    const selected = (() => {
      try { return JSON.parse(localStorage.getItem("hirex_selected_cl") || "null"); }
      catch { return null; }
    })();

    return {
      company:   selected?.company    ?? item?.company    ?? fallback.company,
      role:      selected?.role       ?? item?.role       ?? fallback.role,
      jd:        selected?.jd_text    ?? item?.jd_text    ?? fallback.jd,
      resumeTex: selected?.resume_tex ?? item?.resume_tex ?? fallback.resumeTex,
    };
  };

  /* ------------------------------------------------------------
     🔄 Fetch contexts from backend API (NEW - fixes Issue #2)
  ------------------------------------------------------------ */
  const fetchContextListFromBackend = async (limit = 50) => {
    try {
      const res = await fetch(`${apiBase}/api/context/list?limit=${limit}`, { 
        credentials: "same-origin" 
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return Array.isArray(data.items) ? data.items : [];
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

  /* ------------------------------------------------------------
     📋 Populate select from backend + localStorage (FIXED)
  ------------------------------------------------------------ */
  const populateSelectIfEmpty = async () => {
    if (!selectEl) return;
    
    // Show loading state
    selectEl.innerHTML = "<option disabled selected>Loading jobs...</option>";
    if (genBtn) genBtn.disabled = true;

    // Collect items from both sources
    const localHistory = readHistory();
    const backendItems = await fetchContextListFromBackend(50);
    
    // Merge and deduplicate (prefer backend for same company__role)
    const seen = new Set();
    const allItems = [];
    
    // Add backend items first (they're more authoritative)
    for (const item of backendItems) {
      const key = `${(item.company || "").toLowerCase()}__${(item.role || "").toLowerCase()}`;
      if (!seen.has(key) && (item.company || item.role)) {
        seen.add(key);
        allItems.push({
          ...item,
          source: 'backend',
          key: item.key || key,
        });
      }
    }
    
    // Add local items that aren't in backend
    for (let i = 0; i < localHistory.length; i++) {
      const item = localHistory[i];
      const key = `${(item.company || "").toLowerCase()}__${(item.role || "").toLowerCase()}`;
      if (!seen.has(key) && (item.company || item.role)) {
        seen.add(key);
        allItems.push({
          ...item,
          source: 'local',
          localIndex: i,
          key: key,
        });
      }
    }

    if (!allItems.length) {
      selectEl.innerHTML = "<option disabled selected>No saved resumes</option>";
      if (genBtn) genBtn.disabled = true;
      return;
    }

    // Build options (newest first based on timestamp if available)
    allItems.sort((a, b) => {
      const tsA = new Date(a.timestamp || a.updated_at || 0).getTime();
      const tsB = new Date(b.timestamp || b.updated_at || 0).getTime();
      return tsB - tsA;
    });

    selectEl.innerHTML = allItems
      .map((item, idx) => {
        const comp = sanitize(item.company ?? "—");
        const role = sanitize(item.role ?? "—");
        const sourceIcon = item.source === 'backend' ? '☁️' : '💾';
        // Store both source and key/index in data attributes via value encoding
        const valueData = JSON.stringify({ 
          source: item.source, 
          key: item.key, 
          localIndex: item.localIndex 
        });
        return `<option value='${valueData.replace(/'/g, "&#39;")}'>${sourceIcon} ${comp} — ${role}</option>`;
      })
      .join("");
    
    selectEl.selectedIndex = 0;
    if (genBtn) genBtn.disabled = false;

    // Handle selection changes
    const persistSelection = async () => {
      try {
        const valueData = JSON.parse(selectEl.value);
        
        if (valueData.source === 'backend') {
          // Fetch full context from backend
          const ctx = await fetchContextByKey(valueData.key);
          if (ctx) {
            const selected = {
              company: ctx.company || ctx.company_name || "",
              role: ctx.role || "",
              jd_text: ctx.jd_text || ctx.jd || "",
              resume_tex: ctx.humanized?.tex || ctx.humanized_tex || ctx.optimized?.tex || ctx.resume_tex || "",
            };
            localStorage.setItem("hirex_selected_cl", JSON.stringify(selected));
            debug("BACKEND_CONTEXT_SELECTED", selected);
          }
        } else {
          // Use local history
          const raw = localHistory[valueData.localIndex];
          if (raw) {
            localStorage.setItem("hirex_selected_cl", JSON.stringify(raw));
            debug("LOCAL_CONTEXT_SELECTED", raw);
          }
        }
      } catch (e) {
        console.warn("[HIREX] Selection persist failed:", e);
      }
    };
    
    // Persist initial selection
    persistSelection();
    
    // Listen for changes
    selectEl.addEventListener("change", persistSelection, { once: false });
  };

  // Initialize the select dropdown
  populateSelectIfEmpty();

  /* ------------------------------------------------------------
     🧱 Build FormData for POST — FastAPI contract
     jd_text, resume_tex, use_humanize, tone, length
  ------------------------------------------------------------ */
  const buildFormData = ({ jd, resumeTex, useHumanize, tone, length }) => {
    const fd = new FormData();
    fd.append("jd_text", jd || "");
    fd.append("resume_tex", (resumeTex || "").trim());
    fd.append("use_humanize", useHumanize ? "true" : "false");
    fd.append("tone", tone || "balanced");
    fd.append("length", length || "standard");
    return fd;
  };

  /* ------------------------------------------------------------
     🚀 POST /api/coverletter with cancel/timeout
  ------------------------------------------------------------ */
  const postCoverLetter = async (url, fd, controller) => {
    const res = await fetch(url, { method: "POST", body: fd, signal: controller.signal });
    let payload = null;
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    try {
      payload = ct.includes("json") ? await res.json() : await res.text();
    } catch {
      payload = null;
    }
    if (!res.ok) {
      const msg =
        (payload && (payload.detail || payload.error || payload.message)) ||
        (typeof payload === "string" && payload) ||
        `HTTP ${res.status}`;
      throw new Error(msg);
    }
    if (typeof payload === "string") {
      try { return JSON.parse(payload); } catch { throw new Error("Invalid JSON from backend."); }
    }
    return payload || {};
  };

  /* ------------------------------------------------------------
     🖼 Rendering helpers
  ------------------------------------------------------------ */
  const objectUrls = [];
  let lastPdfUrl = "";

  const urlFromPdfB64 = (b64) => {
    const blob = b64ToBlob(b64);
    if (!blob) return "";
    const url = URL.createObjectURL(blob);
    objectUrls.push(url);
    return url;
  };

  const ensurePdfToolbar = (containerEl) => {
    if (!containerEl) return null;
    let bar = containerEl.querySelector(".cl-toolbar");
    if (bar) return bar;

    bar = document.createElement("div");
    bar.className = "cl-toolbar";
    bar.style.display = "flex";
    bar.style.gap = ".5rem";
    bar.style.justifyContent = "flex-end";
    bar.style.margin = ".5rem 0 0";

    const btnPdf = document.createElement("button");
    btnPdf.id = "cl_dl_pdf";
    btnPdf.className = "btn accent";
    btnPdf.textContent = "⬇️ Download PDF";

    const btnTex = document.createElement("button");
    btnTex.id = "cl_dl_tex";
    btnTex.className = "btn";
    btnTex.textContent = "⬇️ Download .tex";

    const btnCopy = document.createElement("button");
    btnCopy.id = "cl_copy_tex";
    btnCopy.className = "btn";
    btnCopy.textContent = "📋 Copy LaTeX";

    bar.append(btnPdf, btnTex, btnCopy);
    containerEl.prepend(bar);
    return bar;
  };

  // ✅ Robust BODY extraction: supports anchored comments and multiline content
  const extractBodyFromLatex = (tex = "") => {
    if (!tex) return "";

    // 1) Prefer explicit BODY anchors
    const anchor = tex.match(
      /^\s*%[-\s]*BODY-START[-\s]*\s*$([\s\S]*?)^\s*%[-\s]*BODY-END[-\s]*\s*$/im
    );
    if (anchor) return anchor[1].trim();

    // 2) Fallback to document environment
    const docMatch = tex.match(/\\begin\{document\}([\s\S]*?)\\end\{document\}/i);
    if (docMatch) return docMatch[1].trim();

    // 3) Last resort: strip obvious preamble/comment lines
    return tex
      .split("\n")
      .filter((l) => !/^\\documentclass|^\\usepackage|^%/.test(l.trim()))
      .join("\n")
      .trim();
  };

  const renderOutputs = (data = {}) => {
    // Backend returns: company, role, tone, use_humanize, tex_string, pdf_base64 (+ paths/ids)
    const {
      tex_string = "",
      pdf_base64 = "",
      company = "",
      role = "",
    } = data;

    // LaTeX
    if (texOut) {
      texOut.textContent = tex_string.trim() || "% ⚠️ No LaTeX returned.\n% Try again.";
    }

    // Body (plain)
    if (bodyOut) {
      const body = extractBodyFromLatex(tex_string || "");
      bodyOut.textContent = (body || "(no body extracted)").trim();
    }

    // PDF -> iframe
    if (pdfFrame) {
      if (pdf_base64) {
        const url = urlFromPdfB64(pdf_base64);
        if (url) {
          // revoke previous to avoid leaks
          if (lastPdfUrl) { try { URL.revokeObjectURL(lastPdfUrl); } catch {} }
          lastPdfUrl = url;
          pdfFrame.src = `${url}#view=FitH`;
        }

        // Toolbar (use .onclick to avoid duplicate listeners)
        const bar = ensurePdfToolbar(panePdf || pdfFrame.parentElement);
        if (bar) {
          const comp  = sanitize(company || "Company").replace(/\s+/g, "_");
          const rl    = sanitize(role || "Role").replace(/\s+/g, "_");
          const stamp = nowStamp();
          // Match server naming for user download
          const pdfName = `Sri_${comp}_${rl}_Cover_Letter_${stamp}.pdf`;
          const texName = `Sri_${comp}_${rl}_Cover_Letter_${stamp}.tex`;

          const btnPdf  = bar.querySelector("#cl_dl_pdf");
          const btnTex  = bar.querySelector("#cl_dl_tex");
          const btnCopy = bar.querySelector("#cl_copy_tex");

          if (btnPdf) {
            btnPdf.onclick = async () => {
              try {
                const blob = await fetch(url).then((r) => r.blob());
                downloadBlob(pdfName, blob);
              } catch (e) {
                console.error("[HIREX] PDF download error:", e);
                toast("❌ Failed to download PDF.");
              }
            };
          }
          if (btnTex) {
            btnTex.onclick = () => {
              if (!(tex_string || "").trim()) return toast("⚠️ No LaTeX to download!");
              downloadText(texName, tex_string);
            };
          }
          if (btnCopy) {
            btnCopy.onclick = async () => {
              if (!(tex_string || "").trim()) return toast("⚠️ No LaTeX to copy!");
              try {
                await navigator.clipboard.writeText(tex_string);
                toast("✅ LaTeX copied!");
              } catch (e) {
                console.error("[HIREX] Clipboard error:", e);
                toast("⚠️ Clipboard permission denied.");
              }
            };
          }
        }

        // Show PDF tab after generation (if tabs exist)
        document.querySelector('.tab[data-tab="pdf"]')?.click();
      } else {
        pdfFrame.removeAttribute("src");
      }
    }
  };

  /* ------------------------------------------------------------
     💾 Cache results
  ------------------------------------------------------------ */
  const cacheResult = (data = {}, fallbackCtx = {}) => {
    try {
      const record = {
        id: Date.now(),
        company: data.company || fallbackCtx.company || "Company",
        role: data.role || fallbackCtx.role || "Role",
        fit_score: data.fit_score ?? null,
        type: "coverletter",
        timestamp: new Date().toISOString(),
        memory_id: data.memory_id || null,
        memory_path: data.memory_path || data.context_path || null, // ← accept context_path
        pdf_path: data.pdf_path || null,
      };
      const history = JSON.parse(localStorage.getItem("hirex_history") || "[]");
      history.push(record);
      localStorage.setItem("hirex_history", JSON.stringify(history));

      if (data.tex_string) localStorage.setItem("hirex_cl_tex", data.tex_string);
      if (data.pdf_base64) localStorage.setItem("hirex_cl_pdf", data.pdf_base64);
      localStorage.setItem("hirex_cl_company", record.company);
      localStorage.setItem("hirex_cl_role", record.role);
      localStorage.setItem("hirex_cl_version", APP_VERSION);
      if (record.memory_id) localStorage.setItem("hirex_cl_memory_id", record.memory_id);
      if (record.memory_path) localStorage.setItem("hirex_cl_memory_path", record.memory_path);
      if (record.pdf_path) localStorage.setItem("hirex_cl_pdf_path", record.pdf_path);
    } catch (err) {
      console.warn("[HIREX] Cache save failed:", err);
    }
  };

  /* ------------------------------------------------------------
     ✉️ Generate handler
  ------------------------------------------------------------ */
  const setStatus = (txt) => { if (statusBadge) statusBadge.textContent = txt; };

  const generateCoverLetter = async () => {
    const ctx = getSelectedContext();
    if (!ctx.jd?.trim()) {
      toast("⚠️ No Job Description found for the selected item.");
      setStatus("Idle");
      return;
    }

    const useHumanize = getHumanizeState();
    const tone = getTone();
    const length = getLengthPref();

    setStatus("Generating…");
    if (genBtn) genBtn.disabled = true;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 180000); // 3 minutes
    let cancelBtn;

    if (statusBadge && !document.getElementById("cl_cancel_btn")) {
      cancelBtn = document.createElement("button");
      cancelBtn.id = "cl_cancel_btn";
      cancelBtn.type = "button";
      cancelBtn.textContent = "❌ Cancel";
      cancelBtn.className = "btn";
      cancelBtn.style.marginLeft = "0.5rem";
      statusBadge.insertAdjacentElement("afterend", cancelBtn);
      cancelBtn.onclick = () => controller.abort();
    }

    const fd = buildFormData({
      jd: ctx.jd,
      resumeTex: ctx.resumeTex || "",
      useHumanize,
      tone,
      length,
    });

    const endpoint = `${apiBase}/api/coverletter`;

    try {
      const data = await postCoverLetter(endpoint, fd, controller);

      clearTimeout(timeout);
      if (cancelBtn) cancelBtn.remove();

      if (!data?.tex_string && !data?.pdf_base64) {
        throw new Error("Empty cover letter response from backend.");
      }

      renderOutputs(data);
      cacheResult(data, ctx);
      toast(`✅ Cover Letter ready for ${data.company || ctx.company}`);
      setStatus("Ready");
    } catch (err) {
      console.error("[HIREX] CoverLetter Error:", err);
      if (err.name === "AbortError") {
        toast("⚠️ Generation canceled or timed out (3 min).");
        setStatus("Canceled / Timed out");
      } else if (/Failed to fetch|NetworkError/i.test(err.message || "")) {
        toast("🌐 Network error — check FastAPI connection.");
        setStatus("Network error");
      } else {
        toast("❌ " + (err.message || "Unexpected error occurred."));
        setStatus("Error");
      }
    } finally {
      clearTimeout(timeout);
      if (document.getElementById("cl_cancel_btn")) document.getElementById("cl_cancel_btn").remove();
      if (genBtn) genBtn.disabled = false;
    }
  };

  /* ------------------------------------------------------------
     🔘 Wire up Generate button
  ------------------------------------------------------------ */
  genBtn?.addEventListener("click", generateCoverLetter);

  /* ------------------------------------------------------------
     ♻️ Restore last saved cover letter on load (defensive)
  ------------------------------------------------------------ */
  (async function loadLatestCoverLetter() {
    try {
      const res = await fetch(`${apiBase}/api/context/get?latest=true`);
      if (!res.ok) return;
      const data = await res.json();

      // Accept multiple shapes
      const pdfB64 =
        data.cover_letter_pdf_b64 ||
        data.pdf_base64 ||
        (data.cover_letter && data.cover_letter.pdf_b64);

      const tex =
        data.cover_letter_tex ||
        data.tex_string ||
        (data.cover_letter && data.cover_letter.tex);

      if (pdfB64 && pdfFrame) {
        const url = urlFromPdfB64(pdfB64);
        if (url) {
          if (lastPdfUrl) { try { URL.revokeObjectURL(lastPdfUrl); } catch {} }
          lastPdfUrl = url;
          pdfFrame.src = `${url}#view=FitH`;
          document.querySelector('.tab[data-tab="pdf"]')?.click();
        }
      }
      if (tex && texOut) {
        texOut.textContent = tex;
        if (bodyOut) {
          const body = extractBodyFromLatex(tex);
          bodyOut.textContent = (body || "(no body extracted)").trim();
        }
      }
    } catch {
      /* non-fatal */
    }
  })();

  /* ------------------------------------------------------------
     🧹 Revoke object URLs on unload
  ------------------------------------------------------------ */
  window.addEventListener("beforeunload", () => {
    objectUrls.forEach((u) => { try { URL.revokeObjectURL(u); } catch {} });
  });

  /* ------------------------------------------------------------
     ✅ Init log
  ------------------------------------------------------------ */
  console.log(
    `%c✉️ ${APP_NAME} coverletter.js initialized — ${APP_VERSION}`,
    "background:#5bd0ff;color:#00131c;padding:4px 8px;border-radius:4px;font-weight:bold;"
  );
  debug("COVER LETTER JS LOADED", { app: APP_NAME, version: APP_VERSION, apiBase });
});