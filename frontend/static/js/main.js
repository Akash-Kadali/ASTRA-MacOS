/* ============================================================
   ASTRA • main.js (v2.1.2 — Unified Frontend Integration)
   -------------------------------------------------------
   Handles:
   • JD submission → FastAPI /api/optimize/run (smart fallback)
   • Auto-retry with placeholder base_resume_tex on 422
   • Caches LaTeX / PDFs / JD-fit metrics in localStorage
   • Saves JD + (optimized/humanized) resume to /api/context/save
   • Safe Abort + Cancel UI with smooth toast feedback
   • Keyboard + reset shortcuts
   • Offline-aware (graceful error handling + no hard deps)
   • History record includes type: "optimization"
   Author: Sri Akash Kadali
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  const APP_NAME = "ASTRA";
  const APP_VERSION = "v2.1.2";

  /* ------------------------------------------------------------
     🔧 Elements
  ------------------------------------------------------------ */
  const form = document.getElementById("optimize-form");
  const jdInput = document.getElementById("jd");
  const hiddenHumanize = document.getElementById("use_humanize_state");

  /* ------------------------------------------------------------
     🧠 Utilities
  ------------------------------------------------------------ */
  const RT = (window.ASTRA ?? window.HIREX) || {};

  const getApiBase = () => {
    try { if (typeof RT.getApiBase === "function") return RT.getApiBase(); } catch {}
    return ["127.0.0.1", "localhost"].includes(location.hostname)
      ? "http://127.0.0.1:8000"
      : location.origin;
  };
  const apiBase = getApiBase();

  const toast   = (msg, t = 3000) => (RT.toast ? RT.toast(msg, t) : alert(msg));
  const debug   = (msg, data) => RT.debugLog?.(msg, data);
  const sanitize = (s) => String(s || "file").replace(/[\\/:*?"<>|]+/g, "_").trim() || "file";
  const truthy   = (v) => ["on", "true", "1", "yes"].includes(String(v ?? "").toLowerCase());

  const HTTP_TIMEOUT_MS = Number(window.ASTRA_HTTP_TIMEOUT_MS ?? 600000); // 10m default

  const getHumanize = () => {
    if (hiddenHumanize) return truthy(hiddenHumanize.value);
    const a = localStorage.getItem("hirex-use-humanize");
    const b = localStorage.getItem("hirex_use_humanize");
    if (a === null && b === null) return true; // default ON
    return truthy(a ?? b);
  };

  const getActiveModel = () =>
    localStorage.getItem("hirex_model") ||
    (typeof RT.getCurrentModel === "function" ? RT.getCurrentModel() : "") ||
    "gpt-4o-mini";

  const disableForm = (state) => {
    if (!form) return;
    Array.from(form.elements).forEach((el) => (el.disabled = state));
    form.style.opacity = state ? 0.6 : 1;
  };

  const progressFinish = () => document.dispatchEvent(new Event("hirex-finish"));

  /* ------------------------------------------------------------
     💾 Cache utilities
  ------------------------------------------------------------ */
  const persistResults = (data, useHumanize, jdText) => {
    try {
      const score =
        typeof data.rating_score === "number"
          ? data.rating_score
          : typeof data.coverage_ratio === "number"
          ? Math.round((data.coverage_ratio || 0) * 100)
          : null;

      const record = {
        id: Date.now(),
        company: data.company || data.company_name || "UnknownCompany",
        role: data.role || "UnknownRole",
        fit_score: score ?? null,
        timestamp: new Date().toISOString(),
        type: "optimization",
      };

      const history = JSON.parse(localStorage.getItem("hirex_history") || "[]");
      history.push(record);
      localStorage.setItem("hirex_history", JSON.stringify(history));

      const kv = {
        hirex_tex: data.tex_string || "",
        hirex_pdf: data.pdf_base64 || "",
        hirex_pdf_humanized: data.pdf_base64_humanized || "",
        hirex_company: record.company,
        hirex_role: record.role,
        hirex_fit_score: score ?? "n/a",
        hirex_use_humanize: useHumanize ? "true" : "false",
        hirex_timestamp: record.timestamp,
        hirex_version: APP_VERSION,
        hirex_jd_text: jdText || "",
      };
      Object.entries(kv).forEach(([k, v]) => localStorage.setItem(k, String(v)));
      localStorage.setItem("hirex-use-humanize", useHumanize ? "on" : "off");

      debug("✅ Cached optimization results", record);
    } catch (err) {
      console.error("[ASTRA] Cache error:", err);
    }
  };

  /* ------------------------------------------------------------
     🧠 Persist context on backend
  ------------------------------------------------------------ */
  const saveContextOnBackend = async (data, jdText) => {
    try {
      const fd = new FormData();
      const company = data.company || data.company_name || "";
      const role = data.role || "";
      const fit =
        typeof data.rating_score === "number"
          ? String(data.rating_score)
          : typeof data.coverage_ratio === "number"
          ? String(Math.round((data.coverage_ratio || 0) * 100))
          : "";

      const humanizedTex =
        (data.humanized && data.humanized.tex) ||
        data.humanized_tex ||
        data.tex_string_humanized ||
        data.tex_string ||
        "";

      fd.append("company", company);
      fd.append("role", role);
      fd.append("jd_text", jdText || "");
      fd.append("resume_tex", data.tex_string || "");
      fd.append("humanized_tex", humanizedTex);
      fd.append("pdf_base64", data.pdf_base64 || "");
      fd.append("pdf_base64_humanized", data.pdf_base64_humanized || "");
      fd.append("model", getActiveModel());
      fd.append("fit_score", fit);

      const res = await fetch(`${apiBase}/api/context/save`, {
        method: "POST",
        body: fd,
        credentials: "same-origin",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const resp = await res.json().catch(() => ({}));
      debug("🧷 Context saved", resp);
    } catch (e) {
      console.warn("[ASTRA] Context save skipped:", e);
    }
  };

  /* ------------------------------------------------------------
     📄 Placeholder base (for 422)
  ------------------------------------------------------------ */
  const addPlaceholderBase = (fd) => {
    const blob = new Blob(["% USE_SERVER_DEFAULT base_resume.tex\n"], { type: "text/plain" });
    fd.append("base_resume_tex", blob, "USE_SERVER_DEFAULT.tex");
    return fd;
  };

  /* ------------------------------------------------------------
     🧩 FormData builder
  ------------------------------------------------------------ */
  const buildFormData = (jdText, useHumanize) => {
    const fd = new FormData();
    fd.append("jd_text", jdText || "");
    fd.append("use_humanize", useHumanize ? "true" : "false");
    fd.append("latex_safe", "true");
    fd.append("model", getActiveModel());
    // alias for older servers
    fd.append("job_description", jdText || "");
    return fd;
  };

  /* ------------------------------------------------------------
     ⏱️ Robust fetch with per-request timeout + external cancel
  ------------------------------------------------------------ */
  async function fetchWithTimeout(url, options = {}, ms = HTTP_TIMEOUT_MS, externalSignal) {
    const ctrl = new AbortController();

    const onExternalAbort = () => ctrl.abort(new DOMException("UserCancel", "AbortError"));
    if (externalSignal) {
      if (externalSignal.aborted) onExternalAbort();
      else externalSignal.addEventListener("abort", onExternalAbort, { once: true });
    }

    const t = setTimeout(() => ctrl.abort(new DOMException("Timeout", "AbortError")), ms);

    try {
      const res = await fetch(url, {
        ...options,
        headers: { Accept: "application/json", ...(options.headers || {}) },
        signal: ctrl.signal,
        credentials: "same-origin",
        keepalive: true,
      });
      return res;
    } finally {
      clearTimeout(t);
      if (externalSignal) externalSignal.removeEventListener("abort", onExternalAbort);
    }
  }

  /* ------------------------------------------------------------
     🌐 Optimize API discovery + fallback
  ------------------------------------------------------------ */
  const CANDIDATE_PATHS = [
    "/api/optimize/run",   // put your real endpoint first
    "/api/optimize",
    "/api/optimize/submit",
    "/api/optimize/jd",
    "/optimize",
  ];

  const postOptimize = async (url, fd, externalSignal) => {
    const res = await fetchWithTimeout(
      url,
      { method: "POST", body: fd },
      HTTP_TIMEOUT_MS,
      externalSignal
    );
    const text = await res.text().catch(() => "");
    if (!res.ok) {
      const err = new Error(text || `HTTP ${res.status}`);
      err.status = res.status;
      throw err;
    }
    try {
      return JSON.parse(text);
    } catch {
      throw new Error("Invalid JSON response from backend.");
    }
  };

  const discoverOptimizePath = async () => {
    try {
      const spec = await fetchWithTimeout(`${apiBase}/openapi.json`, {}, 15000).then(r => r.json());
      const paths = Object.keys(spec.paths || {});
      const hitsApi = paths.filter(p => /optimiz|resume/i.test(p) && p.startsWith("/api/"));
      const hitsOther = paths.filter(p => /optimiz|resume/i.test(p) && !p.startsWith("/api/"));
      return [...hitsApi, ...hitsOther][0] || null;
    } catch {
      return null;
    }
  };

  const tryOnePath = async (path, jd, useHumanize, externalSignal) => {
    let fd = buildFormData(jd, useHumanize);
    try {
      return await postOptimize(`${apiBase}${path}`, fd, externalSignal);
    } catch (err) {
      const msg = String(err?.message || "").toLowerCase();

      // mark types for the outer loop
      if (err?.name === "AbortError") err._isAbort = true;
      if (/timeout/.test(msg)) err._isTimeout = true;

      // 404 / 405 → not here; let the caller try next without noisy logs
      if (err.status === 404 || err.status === 405) throw err;

      // 422 requiring base_resume_tex → retry once with placeholder
      if ((err.status === 422 || /422/.test(msg)) && /base_resume_tex/i.test(msg)) {
        debug("422 needs base_resume_tex — retrying", { path });
        fd = buildFormData(jd, useHumanize);
        addPlaceholderBase(fd);
        return await postOptimize(`${apiBase}${path}`, fd, externalSignal);
      }

      // Specific backend misconfig hint
      if (err.status === 500 && msg.includes("default base resume")) {
        const e = new Error("Backend missing default base resume (config.DEFAULT_BASE_RESUME).");
        e.status = 500;
        throw e;
      }

      throw err;
    }
  };

  const postOptimizeWithFallback = async (jd, useHumanize, externalSignal) => {
    const cached = localStorage.getItem("hirex_optimize_url");
    const discovered = await discoverOptimizePath();
    const order = [cached, discovered, ...CANDIDATE_PATHS]
      .filter(Boolean)
      .filter((p, i, a) => a.indexOf(p) === i);

    let lastErr;
    for (const path of order) {
      try {
        const data = await tryOnePath(path, jd, useHumanize, externalSignal);
        localStorage.setItem("hirex_optimize_url", path);
        debug("✅ Optimize path selected", { path });
        return data;
      } catch (err) {
        // User canceled → stop immediately
        if (err?._isAbort) throw err;

        // Timeout → try next path silently
        if (err?._isTimeout) {
          debug("⏱️ Timeout — trying next path", { path });
          lastErr = err;
          continue;
        }

        // 404 / 405 → endpoint not present; don't spam logs
        if (err?.status === 404 || err?.status === 405) {
          lastErr = err;
          continue;
        }

        // Known fatal misconfig → stop early
        if (err?.status === 500 && /base resume/i.test(String(err.message))) {
          lastErr = err;
          break;
        }

        // Other errors: log once, then try next
        debug("Path failed — trying next", { path, error: String(err?.message || err) });
        lastErr = err;
      }
    }
    const e = lastErr || new Error("No optimize endpoint responded successfully. Check /docs for available routes.");
    e.status = e.status || 404;
    throw e;
  };

  /* ------------------------------------------------------------
     🚀 Submit handler
  ------------------------------------------------------------ */
  form?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const jd = jdInput?.value?.trim();
    const useHumanize = getHumanize();
    if (!jd) return toast("⚠️ Paste the job description first.");

    disableForm(true);
    toast("⏳ Optimizing your resume…");
    debug("Submitting optimization", { useHumanize });

    const userCtrl = new AbortController();
    let canceled = false;

    const cancelBtn = document.createElement("button");
    cancelBtn.type = "button";
    cancelBtn.textContent = "❌ Cancel";
    cancelBtn.className = "cta-secondary";
    cancelBtn.style.marginTop = "1rem";
    form.appendChild(cancelBtn);
    cancelBtn.onclick = () => {
      canceled = true;
      userCtrl.abort(new DOMException("UserCancel", "AbortError"));
      toast("🛑 Optimization canceled.");
      cancelBtn.remove();
    };

    try {
      const data = await postOptimizeWithFallback(jd, useHumanize, userCtrl.signal);

      if (!data?.tex_string && !data?.pdf_base64)
        throw new Error("Empty or malformed response from backend.");

      persistResults(data, useHumanize, jd);
      saveContextOnBackend(data, jd);

      const company = sanitize(data.company || data.company_name || "Company");
      const role = sanitize(data.role || "Role");
      const score =
        typeof data.rating_score === "number"
          ? `${data.rating_score}/100`
          : typeof data.coverage_ratio === "number"
          ? `${Math.round((data.coverage_ratio || 0) * 100)}/100`
          : "n/a";

      toast(`✅ Optimized for ${company} (${role}) — JD Fit ${score}`);
      setTimeout(() => { if (!canceled) window.location.href = "/preview.html"; }, 1200);
      progressFinish();
    } catch (err) {
      console.error("[ASTRA] Optimization error:", err);
      const msg = String(err?.message || "");
      if (err?.name === "AbortError") {
        toast("⚠️ Request canceled or timed out. Server may still finish.");
      } else if (/Failed to fetch|NetworkError/i.test(msg)) {
        toast("🌐 Network error — check FastAPI connection.");
      } else if (/base resume/i.test(msg)) {
        toast("📄 Backend missing default base resume (config.DEFAULT_BASE_RESUME).");
      } else {
        toast("❌ " + msg);
      }
    } finally {
      disableForm(false);
      if (cancelBtn.isConnected) cancelBtn.remove();
    }
  });

  /* ------------------------------------------------------------
     🧹 Reset
  ------------------------------------------------------------ */
  form?.addEventListener("reset", () => {
    [
      "hirex_tex",
      "hirex_pdf",
      "hirex_pdf_humanized",
      "hirex_company",
      "hirex_role",
      "hirex_fit_score",
      "hirex_use_humanize",
      "hirex-use-humanize",
      "hirex_timestamp",
      "hirex_version",
      "hirex_jd_text",
    ].forEach((k) => localStorage.removeItem(k));
    toast("🧹 Cleared form and local cache.");
  });

  /* ------------------------------------------------------------
     💡 UX Enhancements
  ------------------------------------------------------------ */
  jdInput?.addEventListener("focus", () =>
    jdInput.scrollIntoView({ behavior: "smooth", block: "center" })
  );
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key.toLowerCase() === "enter") form?.requestSubmit();
  });

  /* ------------------------------------------------------------
     ✅ Init Log
  ------------------------------------------------------------ */
  console.log(
    `%c⚙️ ${APP_NAME} main.js initialized — ${APP_VERSION}`,
    "background:#5bd0ff;color:#00121e;padding:4px 8px;border-radius:4px;font-weight:bold;"
  );
  debug("MAIN PAGE LOADED", {
    app: APP_NAME,
    version: APP_VERSION,
    apiBase,
    hasHumanize: getHumanize(),
    model: getActiveModel(),
    origin: location.origin,
  });
});
