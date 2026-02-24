/* ============================================================
   HIREX • ui.js (v1.0.0 — Unified Global UI Layer)
   ------------------------------------------------------------
   Global UI behavior for all pages:
   • Smooth sidebar (desktop + mobile adaptive)
   • Theme persistence with system fallback + cross-tab sync
   • Humanize switch enhancer + global event sync
   • Active nav highlight + scroll-in animations
   • Global helpers (copy, download, filenames, timestamps, model)
   • Long-running API helpers (no accidental aborts)
   • Robust multi-tab + accessibility compatibility

   Author: Sri Akash Kadali
   ============================================================ */

/* Early theme apply to reduce white flash */
(() => {
  try {
    const saved = localStorage.getItem("hirex-theme");
    if (saved) {
      document.documentElement.setAttribute("data-theme", saved === "light" ? "light" : "dark");
    } else {
      const prefersDark = window.matchMedia?.("(prefers-color-scheme: dark)")?.matches;
      document.documentElement.setAttribute("data-theme", prefersDark ? "dark" : "light");
    }
  } catch {}
})();

document.addEventListener("DOMContentLoaded", () => {
  const APP_VERSION   = "v1.0.0";
  const THEME_KEY     = "hirex-theme";
  const HUMANIZE_KEY  = "hirex-use-humanize";   // "on" | "off"
  const HUMANIZE_BOOL = "hirex_use_humanize";   // "true" | "false" (legacy/alt)
  const MODEL_KEY     = "hirex_model";
  const currentPage   = (window.location.pathname.split("/").pop() || "index.html");
  const toastEl       = document.getElementById("toast");
  const html          = document.documentElement;
  const body          = document.body;

  /* ============================================================
     🧠 GLOBAL NAMESPACE + HELPERS
     ============================================================ */
  // Optional meta override for API base
  const metaApiBase = (() => {
    const m = document.querySelector('meta[name="hirex-api-base"]');
    return m?.getAttribute("content")?.trim() || "";
  })();

  window.HIREX = window.HIREX || {};
  Object.assign(window.HIREX, {
    version: APP_VERSION,
    __apiBase: metaApiBase || window.HIREX.__apiBase || "",

    /* ---------- API base ---------- */
    getApiBase: () => {
      try {
        if (typeof window.HIREX.__apiBase === "string" && window.HIREX.__apiBase) return window.HIREX.__apiBase;
      } catch {}
      const host = location.hostname;
      if (["127.0.0.1", "localhost", "0.0.0.0"].includes(host)) return "http://127.0.0.1:8000";
      return location.origin;
    },

    /* ---------- Notifications ---------- */
    toast: (msg, t = 2600) => {
      if (!toastEl) { console.log("[HIREX]", msg); return; }
      toastEl.setAttribute("role", "status");
      toastEl.setAttribute("aria-live", "polite");
      toastEl.textContent = msg;
      toastEl.classList.add("visible");
      clearTimeout(toastEl._timeout);
      toastEl._timeout = setTimeout(() => toastEl.classList.remove("visible"), t);
    },

    /* ---------- Logging (best-effort) ---------- */
    debugLog: (msg, data = {}) => {
      console.log("%c🟦 [HIREX]", "color:#5bd0ff;font-weight:bold;", msg, data);
      try {
        const base = (typeof window.HIREX.getApiBase === "function" ? window.HIREX.getApiBase() : location.origin);
        void fetch(`${base}/api/debug/log`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            msg,
            ...data,
            version: APP_VERSION,
            timestamp: new Date().toISOString(),
            page: currentPage,
          }),
          cache: "no-store",
          credentials: "same-origin",
        }).catch(() => {});
      } catch (err) {
        console.warn("[HIREX] Debug log failed:", err?.message || err);
      }
    },

    /* ---------- Humanize state (SSOT + legacy) ---------- */
    getHumanizeState: () => {
      const a = localStorage.getItem(HUMANIZE_KEY);          // "on" | "off"
      if (a === "on") return true;
      if (a === "off") return false;
      const b = localStorage.getItem(HUMANIZE_BOOL);         // "true" | "false"
      if (b === "true") return true;
      if (b === "false") return false;
      return true; // default ON
    },
    setHumanizeState: (on) => {
      const bool = !!on;
      localStorage.setItem(HUMANIZE_KEY,  bool ? "on"   : "off");
      localStorage.setItem(HUMANIZE_BOOL, bool ? "true" : "false");
      const evt = new CustomEvent("hirex:humanize-change", { detail: { on: bool } });
      window.dispatchEvent(evt);
      document.dispatchEvent(evt);
    },

    /* ---------- Model helpers ---------- */
    getCurrentModel: () => {
      const fromLS = localStorage.getItem(MODEL_KEY);
      if (fromLS) return fromLS;
      const sel = document.getElementById("model");
      return (sel && sel.value) || "gpt-4o-mini";
    },
    setCurrentModel: (model) => {
      if (typeof model === "string" && model.trim()) {
        const m = model.trim();
        localStorage.setItem(MODEL_KEY, m);
        const sel = document.getElementById("model");
        if (sel) sel.value = m;
        window.dispatchEvent(new CustomEvent("hirex:model-change", { detail: { model: m } }));
      }
    },

    /* ---------- File/clipboard helpers ---------- */
    sanitizeFilename: (name) =>
      String(name || "file").replace(/[\\/:*?"<>|]+/g, "_").trim() || "file",

    getTimestamp: () => new Date().toISOString().replace(/[:.]/g, "-"),

    copyToClipboard: async (text) => {
      try {
        await navigator.clipboard.writeText(String(text ?? ""));
        window.HIREX.toast?.("📋 Copied to clipboard!");
        return true;
      } catch {
        try {
          const ta = document.createElement("textarea");
          ta.value = String(text ?? "");
          ta.style.position = "fixed";
          ta.style.opacity = "0";
          document.body.appendChild(ta);
          ta.select();
          document.execCommand("copy");
          ta.remove();
          window.HIREX.toast?.("📋 Copied to clipboard!");
          return true;
        } catch {
          window.HIREX.toast?.("⚠️ Clipboard permission denied.");
          return false;
        }
      }
    },

    downloadTextFile: (filename, text) => {
      const name = window.HIREX.sanitizeFilename(filename);
      const blob = new Blob([String(text ?? "")], { type: "text/plain" });
      const url  = URL.createObjectURL(blob);
      const a    = Object.assign(document.createElement("a"), { href: url, download: name });
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 600);
    },

    downloadBlobFile: (filename, blob) => {
      if (!blob) return;
      const name = window.HIREX.sanitizeFilename(filename);
      const url  = URL.createObjectURL(blob);
      const a    = Object.assign(document.createElement("a"), { href: url, download: name });
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 600);
    },

    /* ---------- Smooth scroll utility ---------- */
    scrollTo: (selector, offset = 0) => {
      const el = document.querySelector(selector);
      if (!el) return false;
      const top = Math.max(0, el.getBoundingClientRect().top + window.scrollY - offset);
      window.scrollTo({ top, behavior: "smooth" });
      return true;
    },
  });

  // Back-compat namespace (other modules may look at window.ASTRA)
  window.ASTRA = window.ASTRA || window.HIREX;

  /* ============================================================
     🌗 THEME PERSISTENCE + SYNC
     ============================================================ */
  const themeMeta = document.querySelector('meta[name="theme-color"]');
  const setThemeMetaColor = (theme) => {
    if (!themeMeta) return;
    themeMeta.setAttribute("content", theme === "dark" ? "#0a1020" : "#ffffff");
  };
  const getSystemTheme = () =>
    (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches)
      ? "light" : "dark";

  const applyTheme = (theme, { persist = true, silent = false } = {}) => {
    const val = theme === "light" ? "light" : "dark";
    html.setAttribute("data-theme", val);
    setThemeMetaColor(val);
    if (persist) localStorage.setItem(THEME_KEY, val);
    window.dispatchEvent(new CustomEvent("hirex:theme-change", { detail: { theme: val } }));
    if (!silent) HIREX.toast(`🌗 ${val === "dark" ? "Dark" : "Light"} Mode`);
  };

  const savedTheme = localStorage.getItem(THEME_KEY);
  applyTheme(savedTheme || getSystemTheme(), { persist: !!savedTheme, silent: true });

  const themeBtn = document.getElementById("themeToggle") || document.querySelector("[data-theme-toggle]");
  themeBtn?.addEventListener("click", () => {
    const cur = html.getAttribute("data-theme") || "dark";
    applyTheme(cur === "dark" ? "light" : "dark");
  });

  const mqlDark = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  if (mqlDark) {
    const onSysChange = (e) => {
      if (!localStorage.getItem(THEME_KEY)) applyTheme(e.matches ? "dark" : "light", { persist: false });
    };
    if (typeof mqlDark.addEventListener === "function") mqlDark.addEventListener("change", onSysChange);
    else if (typeof mqlDark.addListener === "function") mqlDark.addListener(onSysChange);
  }

  window.addEventListener("storage", (e) => {
    if (e.key === THEME_KEY && e.newValue) applyTheme(e.newValue, { persist: false, silent: true });
  });

  /* ============================================================
     🧩 HUMANIZE TOGGLE (Single Source of Truth)
     ============================================================ */
  (function initHumanizeToggle() {
    const toggle = document.getElementById("humanize-toggle") || document.getElementById("humanize_toggle");
    const hidden = document.getElementById("use_humanize_state");
    if (!toggle) return;

    const persist = (on) => {
      localStorage.setItem(HUMANIZE_KEY,  on ? "on"   : "off");
      localStorage.setItem(HUMANIZE_BOOL, on ? "true" : "false"); // keep both in sync
    };

    const setState = (on, { silent = false } = {}) => {
      const isOn = !!on;
      toggle.classList.toggle("on", isOn);
      toggle.querySelector(".opt-off")?.classList.toggle("active", !isOn);
      toggle.querySelector(".opt-on")?.classList.toggle("active", isOn);
      if (hidden) hidden.value = isOn ? "on" : "off";
      persist(isOn);
      const evt = new CustomEvent("hirex:humanize-change", { detail: { on: isOn } });
      window.dispatchEvent(evt);
      document.dispatchEvent(evt);
      if (!silent) HIREX.toast(isOn ? "🧑‍💼 Humanize Enabled" : "⚙️ Optimize Enabled");
    };

    const startOn = (localStorage.getItem(HUMANIZE_KEY) ?? "on") === "on";
    setState(startOn, { silent: true });

    toggle.addEventListener("click", () => setState(!toggle.classList.contains("on")));

    window.addEventListener("storage", (e) => {
      if (e.key === HUMANIZE_KEY || e.key === HUMANIZE_BOOL) {
        const on = (localStorage.getItem(HUMANIZE_KEY) ?? "on") === "on";
        setState(on, { silent: true });
      }
    });
  })();

  /* ============================================================
     🧭 ACTIVE NAV LINK
     ============================================================ */
  document.querySelectorAll(".vnav a").forEach((a) => {
    const href = a.getAttribute("href") || "";
    const isRoot = currentPage === "" || currentPage === "/" || currentPage === "index.html";
    const isActive = href.endsWith(currentPage) || (isRoot && (href === "/" || href.endsWith("/index.html")));
    if (isActive) {
      a.classList.add("active-link");
      a.setAttribute("aria-current", "page");
    } else {
      a.classList.remove("active-link");
      a.removeAttribute("aria-current");
    }
  });

  /* ============================================================
     ✨ SCROLL-IN ANIMATIONS (respects reduced motion)
     ============================================================ */
  const prefersReducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
  const animatedEls = document.querySelectorAll("[data-anim], .anim");
  if (!prefersReducedMotion && animatedEls.length && "IntersectionObserver" in window) {
    const obs = new IntersectionObserver((entries, o) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-animated");
          o.unobserve(entry.target);
        }
      });
    }, { threshold: 0.15 });
    animatedEls.forEach((el) => obs.observe(el));
  } else {
    animatedEls.forEach((el) => el.classList.add("is-animated"));
  }

  /* ============================================================
     📱 RESPONSIVE SIDEBAR
     ============================================================ */
  const sidebar    = document.getElementById("sidebar");
  const menuToggle = document.getElementById("menuToggle") || document.querySelector("[data-menu-toggle]");

  if (sidebar && menuToggle) {
    menuToggle.setAttribute("aria-controls", "sidebar");
    menuToggle.setAttribute("aria-expanded", "false");

    const closeNav = () => {
      body.classList.remove("nav-open");
      menuToggle.setAttribute("aria-expanded", "false");
      sidebar.style.boxShadow = "none";
    };

    menuToggle.addEventListener("click", () => {
      const open = body.classList.toggle("nav-open");
      menuToggle.setAttribute("aria-expanded", String(open));
      sidebar.style.boxShadow = open ? "0 0 30px rgba(91,208,255,0.25)" : "none";
    });

    document.addEventListener("click", (e) => {
      if (body.classList.contains("nav-open") && !sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
        closeNav();
      }
    });

    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && body.classList.contains("nav-open")) closeNav();
    });
  }

  /* ============================================================
     ♿ ACCESSIBILITY ENHANCEMENTS
     ============================================================ */
  window.addEventListener("keydown", (e) => {
    if (e.key === "Tab") body.classList.add("user-is-tabbing");
  });
  window.addEventListener("mousedown", () => body.classList.remove("user-is-tabbing"));

  /* ============================================================
     🌐 ONLINE / OFFLINE FEEDBACK
     ============================================================ */
  window.addEventListener("online",  () => HIREX.toast("✅ Back Online"));
  window.addEventListener("offline", () => HIREX.toast("⚠️ Offline Mode Active"));

  /* ============================================================
     🔁 MODEL SELECT (persist + broadcast, optional)
     ============================================================ */
  const modelSel = document.getElementById("model");
  if (modelSel) {
    const savedModel = localStorage.getItem(MODEL_KEY);
    if (savedModel) modelSel.value = savedModel;
    modelSel.addEventListener("change", () => HIREX.setCurrentModel(modelSel.value));
    window.addEventListener("storage", (e) => {
      if (e.key === MODEL_KEY && e.newValue && modelSel.value !== e.newValue) {
        modelSel.value = e.newValue;
      }
    });
  }

  /* ============================================================
     🧩 LEGACY CLEANUP
     ============================================================ */
  (() => {
    const fileInput = document.getElementById("resume");
    if (fileInput) {
      fileInput.disabled = true;
      fileInput.style.display = "none";
      fileInput.setAttribute("aria-hidden", "true");
    }
  })();

  /* ============================================================
     ✨ CARD HOVER EFFECTS
     ============================================================ */
  document.querySelectorAll(".card").forEach((card) => {
    card.style.transition = "transform .25s ease, box-shadow .25s ease";
    card.addEventListener("mouseenter", () => {
      card.style.transform = "translateY(-4px)";
      card.style.boxShadow = "0 0 25px rgba(91,208,255,0.2)";
    });
    card.addEventListener("mouseleave", () => {
      card.style.transform = "translateY(0)";
      card.style.boxShadow = "none";
    });
  });

  /* ============================================================
     🧰 API HELPERS (fixes: avoid spurious AbortController cancels)
     ============================================================ */
  window.HIREX.api = window.HIREX.api || {};

  // Core POST that never aborts mid-flight. Optional soft timeout for UI only.
  window.HIREX.api.postLong = async function postLong(url, { body, headers, softTimeoutMs = 0 } = {}) {
    const controller = new AbortController(); // kept for compatibility; we don't call abort()
    const base = window.HIREX.getApiBase();
    const full = url.startsWith("http") ? url : `${base}${url}`;

    let softTimer = null;
    if (softTimeoutMs > 0) {
      // Soft timeout only for a heads-up toast; does NOT cancel the request.
      softTimer = setTimeout(() => {
        HIREX.toast("⏳ Still working…");
        HIREX.debugLog("long_request_soft_timeout", { url, softTimeoutMs });
      }, softTimeoutMs);
    }

    try {
      const res = await fetch(full, {
        method: "POST",
        body,
        headers,
        signal: controller.signal, // never aborted by us
        cache: "no-store",
        credentials: "same-origin",
        keepalive: false, // keepalive true limits payload; we want full PDFs
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        const err = new Error(`HTTP ${res.status} on ${url}`);
        err.status = res.status;
        err.body = text;
        throw err;
      }

      const ctype = res.headers.get("content-type") || "";
      if (ctype.includes("application/json")) return res.json();
      return res.text();
    } finally {
      if (softTimer) clearTimeout(softTimer);
    }
  };

  // Convenience: JSON post
  window.HIREX.api.postJSON = (url, data, opts = {}) =>
    window.HIREX.api.postLong(url, {
      body: JSON.stringify(data || {}),
      headers: { "Content-Type": "application/json" },
      ...opts
    });

  /* ============================================================
     🚀 OPTIMIZE SUBMISSION WIRING (safe defaults)
     - Tries /api/optimize/run (canonical), then /api/optimize, then /optimize.
     - No multi-path thrash. No AbortController cancel.
     ============================================================ */
  window.HIREX.optimize = window.HIREX.optimize || {};
  window.HIREX.optimize.submit = async function submitOptimize(formEl) {
    const useHumanize = HIREX.getHumanizeState();
    const fd = new FormData(formEl);
    fd.set("use_humanize", useHumanize ? "true" : "false"); // backend reads bool-ish
    // alias for older servers that expect both names
    if (!fd.has("jd_text") && fd.get("job_description")) fd.set("jd_text", fd.get("job_description"));
    if (!fd.has("job_description") && fd.get("jd_text")) fd.set("job_description", fd.get("jd_text"));

    const tryPaths = ["/api/optimize/run", "/api/optimize", "/optimize"]; // at most two fallbacks
    let lastErr = null;

    HIREX.debugLog("Submitting optimization", {
      useHumanize,
      version: APP_VERSION,
      origin: location.origin,
      page: currentPage,
    });

    for (const path of tryPaths) {
      try {
        const json = await HIREX.api.postLong(path, { body: fd, softTimeoutMs: 120000 }); // 2 min soft heads-up
        HIREX.debugLog("optimize_success", { path, coverage_ratio: json?.coverage_ratio });
        return json;
      } catch (e) {
        lastErr = e;
        HIREX.debugLog("optimize_path_failed", { path, error: e?.message || String(e) });
        // try next path once
      }
    }
    throw lastErr || new Error("optimize_failed");
  };

  // Optional: wire default form if present
  (function wireOptimizeFormIfPresent() {
    const form = document.getElementById("optimizeForm");
    const out  = document.getElementById("optimizeOutput"); // optional div/pre for messages
    if (!form) return;

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = form.querySelector("[type=submit]");
      btn && (btn.disabled = true);
      HIREX.toast("🔧 Optimizing…");
      HIREX.debugLog("optimize_submit_clicked");

      try {
        const result = await HIREX.optimize.submit(form);
        HIREX.toast("✅ Optimized");
        if (out) {
          out.textContent = JSON.stringify({
            company: result?.company || result?.company_name,
            role: result?.role,
            coverage: result?.coverage_ratio,
            optimized_file: result?.optimized?.filename || "",
            humanized_file: result?.humanized?.filename || "",
          }, null, 2);
        }
        window.dispatchEvent(new CustomEvent("hirex:optimize-success", { detail: result }));
      } catch (err) {
        const msg = err?.body || err?.message || String(err);
        HIREX.toast("❌ Optimization failed");
        HIREX.debugLog("optimize_failed", { error: msg });
        if (out) out.textContent = msg;
        window.dispatchEvent(new CustomEvent("hirex:optimize-fail", { detail: { error: msg } }));
      } finally {
        btn && (btn.disabled = false);
      }
    });
  })();

  /* ============================================================
     ✅ INIT LOG
     ============================================================ */
  console.log(
    "%c⚙️ HIREX ui.js initialized — v1.0.0",
    "background:#5bd0ff;color:#fff;padding:4px 8px;border-radius:4px;font-weight:bold;"
  );
  HIREX.debugLog("UI LOADED", {
    version: APP_VERSION,
    page: currentPage,
    origin: window.location.origin,
    theme: html.getAttribute("data-theme"),
    humanize_on: window.HIREX.getHumanizeState(),
    model: window.HIREX.getCurrentModel(),
  });
});
