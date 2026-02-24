/* ============================================================
   HIREX • util.js (v1.0.0 — Unified Utility Layer)
   ------------------------------------------------------------
   Shared helper utilities across HIREX front-end modules.

   • Global safe fetch patch (10m timeout, same-origin, abort-link)
   • Safe JSON/text/binary fetch with retry, timeout & backoff
   • Base64 ↔ Blob conversions, data:URL parsing, ObjectURL mgmt
   • Download helpers, clipboard, debounce/throttle, sleep
   • Theme & Humanize state helpers (persist + event emitters)
   • Storage wrappers, filename sanitizer, FormData builders
   • Integrated with HIREX logging + toast + cross-tab events

   Author: Sri Akash Kadali
   ============================================================ */

/* ============================================================
   🔧 Constants
   ============================================================ */
const UTIL_VERSION = "v1.0.0";
const DEFAULT_BASE_TEX_PATH = "data/samples/base_resume.tex";
const LS_KEYS = {
  THEME: "hirex-theme",
  HUMANIZE: "hirex-use-humanize", // "on" | "off" (legacy alt: hirex_use_humanize = "true"|"false")
  MODEL: "hirex_model",
};

/* ============================================================
   🌐 Global Safe Fetch Patch (prevents “signal is aborted”)
   - Default timeout 10 minutes for long-running jobs
   - Defaults credentials to 'same-origin'
   - Links caller AbortSignal to our internal timeout signal
   - Accepts non-standard init.timeoutMs from callers
   - No-op if already patched
   ============================================================ */
(function installGlobalFetchPatch() {
  if (window.__HIREX_PATCHED_FETCH__) return;
  const NATIVE_FETCH = window.fetch;
  const DEFAULT_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes

  function linkSignals(innerCtrl, outerSignal) {
    if (!outerSignal) return;
    if (outerSignal.aborted) {
      try { innerCtrl.abort(outerSignal.reason); } catch {}
      return;
    }
    outerSignal.addEventListener(
      "abort",
      () => { try { innerCtrl.abort(outerSignal.reason); } catch {} },
      { once: true }
    );
  }

  window.fetch = function patchedFetch(input, init = {}) {
    const timeoutMs = Number.isFinite(init.timeoutMs) ? init.timeoutMs : DEFAULT_TIMEOUT_MS;
    const ctrl = new AbortController();
    linkSignals(ctrl, init.signal);

    const opts = {
      credentials: init.credentials ?? "same-origin",
      ...init,
      signal: ctrl.signal,
    };

    const t = setTimeout(() => {
      try { ctrl.abort(new DOMException("Timeout", "AbortError")); } catch { try { ctrl.abort(); } catch {} }
    }, Math.max(0, timeoutMs));

    const done = (p) => p.finally(() => clearTimeout(t));
    return done(NATIVE_FETCH(input, opts));
  };

  window.__HIREX_PATCHED_FETCH__ = true;
  console.log("%c🔒 [HIREX] global fetch patched", "color:#5bd0ff");
})();

/* ============================================================
   🧰 Safe localStorage
   ============================================================ */
function lsGet(key) { try { return localStorage.getItem(key); } catch { return null; } }
function lsSet(key, val) { try { localStorage.setItem(key, val); return true; } catch { return false; } }
function lsRemove(key) { try { localStorage.removeItem(key); } catch {} }

/* ============================================================
   🌐 API Base Resolver (honor global if provided)
   ============================================================ */
function getApiBase() {
  try {
    if (window.HIREX && typeof window.HIREX.getApiBase === "function") {
      return window.HIREX.getApiBase();
    }
  } catch {}
  const host = location.hostname;
  return (["127.0.0.1", "localhost", "0.0.0.0"].includes(host)
    ? "http://127.0.0.1:8000"
    : location.origin);
}

/* ============================================================
   🌗 Theme Helpers
   ============================================================ */
function getSystemTheme() {
  try {
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  } catch { return "dark"; }
}
function getTheme() { return lsGet(LS_KEYS.THEME) || getSystemTheme(); }
function setTheme(theme) {
  const t = theme === "light" ? "light" : "dark";
  try {
    lsSet(LS_KEYS.THEME, t);
    document.documentElement.setAttribute("data-theme", t);
    // Update meta theme-color to avoid white flash on mobile
    const themeMeta = document.querySelector('meta[name="theme-color"]');
    if (themeMeta) themeMeta.setAttribute("content", t === "dark" ? "#0a1020" : "#ffffff");
    window.dispatchEvent(new CustomEvent("hirex:theme-change", { detail: { theme: t } }));
    window.HIREX?.debugLog?.("Theme changed", { theme: t });
  } catch (err) {
    console.warn("[HIREX util] Failed to set theme:", err);
  }
  return t;
}
function onThemeChange(handler) {
  const fn = (e) => handler?.(e.detail?.theme ?? getTheme());
  window.addEventListener("hirex:theme-change", fn);
  return () => window.removeEventListener("hirex:theme-change", fn);
}

/* ============================================================
   🧑‍💼 Humanize Helpers (default ON)
   ============================================================ */
function getHumanizeState() {
  // Support both new ("on"/"off") and legacy ("true"/"false") keys
  const a = lsGet(LS_KEYS.HUMANIZE);
  if (a === "on") return true;
  if (a === "off") return false;
  const b = lsGet("hirex_use_humanize");
  if (b === "true") return true;
  if (b === "false") return false;
  return true; // default ON
}
function setHumanizeState(on) {
  lsSet(LS_KEYS.HUMANIZE, on ? "on" : "off");
  lsSet("hirex_use_humanize", on ? "true" : "false"); // keep legacy key mirrored
  const evt = new CustomEvent("hirex:humanize-change", { detail: { on } });
  window.dispatchEvent(evt);
  document.dispatchEvent(evt);
  window.HIREX?.debugLog?.("Humanize state changed", { on });
  return on;
}
function onHumanizeChange(handler) {
  const fn = (e) => handler?.(!!e.detail?.on);
  window.addEventListener("hirex:humanize-change", fn);
  return () => window.removeEventListener("hirex:humanize-change", fn);
}

/* ============================================================
   🔌 Model Helpers
   ============================================================ */
function getCurrentModel() {
  const fromLS = lsGet(LS_KEYS.MODEL);
  if (fromLS) return fromLS;
  if (window.HIREX?.getCurrentModel) return window.HIREX.getCurrentModel();
  const sel = document.getElementById("model");
  return (sel && sel.value) || "gpt-4o-mini";
}
function setCurrentModel(model) {
  if (typeof model === "string" && model.trim()) {
    lsSet(LS_KEYS.MODEL, model.trim());
    const sel = document.getElementById("model");
    if (sel) sel.value = model.trim();
    window.dispatchEvent(new CustomEvent("hirex:model-change", { detail: { model: model.trim() } }));
  }
}

/* ============================================================
   🌍 Fetch Helpers (retry + timeout + exponential backoff)
   - Works with JSON bodies, FormData, Blobs, etc.
   - Same-origin credentials by default (matches FastAPI)
   - Respects caller-supplied AbortSignal (linked)
   - NOTE: Global fetch is already patched with 10m timeout.
   ============================================================ */
function _linkSignals(innerController, externalSignal) {
  if (externalSignal && typeof externalSignal.addEventListener === "function") {
    if (externalSignal.aborted) innerController.abort();
    else externalSignal.addEventListener("abort", () => innerController.abort(), { once: true });
  }
}

function _composeQuery(url, qs) {
  if (!qs || typeof qs !== "object") return url;
  const u = new URL(url, location.origin);
  Object.entries(qs).forEach(([k, v]) => {
    if (v === undefined || v === null) return;
    if (Array.isArray(v)) v.forEach(x => u.searchParams.append(k, x));
    else u.searchParams.set(k, String(v));
  });
  return u.toString();
}

async function _doFetch(url, options = {}, retries = 2) {
  const internalController = new AbortController();
  // Accept both options.timeoutMs and options.timeout (ms)
  const timeoutMs = Number.isFinite(options.timeoutMs)
    ? options.timeoutMs
    : (Number.isFinite(options.timeout) ? options.timeout : (10 * 60 * 1000)); // 10m

  // Independent timeout in case callers bypass the patch elsewhere
  const timeout = setTimeout(() => internalController.abort(), timeoutMs);
  _linkSignals(internalController, options.signal);

  const isJsonBody =
    options.body &&
    typeof options.body === "object" &&
    !(options.body instanceof FormData) &&
    !(options.body instanceof Blob) &&
    !(options.body instanceof ArrayBuffer);

  // Default Accept to */* unless caller overrides
  const headers = {
    ...(isJsonBody ? { "Content-Type": "application/json" } : {}),
    ...(options.headers || {}),
  };

  const reqUrl = options.qs ? _composeQuery(url, options.qs) : url;

  try {
    const res = await fetch(reqUrl, {
      credentials: options.credentials || "same-origin",
      ...options,
      headers,
      body: isJsonBody ? JSON.stringify(options.body) : options.body,
      signal: internalController.signal,
      timeoutMs, // read by our global fetch patch
    });
    clearTimeout(timeout);

    if (!res.ok) {
      const msg = await res.text().catch(() => "");
      const err = new Error(`HTTP ${res.status}: ${msg || "Unknown error"}`);
      err.status = res.status;
      window.HIREX?.debugLog?.("fetch ERROR", { url: reqUrl, status: res.status, msg });
      throw err;
    }
    return res;
  } catch (err) {
    clearTimeout(timeout);
    if (retries > 0) {
      const attempt = 3 - retries + 1; // 1..3
      const jitter = Math.floor(Math.random() * 300);
      const delay = 500 * attempt + 600 + jitter; // progressive backoff + jitter
      window.HIREX?.toast?.("⚠️ Network hiccup — retrying…");
      await new Promise((r) => setTimeout(r, delay));
      return _doFetch(url, options, retries - 1);
    }
    window.HIREX?.toast?.(`❌ Network error: ${err.message || err}`);
    window.HIREX?.debugLog?.("fetch FAIL", { url: reqUrl, err: err.message });
    throw err;
  }
}

async function fetchJSON(url, options = {}, retries = 2) {
  const res = await _doFetch(url, {
    ...options,
    headers: { Accept: "application/json", ...(options.headers || {}) },
  }, retries);

  const ct = (res.headers.get("content-type") || "").toLowerCase();
  if (ct.includes("application/json")) {
    try { return await res.json(); }
    catch { return {}; }
  }
  // Some backends return JSON as text/plain
  const text = await res.text().catch(() => "");
  try {
    const parsed = JSON.parse(text);
    window.HIREX?.debugLog?.("fetchJSON (text->json)", { url, keys: Object.keys(parsed || {}) });
    return parsed;
  } catch {
    window.HIREX?.debugLog?.("fetchJSON non-json", { url, length: text.length });
    return {};
  }
}

async function fetchText(url, options = {}, retries = 2) {
  const res = await _doFetch(url, {
    ...options,
    headers: { Accept: "text/plain, */*;q=0.1", ...(options.headers || {}) },
  }, retries);
  const text = await res.text().catch(() => "");
  window.HIREX?.debugLog?.("fetchText OK", { url, len: text.length });
  return text;
}

async function fetchBinary(url, options = {}, retries = 2) {
  const res = await _doFetch(url, {
    ...options,
    headers: { Accept: "*/*", ...(options.headers || {}) },
  }, retries);
  const buf = await res.arrayBuffer();
  window.HIREX?.debugLog?.("fetchBinary OK", { url, bytes: buf.byteLength });
  return buf;
}

function postJSON(url, data, options = {}, retries = 2) {
  return fetchJSON(url, { method: "POST", body: data, ...(options || {}) }, retries);
}

function withTimeout(promise, ms, signal) {
  const ctrl = new AbortController();
  _linkSignals(ctrl, signal);
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => {
      ctrl.abort();
      reject(new Error("Timeout"));
    }, ms);
    promise.then(
      (v) => { clearTimeout(t); resolve(v); },
      (e) => { clearTimeout(t); reject(e); }
    );
  });
}

/* ============================================================
   🧪 Base64 / Blob / data:URL Conversion
   ============================================================ */
function _normalizeBase64(b64 = "") {
  let x = String(b64 || "").trim();
  const i = x.indexOf("base64,");
  if (i >= 0) x = x.slice(i + 7);
  x = x.replace(/[\r\n\s]/g, "").replace(/-/g, "+").replace(/_/g, "/");
  const pad = x.length % 4;
  return pad ? x + "=".repeat(4 - pad) : x;
}

function parseDataURL(dataUrl = "") {
  if (!String(dataUrl).startsWith("data:")) return { mime: "", data: "" };
  const [head, data] = dataUrl.split(",", 2);
  const mime = (head.split(":")[1] || "").split(";")[0] || "";
  return { mime, data: data || "" };
}

function base64ToBlob(base64, mime = "application/octet-stream") {
  try {
    let raw = String(base64 || "");
    if (raw.startsWith("data:")) {
      const { mime: m, data } = parseDataURL(raw);
      mime = m || mime;
      raw = data;
    }
    const norm = _normalizeBase64(raw);
    const bin = atob(norm);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i) & 0xff;
    const blob = new Blob([bytes], { type: mime });
    window.HIREX?.debugLog?.("base64ToBlob OK", { size: blob.size, mime });
    return blob;
  } catch (e) {
    console.error("[HIREX] base64ToBlob error:", e);
    window.HIREX?.toast?.("⚠️ Failed to decode Base64 data.");
    return null;
  }
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      try {
        const str = reader.result?.toString() || "";
        resolve(str.split(",")[1] || "");
      } catch (err) { reject(err); }
    };
    reader.onerror = (e) => reject(e);
    reader.readAsDataURL(blob);
  });
}

function toDataURL(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result?.toString() || "");
    reader.onerror = (e) => reject(e);
    reader.readAsDataURL(blob);
  });
}

/* ============================================================
   🔗 ObjectURL Registry (auto-revoke helpers)
   ============================================================ */
const __objectUrls = new Set();
function makeObjectURL(blob) {
  try {
    const url = URL.createObjectURL(blob);
    __objectUrls.add(url);
    return url;
  } catch { return ""; }
}
function revokeObjectURL(url) {
  try { URL.revokeObjectURL(url); __objectUrls.delete(url); } catch {}
}
function revokeAllObjectURLs() {
  __objectUrls.forEach((u) => { try { URL.revokeObjectURL(u); } catch {} });
  __objectUrls.clear();
}
window.addEventListener("beforeunload", revokeAllObjectURLs);

/* ============================================================
   ⬇️ Download + Clipboard
   ============================================================ */
function sanitizeFilename(name, fallback = "file") {
  try {
    const n = String(name ?? "")
      .replace(/[\\/:*?"<>|]+/g, "_")
      .replace(/\s+/g, " ")
      .trim();
    return n || fallback;
  } catch { return fallback; }
}

function downloadFile(filename, data, mime = "application/octet-stream") {
  try {
    const name = (window.HIREX?.sanitizeFilename || sanitizeFilename)(filename);
    const blob =
      data instanceof Blob
        ? data
        : typeof data === "string"
        ? new Blob([data], { type: mime })
        : new Blob([JSON.stringify(data)], { type: mime });

    const url = makeObjectURL(blob);
    const a = Object.assign(document.createElement("a"), { href: url, download: name });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => revokeObjectURL(url), 600);
    window.HIREX?.toast?.(`⬇️ Downloading ${name}`);
  } catch (e) {
    console.error("[HIREX] downloadFile error:", e);
    window.HIREX?.toast?.("❌ Download failed.");
  }
}

function downloadTextFile(filename, text) { downloadFile(filename, text, "text/plain"); }

async function copyToClipboard(text) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(String(text ?? ""));
    } else {
      const ta = document.createElement("textarea");
      ta.value = String(text ?? "");
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    window.HIREX?.toast?.("📋 Copied to clipboard!");
    return true;
  } catch (e) {
    console.error("[HIREX] copy error:", e);
    window.HIREX?.toast?.("⚠️ Clipboard permission denied.");
    return false;
  }
}

/* ============================================================
   🕒 Misc Helpers
   ============================================================ */
function getTimestamp() {
  const ts = new Date().toISOString().replace(/[:.]/g, "-");
  window.HIREX?.debugLog?.("getTimestamp", { ts });
  return ts;
}

function debounce(fn, delay = 300, immediate = false) {
  let t;
  return (...args) => {
    const callNow = immediate && !t;
    clearTimeout(t);
    t = setTimeout(() => { t = null; if (!immediate) fn(...args); }, delay);
    if (callNow) fn(...args);
  };
}

function throttle(fn, interval = 200) {
  let last = 0, timer = null, lastArgs = null;
  return (...args) => {
    const now = Date.now();
    lastArgs = args;
    if (now - last >= interval) {
      last = now; fn(...args);
    } else if (!timer) {
      const wait = interval - (now - last);
      timer = setTimeout(() => { last = Date.now(); timer = null; fn(...lastArgs); }, wait);
    }
  };
}

function sleep(ms = 500) { return new Promise((r) => setTimeout(r, ms)); }

function formatBytes(bytes, decimals = 2) {
  if (!+bytes) return "0 Bytes";
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
}

/* ============================================================
   📝 FormData / Payload Builders (align with backend v1.0.0)
   ============================================================ */
function buildOptimizeFormData(jdText, useHumanize) {
  const fd = new FormData();
  fd.append("jd_text", jdText || "");
  fd.append("use_humanize", useHumanize ? "true" : "false");
  fd.append("latex_safe", "true");
  // Compatibility aliases (for older routes)
  fd.append("jd", jdText || "");
  fd.append("job_description", jdText || "");
  fd.append("humanize", useHumanize ? "true" : "false");
  fd.append("model", getCurrentModel());
  return fd;
}

function buildCoverLetterFormData({ jd_text = "", resume_tex = "", use_humanize = true, tone = "balanced", length = "standard" } = {}) {
  const fd = new FormData();
  fd.append("jd_text", jd_text || "");
  fd.append("resume_tex", (resume_tex || "").trim());
  fd.append("use_humanize", use_humanize ? "true" : "false");
  fd.append("tone", tone || "balanced");
  fd.append("length", length || "standard");
  fd.append("model", getCurrentModel());
  return fd;
}

function buildSuperhumanPayload(text, tone = "balanced", mode = "paragraph", latex_safe = true) {
  return {
    text: String(text || ""),
    tone: String(tone || "balanced").toLowerCase(),
    mode: String(mode || "paragraph").toLowerCase(), // "paragraph" | "resume" | "coverletter" | "sentence" | "custom"
    latex_safe: !!latex_safe,
    model: getCurrentModel(),
  };
}

/* ============================================================
   🔗 Export Namespace
   ============================================================ */
window.HIREX = window.HIREX || {};
Object.assign(window.HIREX, {
  // Versions/keys
  UTIL_VERSION,
  DEFAULT_BASE_TEX_PATH,
  LS_KEYS,

  // Storage
  lsGet,
  lsSet,
  lsRemove,

  // API base
  getApiBase,

  // Theme
  getSystemTheme,
  getTheme,
  setTheme,
  onThemeChange,

  // Humanize
  getHumanizeState,
  setHumanizeState,
  onHumanizeChange,

  // Model
  getCurrentModel,
  setCurrentModel,

  // Fetch
  fetchJSON,
  fetchText,
  fetchBinary,
  postJSON,
  withTimeout,

  // Base64 / Blob / data URL
  base64ToBlob,
  blobToBase64,
  toDataURL,
  parseDataURL,

  // ObjectURL helpers
  makeObjectURL,
  revokeObjectURL,
  revokeAllObjectURLs,

  // Download / clipboard
  downloadFile,
  downloadTextFile,
  copyToClipboard,

  // Utils
  getTimestamp,
  debounce,
  throttle,
  sleep,
  formatBytes,
  sanitizeFilename,

  // Builders
  buildOptimizeFormData,
  buildCoverLetterFormData,
  buildSuperhumanPayload,
});

console.log(
  `%c⚙️ [HIREX] util.js initialized — ${UTIL_VERSION}`,
  "background:#5bd0ff;color:#fff;padding:4px 8px;border-radius:4px;font-weight:bold;"
);
window.HIREX?.debugLog?.("UTIL LOADED", {
  version: UTIL_VERSION,
  origin: window.location.origin,
  theme: getTheme(),
  humanize: getHumanizeState(),
  model: getCurrentModel(),
});
