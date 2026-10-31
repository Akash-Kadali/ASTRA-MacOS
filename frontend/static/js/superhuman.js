/* ============================================================
   HIREX • superhuman.js (v1.0.0 — Humanizer Engine)
   ------------------------------------------------------------
   Features:
   • Sends text (resume, paragraph, or cover letter body)
     to /api/superhuman/rewrite for tone refinement
   • Uses AI Humanize API via backend proxy
   • Supports tone modes (Formal, Balanced, Conversational, Academic, Confident)
   • Honors global Humanize toggle (skips API when off)
   • Displays rewritten text side-by-side
   • Cache: tone + input/output persisted across sessions
   • Clear control + graceful backend error handling
   • Persistent global toggle synced across pages
   • Emits history event (type: "humanize") for dashboard
   Author: Sri Akash Kadali
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  const APP_VERSION = "v1.0.0";
  const LS_KEY = "hirex_superhuman_cache";
  const TIMEOUT_MS = 120000;
  const HUMANIZE_KEY_NEW = "hirex_use_humanize";
  const HUMANIZE_KEY_OLD = "hirex-use-humanize";

  /* ------------------------------------------------------------
     🔧 DOM Elements
  ------------------------------------------------------------ */
  const inputEl     = document.getElementById("input_text");
  const modeEl      = document.getElementById("mode");
  const toneEl      = document.getElementById("tone");
  const latexSafeEl = document.getElementById("latex_safe");
  const rewriteBtn  = document.getElementById("rewrite_btn");
  const clearBtn    = document.getElementById("clear_btn");
  const statusBadge = document.getElementById("status_badge");
  const outEl       = document.getElementById("human_output");
  const toggleEl    = document.getElementById("humanize_toggle");

  /* ------------------------------------------------------------
     🧩 Utilities
  ------------------------------------------------------------ */
  const RT = (window.ASTRA ?? window.HIREX) || {};

  const getApiBase = () => {
    try { if (typeof RT.getApiBase === "function") return RT.getApiBase(); } catch {}
    return ["127.0.0.1", "localhost"].includes(location.hostname)
      ? "http://127.0.0.1:8000"
      : location.origin;
  };
  const apiBase = getApiBase();

  const toast = (msg, t = 3000) => (RT.toast ? RT.toast(msg, t) : alert(msg));
  const debug = (msg, data) => RT.debugLog?.(msg, data);

  const pushHistoryEvent = (meta = {}) => {
    try {
      const history = JSON.parse(localStorage.getItem("hirex_history") || "[]");
      history.push({
        id: Date.now(),
        type: "humanize",
        timestamp: new Date().toISOString(),
        tone: toneEl?.value || "balanced",
        mode: modeEl?.value || "paragraph",
        ...meta,
      });
      localStorage.setItem("hirex_history", JSON.stringify(history));
    } catch {}
  };

  /* ------------------------------------------------------------
     🌐 Global Humanize Toggle
  ------------------------------------------------------------ */
  const getHumanizeState = () => {
    try { if (window.ASTRA?.getHumanizeState) return !!window.ASTRA.getHumanizeState(); } catch {}
    const v = localStorage.getItem(HUMANIZE_KEY_NEW);
    if (v === null) return true; // default ON
    if (v === "true" || v === "false") return v === "true";
    return (localStorage.getItem(HUMANIZE_KEY_OLD) ?? "on") === "on";
  };

  const setHumanizeState = (on) => {
    localStorage.setItem(HUMANIZE_KEY_NEW, on ? "true" : "false");
    localStorage.setItem(HUMANIZE_KEY_OLD, on ? "on" : "off");
    updateHumanizeBadge(on);
    const evt = new CustomEvent("hirex:humanize-change", { detail: { on } });
    // broadcast to both targets for maximum compatibility
    window.dispatchEvent(evt);
    document.dispatchEvent(evt);
  };

  const updateHumanizeBadge = (on = getHumanizeState()) => {
    if (!statusBadge) return;
    statusBadge.textContent = on
      ? "🟢 Humanize ON"
      : "⚪ Humanize OFF — showing original text";
  };

  if (toggleEl) {
    toggleEl.checked = getHumanizeState();
    toggleEl.addEventListener("change", () => {
      setHumanizeState(toggleEl.checked);
      toast(toggleEl.checked ? "✨ Humanize enabled" : "⛔ Humanize disabled");
    });
  }

  updateHumanizeBadge();

  /* ------------------------------------------------------------
     💾 Cache load / restore
  ------------------------------------------------------------ */
  const loadCache = () => {
    try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}"); }
    catch { return {}; }
  };

  const saveCache = (patch) => {
    try {
      const prev = loadCache();
      localStorage.setItem(
        LS_KEY,
        JSON.stringify({ ...prev, ...patch, _v: APP_VERSION, _t: Date.now() })
      );
    } catch (e) {
      console.warn("[HIREX] SuperHuman cache save failed:", e);
    }
  };

  const setOutputText = (text) => { if (outEl) outEl.textContent = text || ""; };

  // Prefill input from cache or recent selection
  (() => {
    const boot = loadCache();
    if (boot.lastTone && toneEl) toneEl.value = boot.lastTone;
    if (boot.lastMode && modeEl) modeEl.value = boot.lastMode;
    if (typeof boot.latexSafe === "boolean" && latexSafeEl) latexSafeEl.checked = boot.latexSafe;

    let seeded = false;
    if (boot.lastInput && inputEl && !inputEl.value) {
      inputEl.value = boot.lastInput;
      seeded = true;
    }
    // If not seeded, try currently selected resume text from other pages
    if (!seeded && inputEl && !inputEl.value) {
      try {
        const sel = JSON.parse(localStorage.getItem("hirex_selected_context") || "null");
        const fallback =
          localStorage.getItem("hirex_resume_text") ||
          localStorage.getItem("hirex_resume_plain") ||
          localStorage.getItem("hirex_tex") ||
          "";
        inputEl.value = sel?.resume_tex || fallback || "";
      } catch {
        inputEl.value = localStorage.getItem("hirex_tex") || "";
      }
    }
    if (boot.lastOutput) setOutputText(boot.lastOutput);
  })();

  /* ------------------------------------------------------------
     🛰️ API Call
  ------------------------------------------------------------ */
  const MODE_MAP = { sentence: "paragraph", custom: "paragraph" };
  const TONE_MAP = new Set(["balanced", "formal", "conversational", "academic", "confident"]);

  const buildPayload = (text, tone, mode, latexSafe) => {
    const normTone = String(tone || "balanced").toLowerCase();
    const toneFinal = TONE_MAP.has(normTone) ? normTone : "balanced";

    const rawMode = String(mode || "paragraph").toLowerCase();
    const modeFinal = ["paragraph", "resume", "coverletter"].includes(rawMode)
      ? rawMode
      : (MODE_MAP[rawMode] || "paragraph");

    const hints = [];
    if (rawMode === "sentence") hints.push("granularity:sentence");
    if (rawMode === "custom") hints.push("style:custom");

    return { text, tone: toneFinal, mode: modeFinal, latex_safe: !!latexSafe, hints };
  };

  async function callRewrite(payload, controller) {
    const url = `${apiBase}/api/superhuman/rewrite`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    const raw = await res.text().catch(() => "");
    if (!res.ok) {
      try {
        const j = JSON.parse(raw || "{}");
        const msg = j.detail || j.error || j.message || raw || `HTTP ${res.status}`;
        throw new Error(msg);
      } catch {
        throw new Error(raw || `HTTP ${res.status}`);
      }
    }

    try { return JSON.parse(raw || "{}"); }
    catch { throw new Error("Invalid JSON from backend."); }
  }

  const extractRewritten = (response, originalText) => {
    // Accept multiple shapes
    // { rewritten: "..." } | { rewritten: ["...","..."] } | { text: "..." } | { output: "..." }
    // { result: { text: "..." }} | { outputs: [{text:"..."}] } | { data: { rewritten: "..." } }
    const r = response?.rewritten ?? response?.data?.rewritten ?? response?.result?.text;
    if (Array.isArray(r)) return r.join("\n");
    if (typeof r === "string" && r.trim()) return r;

    const fromOutputs = Array.isArray(response?.outputs) && response.outputs[0]?.text;
    if (fromOutputs) return String(fromOutputs);

    if (typeof response?.text === "string" && response.text.trim()) return response.text;
    if (typeof response?.output === "string" && response.output.trim()) return response.output;

    return originalText || "";
  };

  const setBusy = (busy, msg = "") => {
    [rewriteBtn, clearBtn, inputEl, modeEl, toneEl, latexSafeEl]
      .filter(Boolean)
      .forEach((el) => (el.disabled = !!busy));
    if (statusBadge) statusBadge.textContent = busy
      ? (msg || "Working…")
      : (getHumanizeState() ? "🟢 Ready" : "⚪ Humanize OFF — showing original text");
  };

  /* ------------------------------------------------------------
     ⚡ Run Humanize
  ------------------------------------------------------------ */
  const runHumanize = async () => {
    const text = inputEl?.value?.trim() || "";
    if (!text) return toast("⚠️ Please paste text to refine.");

    const tone = toneEl?.value || "balanced";
    const mode = modeEl?.value || "paragraph";
    const latexSafe = !!latexSafeEl?.checked;
    const humanizeOn = getHumanizeState();

    if (!humanizeOn) {
      setOutputText(text);
      saveCache({ lastInput: text, lastOutput: text, lastTone: tone, lastMode: mode, latexSafe });
      pushHistoryEvent({ event: "show_original" });
      toast("⚪ Humanize is off — showing original text.");
      updateHumanizeBadge(false);
      return;
    }

    setBusy(true, "Refining…");
    toast(`⚡ Humanizing in ${tone} tone…`);
    updateHumanizeBadge(true);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
      const payload = buildPayload(text, tone, mode, latexSafe);
      let response;
      try {
        response = await callRewrite(payload, controller);
      } catch (err) {
        console.error("[HIREX] SuperHuman fetch failed:", err);
        toast("⚠️ Backend unreachable — showing original text.");
        response = { rewritten: text };
      }

      const humanized = extractRewritten(response, text) || "⚠️ No response received.";
      setOutputText(humanized);
      toast("✅ Text refined successfully!");
      saveCache({ lastInput: text, lastOutput: humanized, lastTone: tone, lastMode: mode, latexSafe });
      pushHistoryEvent({ event: "rewrite_success" });
    } catch (err) {
      if (err.name === "AbortError") toast("⚠️ Request timed out (2 min).");
      else toast("❌ " + (err.message || "Unexpected error."));
      console.error("[HIREX] SuperHuman error:", err);
      pushHistoryEvent({ event: "rewrite_error", error: String(err?.message || err) });
    } finally {
      clearTimeout(timer);
      setBusy(false);
    }
  };

  rewriteBtn?.addEventListener("click", runHumanize);

  /* ------------------------------------------------------------
     🧹 Clear
  ------------------------------------------------------------ */
  clearBtn?.addEventListener("click", () => {
    if (inputEl) inputEl.value = "";
    setOutputText("");
    saveCache({ lastInput: "", lastOutput: "", lastTone: toneEl?.value, lastMode: modeEl?.value, latexSafe: !!latexSafeEl?.checked });
    pushHistoryEvent({ event: "cleared" });
    toast("🧹 Cleared input and output.");
    updateHumanizeBadge(getHumanizeState());
  });

  /* ------------------------------------------------------------
     ⌨️ Shortcut
  ------------------------------------------------------------ */
  inputEl?.addEventListener("keydown", (e) => {
    const isMac = /Mac|iPhone|iPad/i.test(navigator.platform);
    const mod = isMac ? e.metaKey : e.ctrlKey;
    if (mod && e.key.toLowerCase() === "enter") {
      e.preventDefault();
      runHumanize();
    }
  });

  /* ------------------------------------------------------------
     🔄 Persist Choices
  ------------------------------------------------------------ */
  toneEl?.addEventListener("change", () => saveCache({ lastTone: toneEl.value }));
  modeEl?.addEventListener("change", () => saveCache({ lastMode: modeEl.value }));
  latexSafeEl?.addEventListener("change", () => saveCache({ latexSafe: !!latexSafeEl.checked }));

  /* ------------------------------------------------------------
     ✅ Init Log
  ------------------------------------------------------------ */
  console.log(
    `%c⚡ HIREX superhuman.js initialized — ${APP_VERSION}`,
    "background:#5bd0ff;color:#00131c;padding:4px 8px;border-radius:4px;font-weight:bold;"
  );
  debug("SUPERHUMAN PAGE LOADED", {
    version: APP_VERSION,
    apiBase,
    tone: toneEl?.value,
    mode: modeEl?.value,
    humanizeOn: getHumanizeState(),
  });
});
