/* ============================================================
   HIREX • talk.js (v2.1.4 — Talk to HIREX)
   ------------------------------------------------------------
   Connectivity fixes:
   • Auto-detect & cache working API base (127.0.0.1 ↔ localhost ↔ same-origin)
   • Health ping on init with timeout + graceful fallbacks
   • Retries: /api/talk/answer → /api/talk across candidate bases
   • Robust error parsing (detail/error/message), timeout/cors/mixed-content hints
   • Model auto-map if a responses-only model is selected
   • Keeps typewriter, history, dashboard events
   Author: Sri Akash Kadali
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  const APP_VERSION = "v2.1.4";
  const LS_KEY = "hirex_talk_history";
  const LS_API_BASE = "hirex_api_base";
  const MAX_HISTORY = 40;
  const TIMEOUT_MS = 120000; // 2 minutes
  const PING_TIMEOUT_MS = 4000;
  const CANDIDATE_PORTS = [8000]; // extend if you proxy elsewhere

  /* ------------------------------------------------------------
     🔧 DOM References (support legacy/new IDs)
  ------------------------------------------------------------ */
  const form     = document.getElementById("talk-form")  || null;
  const inputEl  = document.getElementById("talk-input") || document.getElementById("question");
  const chatEl   = document.getElementById("talk-chat")  || document.getElementById("chat_box");
  const sendBtn  = document.getElementById("talk-send")  || document.getElementById("send_btn");
  const clearBtn = document.getElementById("talk-clear") || null;
  const statusEl = document.getElementById("talk-status")|| document.getElementById("contextStatus");
  const toneEl   = document.getElementById("talk-tone")  || null; // optional <select>

  /* ------------------------------------------------------------
     🧠 Utilities
  ------------------------------------------------------------ */
  const RT = (window.ASTRA ?? window.HIREX) || {};
  const toast = (msg, t = 3000) => (RT.toast ? RT.toast(msg, t) : console.info(msg));
  const debug = (msg, data) => RT.debugLog?.(msg, data);
  const setStatus = (txt) => { if (statusEl) statusEl.textContent = txt; };
  const scrollBottom = () => {
    if (!chatEl) return;
    try { chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: "smooth" }); }
    catch { chatEl.scrollTop = chatEl.scrollHeight; }
  };

  const sanitize = (s) => String(s || "").replace(/[\\/:*?"<>|]+/g, "_").trim();
  const keyFrom = (company = "", role = "") =>
    `${sanitize(company).toLowerCase().replace(/\s+/g, "_")}__${sanitize(role).toLowerCase().replace(/\s+/g, "_")}`;

  const getHumanize = () => {
    try { if (typeof RT.getHumanizeState === "function") return !!RT.getHumanizeState(); } catch {}
    const a = localStorage.getItem("hirex-use-humanize"); // "on" | "off"
    const b = localStorage.getItem("hirex_use_humanize"); // "true" | "false"
    if (a === "on" || a === "off") return a === "on";
    if (b === "true" || b === "false") return b === "true";
    return true; // default ON
  };
  const getTone = () => (toneEl?.value || "balanced").toLowerCase();

  // --- Model selection: prefer chat-safe default + auto-fix responses-only picks ---
  const CHAT_SAFE_DEFAULT_MODEL = "gpt-4o-mini";
  const RESPONSES_ONLY_HINTS = [/^gpt-image/i, /^dall[- ]?e/i, /^whisper/i];
  const isResponsesOnlyModel = (name) => {
    if (!name) return false;
    const n = String(name).trim();
    return RESPONSES_ONLY_HINTS.some((rx) => rx.test(n));
  };
  const chooseModel = () => {
    const raw =
      localStorage.getItem("hirex_talk_model") ||
      localStorage.getItem("hirex_model") ||
      localStorage.getItem("hirex_default_model") ||
      (typeof RT.getCurrentModel === "function" ? RT.getCurrentModel() : "") ||
      "";
    const model = (raw || "").trim();
    if (!model) return CHAT_SAFE_DEFAULT_MODEL;
    return isResponsesOnlyModel(model) ? CHAT_SAFE_DEFAULT_MODEL : model;
  };

  /* ------------------------------------------------------------
     💾 Local History
  ------------------------------------------------------------ */
  const loadHistory = () => {
    try {
      const arr = JSON.parse(localStorage.getItem(LS_KEY) || "[]");
      return Array.isArray(arr) ? arr : [];
    } catch { return []; }
  };
  const saveHistory = (hist) => {
    try { localStorage.setItem(LS_KEY, JSON.stringify(hist.slice(-MAX_HISTORY))); }
    catch (e) { console.warn("[HIREX] Talk history save failed:", e); }
  };
  let history = loadHistory();

  // Unified dashboard history
  const pushDashboardEvent = (meta = {}) => {
    try {
      const h = JSON.parse(localStorage.getItem("hirex_history") || "[]");
      h.push({ id: Date.now(), type: "talk", timestamp: new Date().toISOString(), ...meta });
      localStorage.setItem("hirex_history", JSON.stringify(h));
    } catch {}
  };

  /* ------------------------------------------------------------
     🧩 Context
  ------------------------------------------------------------ */
  const getSelectedContext = () => {
    try {
      const sel =
        JSON.parse(localStorage.getItem("hirex_selected_context") || "null") ||
        JSON.parse(localStorage.getItem("hirex_selected_cl") || "null");
      if (sel && (sel.jd_text || sel.resume_tex || sel.jd)) return sel;
    } catch {}
    return {
      jd_text:      localStorage.getItem("hirex_jd_text") || "",
      resume_tex:   localStorage.getItem("hirex_tex") || "",
      resume_plain: localStorage.getItem("hirex_resume_plain") || localStorage.getItem("hirex_resume_text") || "",
      company:      localStorage.getItem("hirex_company") || "",
      role:         localStorage.getItem("hirex_role") || "",
    };
  };
  const resolveContextKey = (ctx) => {
    if (ctx?.key) return String(ctx.key);
    if (ctx?.company || ctx?.role) return keyFrom(ctx.company || "", ctx.role || "");
    return "";
  };

  /* ------------------------------------------------------------
     🌐 API base discovery with ping
  ------------------------------------------------------------ */
  const storedBase = localStorage.getItem(LS_API_BASE) || "";
  const sameOrigin = location.origin;

  const localhostBases = [];
  for (const port of CANDIDATE_PORTS) {
    localhostBases.push(`http://127.0.0.1:${port}`);
    localhostBases.push(`http://localhost:${port}`);
  }

  // Meta tag override: <meta name="hirex-api-base" content="http://127.0.0.1:8000">
  const metaBase = (() => {
    const meta = document.querySelector('meta[name="hirex-api-base"]');
    return meta?.getAttribute("content")?.trim() || "";
  })();

  const candidBases = [
    storedBase,
    metaBase,
    sameOrigin.startsWith("http") ? sameOrigin : "",
    ...localhostBases,
  ].filter(Boolean);

  const controllerWithTimeout = (ms) => {
    const c = new AbortController();
    const t = setTimeout(() => c.abort(), ms);
    return { controller: c, cancel: () => clearTimeout(t) };
  };

  const pingBase = async (base) => {
    const { controller, cancel } = controllerWithTimeout(PING_TIMEOUT_MS);
    try {
      const res = await fetch(`${base}/api/talk/ping`, { signal: controller.signal });
      cancel();
      if (!res.ok) return false;
      const text = await res.text();
      return /"ok"\s*:\s*true/i.test(text);
    } catch {
      cancel();
      return false;
    }
  };

  let apiBase = sameOrigin; // optimistic default if served by backend
  const pickApiBase = async () => {
    for (const base of candidBases) {
      if (!base) continue;
      if (base.startsWith("https:") && /localhost|127\.0\.0\.1/.test(base)) continue;
      if (location.protocol === "https:" && base.startsWith("http:") && !/localhost|127\.0\.0\.1/.test(base)) continue;
      const ok = await pingBase(base);
      if (ok) {
        localStorage.setItem(LS_API_BASE, base);
        return base;
      }
    }
    // Final fallback; reverse proxy setups
    return sameOrigin;
  };

  /* ------------------------------------------------------------
     💬 Rendering
  ------------------------------------------------------------ */
  const renderMessage = (role /* 'user' | 'bot' */, text, { trusted = false } = {}) => {
    if (!chatEl) return null;
    const msg = document.createElement("div");
    msg.className = `msg ${role}${role === "bot" ? " ai" : ""}`;
    if (trusted) msg.innerHTML = text; else msg.textContent = text;
    chatEl.appendChild(msg);
    scrollBottom();
    return msg;
  };

  const appendTyping = (txt = "Thinking…") => {
    const el = renderMessage("bot", txt);
    if (el) el.classList.add("typing");
    return el;
  };

  const typeWriter = async (text, target, speed = 17) => {
    if (!target) return;
    target.textContent = "";
    for (const ch of text) {
      target.textContent += ch;
      // eslint-disable-next-line no-await-in-loop
      await new Promise((r) => setTimeout(r, speed));
    }
  };

  /* ------------------------------------------------------------
     🛰️ API core (with retries and fallbacks)
  ------------------------------------------------------------ */
  async function askBackend(question, controller, base) {
    const ctx = getSelectedContext();
    const chosenModel = chooseModel();

    const payload = {
      jd_text: ctx?.jd_text || ctx?.jd || "",
      question,
      resume_tex: ctx?.resume_tex || "",
      resume_plain: ctx?.resume_plain || "",
      tone: getTone(),
      humanize: getHumanize(),
      model: chosenModel,
      context_key: resolveContextKey(ctx) || undefined,
      context_id: ctx?.id || ctx?.context_id || undefined,
      title: ctx?.title || undefined,
      use_latest: true,
    };

    const tryPost = async (b, path) => {
      const res = await fetch(`${b}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      const raw = await res.text().catch(() => "");
      if (!res.ok) {
        let msg = raw || `HTTP ${res.status}`;
        try {
          const j = JSON.parse(raw || "{}");
          msg = j.detail || j.error || j.message || msg;
        } catch {}
        const err = new Error(msg);
        err.status = res.status;
        err.base = b;
        err.path = path;
        throw err;
      }
      return JSON.parse(raw || "{}");
    };

    const paths = ["/api/talk/answer", "/api/talk"];
    const bases = Array.from(new Set([base, sameOrigin, ...localhostBases].filter(Boolean)));

    let lastErr;
    for (const b of bases) {
      for (const p of paths) {
        try {
          const data = await tryPost(b, p);
          const reply =
            (typeof data.final_text === "string" && data.final_text.trim()) ||
            (typeof data.answer === "string" && data.answer.trim()) ||
            (typeof data.draft_answer === "string" && data.draft_answer.trim()) ||
            "";

          const usedModel = data.model || chosenModel;
          debug("Talk API OK", {
            base: b, path: p, model: usedModel,
            humanized_requested: payload.humanize,
            humanized_applied: !!data.humanized,
            tone: data.tone,
            context: data.context || {},
          });

          localStorage.setItem(LS_API_BASE, b);

          return {
            reply: reply || "⚠️ No response received.",
            model: usedModel,
            ctxMeta: data.context || {},
            humanized: !!data.humanized,
          };
        } catch (err) {
          lastErr = err;
          if (err.status === 404 || err.status === 405) continue; // try next path
          break; // try next base
        }
      }
    }

    const m = String(lastErr?.message || "").toLowerCase();
    if (m.includes("only supported in v1/responses")) {
      toast("ℹ️ Switched to a chat-safe model (gpt-4o-mini). Try again.");
      localStorage.setItem("hirex_talk_model", CHAT_SAFE_DEFAULT_MODEL);
    } else if (location.protocol === "https:" && /http:\/\/(127\.0\.0\.1|localhost)/.test(localStorage.getItem(LS_API_BASE) || "")) {
      toast("⚠️ Mixed-content blocked: run both UI and backend on the same scheme/host, or use http locally.");
    } else if (lastErr?.status === 0) {
      toast("⚠️ Network blocked by CORS or an extension.");
    } else {
      toast(`⚠️ Backend error: ${lastErr?.message || "unreachable"}`);
    }
    return { reply: "⚠️ Unable to connect to backend.", model: "offline" };
  }

  /* ------------------------------------------------------------
     🚀 Send Flow
  ------------------------------------------------------------ */
  const sendFlow = async (resolvedBase) => {
    const text = inputEl?.value?.trim();
    if (!text) return toast("💬 Please enter a question first.");

    const ctx = getSelectedContext();
    if (!(ctx?.jd_text || ctx?.jd) || !(ctx?.resume_tex || ctx?.resume_plain)) {
      console.warn("[HIREX] Missing JD or Resume in local context — backend will attempt latest saved context.");
    }

    renderMessage("user", text);
    history.push({ role: "user", content: text });
    saveHistory(history);
    pushDashboardEvent({
      event: "user_msg",
      company: ctx.company || "",
      role: ctx.role || "",
      context_key: resolveContextKey(ctx) || "",
    });

    if (inputEl) { inputEl.value = ""; inputEl.disabled = true; }
    if (sendBtn) sendBtn.disabled = true;
    setStatus("Thinking…");

    const typingEl = appendTyping();
    const { controller, cancel } = controllerWithTimeout(TIMEOUT_MS);

    let result;
    try {
      result = await askBackend(text, controller, resolvedBase);
    } finally {
      cancel();
    }

    const reply = result.reply || "⚠️ No response received.";
    if (typingEl) {
      typingEl.classList.remove("typing");
      await typeWriter(reply, typingEl, result.humanized ? 12 : 17);
    } else {
      renderMessage("bot", reply);
    }

    history.push({ role: "bot", content: reply, model: result.model, humanized: !!result.humanized });
    saveHistory(history);
    pushDashboardEvent({
      event: "assistant_msg",
      company: ctx.company || "",
      role: ctx.role || "",
      context_key: resolveContextKey(ctx) || "",
    });

    if (inputEl) { inputEl.disabled = false; inputEl.focus(); }
    if (sendBtn) sendBtn.disabled = false;
    setStatus("Ready");
  };

  /* ------------------------------------------------------------
     🔘 Wiring (form or button)
  ------------------------------------------------------------ */
  let ACTIVE_API_BASE = sameOrigin; // will be replaced after ping
  (async () => {
    setStatus("Checking backend…");

    try {
      if (typeof RT.getApiBase === "function") {
        const fromRT = RT.getApiBase();
        if (fromRT && await pingBase(fromRT)) {
          ACTIVE_API_BASE = fromRT;
          localStorage.setItem(LS_API_BASE, fromRT);
        } else {
          ACTIVE_API_BASE = await pickApiBase();
        }
      } else {
        ACTIVE_API_BASE = await pickApiBase();
      }
    } catch {
      ACTIVE_API_BASE = await pickApiBase();
    }

    const ok = await pingBase(ACTIVE_API_BASE);
    if (!ok) {
      toast("⚠️ Backend ping failed, will retry on send.");
      setStatus("Backend unreachable (will retry)");
    } else {
      setStatus("Ready");
    }

    form?.addEventListener("submit", (e) => { e.preventDefault(); sendFlow(ACTIVE_API_BASE); });
    sendBtn?.addEventListener("click", () => sendFlow(ACTIVE_API_BASE));
  })();

  // Ctrl/Cmd + Enter (send), Shift+Enter (newline)
  inputEl?.addEventListener("keydown", (e) => {
    const isMac = /Mac|iPhone|iPad/i.test(navigator.platform);
    const mod = isMac ? e.metaKey : e.ctrlKey;
    if (mod && e.key.toLowerCase() === "enter") {
      e.preventDefault();
      const b = localStorage.getItem(LS_API_BASE) || sameOrigin;
      sendFlow(b);
    }
  });

  /* ------------------------------------------------------------
     🧹 Clear Chat
  ------------------------------------------------------------ */
  clearBtn?.addEventListener("click", () => {
    if (chatEl) chatEl.innerHTML = "";
    history = [];
    saveHistory(history);
    toast("🧹 Chat cleared.");
    setStatus("Ready");
  });

  /* ------------------------------------------------------------
     ♻️ Restore or Greet
  ------------------------------------------------------------ */
  if (chatEl && history.length) {
    history.forEach((m) => {
      const role = m.role === "assistant" || m.role === "ai" ? "bot" : (m.role === "bot" ? "bot" : "user");
      renderMessage(role, m.content, { trusted: false });
    });
    scrollBottom();
  } else if (chatEl) {
    renderMessage(
      "bot",
      "👋 Hi! I’m <b>HIREX</b> — ask any JD-specific or interview question. I’ll answer based on your latest optimized/humanized resume and job description.",
      { trusted: true }
    );
  }

  /* ------------------------------------------------------------
     🎨 Theme + Humanize Signals
  ------------------------------------------------------------ */
  window.addEventListener("hirex:theme-change", (e) =>
    debug("Talk theme changed", { theme: e.detail?.theme })
  );
  window.addEventListener("hirex:humanize-change", (e) =>
    debug("Talk humanize toggled", { on: e.detail?.on })
  );

  /* ------------------------------------------------------------
     ✅ Init Log
  ------------------------------------------------------------ */
  console.log(
    `%c💬 HIREX talk.js initialized — ${APP_VERSION}`,
    "background:#5bd0ff;color:#fff;padding:4px 8px;border-radius:4px;font-weight:bold;"
  );
  const hasCtx = (() => {
    const c = getSelectedContext();
    return !!(c && (c.jd_text || c.jd) && (c.resume_tex || c.resume_plain));
  })();
  debug("TALK PAGE LOADED", {
    version: APP_VERSION,
    historyCount: history.length,
    humanize: getHumanize(),
    hasContext: hasCtx,
    apiBase: localStorage.getItem(LS_API_BASE) || "(auto)",
    tone: getTone(),
    model: chooseModel(),
  });
});
