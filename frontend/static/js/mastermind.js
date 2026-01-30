/* ============================================================
   ASTRA • mastermind.js (v2.1.0 — MasterMind Chat Core)
   ------------------------------------------------------------
   Features:
   • Chat assistant powered by /api/mastermind (backend sessions)
   • Starts/loads sessions; renders history; persists to localStorage
   • Persona/model selectors (optional); tone control (balanced default)
   • Typing indicator + typewriter animation (no duplicate bubbles)
   • Intelligent timeout + graceful offline fallback
   • Emits history events for dashboard (type: "mastermind")
   • Works even if some DOM nodes are missing; guards double-binding
   Author: Sri Akash Kadali
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  const APP = "ASTRA";
  const APP_VERSION = "v2.1.0";
  const TIMEOUT_MS = 180000; // 3 minutes

  /* ------------------------------------------------------------
     🌐 DOM References (supports both legacy/new IDs)
  ------------------------------------------------------------ */
  const $ = (id) => document.getElementById(id);

  // Chat stream container (new id first, legacy second)
  const chatBox   = $("chat_stream") || $("chat_box");
  const chatInput = $("chat_input");
  const sendBtn   = $("send_btn");

  // Optional controls (exist on mastermind.html)
  const sessionList   = $("session_list");
  const newSessionBtn = $("new_session");
  const refreshBtn    = $("refresh_sessions");
  const personaSel    = $("persona");
  const modelSel      = $("model");
  const toneSel       = $("tone");           // optional (fallback to "balanced")
  const useContextEl  = $("use_context");    // optional

  /* ------------------------------------------------------------
     🧠 Runtime helpers
  ------------------------------------------------------------ */
  const RT = (window.ASTRA ?? window.HIREX) || {};
  const toast = (msg, t = 2600) => (RT.toast ? RT.toast(msg, t) : console.info(msg));
  const debug = (msg, data) => RT.debugLog?.(msg, data);

  const getApiBase = () => {
    try { if (typeof RT.getApiBase === "function") return RT.getApiBase(); } catch {}
    return ["127.0.0.1", "localhost"].includes(location.hostname)
      ? "http://127.0.0.1:8000"
      : location.origin;
  };
  const apiBase = getApiBase();

  const safeHTML = (s = "") =>
    String(s).replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/\n/g, "<br>");

  const scrollToBottom = () => chatBox?.scrollTo({ top: chatBox.scrollHeight, behavior: "smooth" });

  const persistPref = (key, val) => {
    try { localStorage.setItem(key, val); } catch {}
  };
  const readPref = (key, fallback) => {
    try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
  };

  // Restore selector prefs if available
  if (personaSel) personaSel.value = readPref("astra_mm_persona", personaSel.value || "General");
  if (modelSel)   modelSel.value   = readPref("astra_mm_model",   modelSel.value   || "gpt-4o-mini");
  if (toneSel)    toneSel.value    = readPref("astra_mm_tone",    toneSel.value    || "balanced");

  personaSel?.addEventListener("change", () => persistPref("astra_mm_persona", personaSel.value));
  modelSel?.addEventListener("change",   () => persistPref("astra_mm_model",   modelSel.value));
  toneSel?.addEventListener("change",    () => persistPref("astra_mm_tone",    toneSel.value));

  /* ------------------------------------------------------------
     💾 Local session store (mirrors backend session IDs)
  ------------------------------------------------------------ */
  const SESS_KEY = "astra_mastermind_sessions_v2";
  const CURR_KEY = "astra_mm_current_id";

  /** @typedef {{id:string,title:string,createdAt:string,messages:Array<{role:'user'|'assistant',content:string,model?:string,tone?:string,ts?:string}>}} LocalSession */

  /** @returns {LocalSession[]} */
  const loadSessions = () => {
    try { return JSON.parse(localStorage.getItem(SESS_KEY) || "[]"); }
    catch { return []; }
  };
  const saveSessions = (sessions) => {
    try { localStorage.setItem(SESS_KEY, JSON.stringify(sessions)); }
    catch (e) { console.warn("[ASTRA] Failed to persist sessions:", e); }
  };

  // Emit events to unified dashboard history
  const pushHistoryEvent = (meta = {}) => {
    try {
      const history = JSON.parse(localStorage.getItem("hirex_history") || "[]");
      history.push({
        id: Date.now(),
        type: "mastermind",
        timestamp: new Date().toISOString(),
        persona: personaSel?.value || "General",
        model: modelSel?.value || "gpt-4o-mini",
        ...meta,
      });
      localStorage.setItem("hirex_history", JSON.stringify(history));
    } catch {}
  };

  let sessions = loadSessions();
  let currentId = localStorage.getItem(CURR_KEY) || "";

  const setCurrent = (id) => {
    currentId = id;
    localStorage.setItem(CURR_KEY, id);
    renderSessionList();
    renderConversation();
  };

  const getCurrent = () => sessions.find((s) => s.id === currentId) || null;

  const upsertLocalSession = (id, meta = {}) => {
    let s = sessions.find((x) => x.id === id);
    if (!s) {
      s = {
        id,
        title: meta.title || `Session ${sessions.length + 1}`,
        createdAt: meta.created || new Date().toISOString(),
        messages: [],
      };
      sessions.push(s);
    } else {
      if (meta.title)   s.title = meta.title;
      if (meta.created) s.createdAt = meta.created;
    }
    saveSessions(sessions);
    return s;
  };

  /* ------------------------------------------------------------
     🖼️ Renderers — match mastermind.html structure (.msg-row)
  ------------------------------------------------------------ */
  const renderRow = (role, html) => {
    if (!chatBox) return { row: null, bubble: null };
    const row = document.createElement("div");
    row.className = "msg-row " + (role === "user" ? "user" : "bot");

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.innerHTML = `<span>${role === "user" ? "🧑" : "🤖"}</span>`;

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = html;

    row.append(avatar, bubble);
    chatBox.append(row);
    scrollToBottom();
    return { row, bubble };
  };

  const renderSessionList = () => {
    if (!sessionList) return;
    sessionList.innerHTML = "";
    sessions.forEach((s) => {
      const item = document.createElement("div");
      item.className = "session-item" + (s.id === currentId ? " active" : "");
      item.textContent = s.title;
      item.title = new Date(s.createdAt).toLocaleString();
      item.addEventListener("click", () => setCurrent(s.id));
      sessionList.appendChild(item);
    });
  };

  const renderConversation = () => {
    if (!chatBox) return;
    chatBox.innerHTML = "";
    const s = getCurrent();
    if (!s) return;
    s.messages.forEach((m) => renderRow(m.role, safeHTML(m.content)));
    scrollToBottom();
  };

  const addMsg = (role, content, extras = {}) => {
    const s = getCurrent();
    if (!s) return { row: null, bubble: null };
    const msg = { role, content, ...extras, ts: new Date().toISOString() };
    s.messages.push(msg);
    saveSessions(sessions);
    return renderRow(role, safeHTML(content));
  };

  const addTyping = (text = "…thinking…") => {
    if (!chatBox) return { row: null, bubble: null };
    const row = document.createElement("div");
    row.className = "msg-row bot";
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.innerHTML = "<span>🤖</span>";
    const bubble = document.createElement("div");
    bubble.className = "bubble typing";
    bubble.textContent = text;
    row.append(avatar, bubble);
    chatBox.append(row);
    scrollToBottom();
    return { row, bubble };
  };

  const typeWriter = async (fullText, el, speed = 14) => {
    if (!el) return;
    el.innerHTML = "";
    for (const ch of fullText) {
      // eslint-disable-next-line no-await-in-loop
      await new Promise((r) => setTimeout(r, speed));
      el.innerHTML += safeHTML(ch);
    }
  };

  /* ------------------------------------------------------------
     🌐 Backend API (updated routes)
     - POST /api/mastermind/start   (FormData: persona, model, purpose)
     - POST /api/mastermind/chat    (FormData: session_id, prompt, tone, model, persona, temperature[, use_context])
     - GET  /api/mastermind/history?session_id=...
     - GET  /api/mastermind/sessions
  ------------------------------------------------------------ */
  const apiStartSession = async ({
    persona = "General",
    model = "gpt-4o-mini",
    purpose = "interactive reasoning",
  } = {}) => {
    const url = `${apiBase}/api/mastermind/start`;
    const fd = new FormData();
    fd.append("persona", persona);
    fd.append("model", model);
    fd.append("purpose", purpose);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
      const res = await fetch(url, {
        method: "POST",
        body: fd,
        signal: controller.signal,
        credentials: "same-origin",
      });
      clearTimeout(timeout);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const sid = data?.session?.id;
      const created = data?.session?.created || new Date().toISOString();
      const title = data?.session?.title || `${persona} Chat`;
      const s = upsertLocalSession(sid, { title, created });
      pushHistoryEvent({ session_id: sid, event: "start" });
      return s;
    } catch (e) {
      clearTimeout(timeout);
      console.warn("[ASTRA] start_session failed:", e);
      const localId = "local_" + Date.now().toString(36);
      const s = upsertLocalSession(localId, { title: "Local Session", created: new Date().toISOString() });
      toast("⚠️ Offline mode: local session created.");
      pushHistoryEvent({ session_id: localId, event: "start_offline" });
      return s;
    }
  };

  const apiListSessions = async () => {
    try {
      const res = await fetch(`${apiBase}/api/mastermind/sessions`, { credentials: "same-origin" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return Array.isArray(data.sessions) ? data.sessions : [];
    } catch (e) {
      console.warn("[ASTRA] list_sessions failed:", e);
      return [];
    }
  };

  const apiGetHistory = async (session_id) => {
    try {
      const res = await fetch(`${apiBase}/api/mastermind/history?session_id=${encodeURIComponent(session_id)}`, {
        credentials: "same-origin",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return (data.session && Array.isArray(data.session.messages)) ? data.session.messages : [];
    } catch (e) {
      console.warn("[ASTRA] get_history failed:", e);
      return [];
    }
  };

  const apiChat = async ({
    session_id,
    prompt,
    tone = "balanced",
    model = "gpt-4o-mini",
    persona = "General",
    temperature = 0.6,
    use_context = true,
  }) => {
    const url = `${apiBase}/api/mastermind/chat`;
    const fd = new FormData();
    fd.append("session_id", session_id);
    fd.append("prompt", prompt);
    fd.append("tone", tone);
    fd.append("model", model);
    fd.append("persona", persona);
    fd.append("temperature", String(temperature));
    fd.append("use_context", use_context ? "true" : "false");

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
      debug("MM_CHAT_POST", { session_id, persona, model, tone, version: APP_VERSION });
      const res = await fetch(url, {
        method: "POST",
        body: fd,
        signal: controller.signal,
        credentials: "same-origin",
      });
      clearTimeout(timeout);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      return {
        reply: data.reply ?? "",
        model: data.model ?? model,
        tone: data.tone ?? tone,
        ts: data.timestamp || new Date().toISOString(),
      };
    } catch (e) {
      clearTimeout(timeout);
      console.warn("[ASTRA] mastermind/chat failed:", e);
      return {
        reply: "⚠️ I couldn’t reach MasterMind right now. Please try again shortly.",
        model: "offline",
        tone,
        ts: new Date().toISOString(),
      };
    }
  };

  /* ------------------------------------------------------------
     🔁 Sync helpers
  ------------------------------------------------------------ */
  const syncSessionsFromBackend = async () => {
    const remote = await apiListSessions(); // [{id, created, persona, message_count}, ...]
    if (!remote.length) return;
    remote.forEach((r, idx) =>
      upsertLocalSession(r.id, {
        title: r.persona ? `${r.persona} #${idx + 1}` : `Session ${idx + 1}`,
        created: r.created,
      })
    );
    renderSessionList();
  };

  const loadHistoryIntoLocal = async (session_id) => {
    const s = upsertLocalSession(session_id);
    const history = await apiGetHistory(session_id);
    if (history.length) {
      s.messages = history.map((m) => ({
        role: m.role === "assistant" ? "assistant" : "user",
        content: String(m.content || ""),
        model: m.model || "",
        tone: m.tone || "",
        ts: m.ts || m.timestamp || new Date().toISOString(),
      }));
      saveSessions(sessions);
      if (session_id === currentId) renderConversation();
    }
  };

  /* ------------------------------------------------------------
     🧾 Send flow (update typing bubble instead of adding new)
  ------------------------------------------------------------ */
  const doSend = async () => {
    const text = chatInput?.value?.trim();
    if (!text) return;

    // Ensure session exists
    let s = getCurrent();
    if (!s) {
      s = await apiStartSession({
        persona: personaSel?.value || "General",
        model: modelSel?.value || "gpt-4o-mini",
      });
      setCurrent(s.id);
    }

    // Render + persist user message
    addMsg("user", text);
    pushHistoryEvent({ session_id: s.id, event: "user_msg" });

    // Disable UI
    if (chatInput) { chatInput.value = ""; chatInput.disabled = true; }
    if (sendBtn)   { sendBtn.disabled  = true; }

    // Typing indicator
    const typing = addTyping();

    const persona = personaSel?.value || "General";
    const model   = modelSel?.value || "gpt-4o-mini";
    const tone    = toneSel?.value || "balanced";
    const useCtx  = useContextEl ? !!useContextEl.checked : true;

    debug("MM_SEND", { persona, model, text, version: APP_VERSION });

    const { reply, model: usedModel, ts } = await apiChat({
      session_id: s.id,
      prompt: text,
      tone,
      model,
      persona,
      temperature: 0.6,
      use_context: useCtx,
    });

    // Replace typing bubble contents (no duplicate DOM)
    if (typing?.bubble) {
      typing.bubble.classList.remove("typing");
      await typeWriter(reply || "⚠️ No response.", typing.bubble, 16);
    } else {
      renderRow("assistant", safeHTML(reply || "⚠️ No response."));
    }

    // Persist assistant message (without adding another bubble)
    const cur = getCurrent();
    if (cur) {
      cur.messages.push({
        role: "assistant",
        content: reply || "⚠️ No response.",
        model: usedModel,
        tone,
        ts: ts || new Date().toISOString(),
      });
      saveSessions(sessions);
    }
    pushHistoryEvent({ session_id: s.id, event: "assistant_msg" });

    // Re-enable UI
    if (chatInput) { chatInput.disabled = false; chatInput.focus(); }
    if (sendBtn)   { sendBtn.disabled  = false; }
  };

  /* ------------------------------------------------------------
     🧭 Wire up events (guard from double-binding)
  ------------------------------------------------------------ */
  if (sendBtn && !sendBtn.dataset.mmBound) {
    sendBtn.addEventListener("click", doSend);
    sendBtn.dataset.mmBound = "1";
  }

  if (chatInput && !chatInput.dataset.mmBound) {
    chatInput.addEventListener("keydown", (e) => {
      // Enter to send; Shift+Enter = newline; also allow Ctrl/Cmd+Enter
      const k = e.key.toLowerCase();
      if (k === "enter" && (!e.shiftKey || e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        doSend();
      }
    });
    chatInput.dataset.mmBound = "1";
  }

  if (newSessionBtn && !newSessionBtn.dataset.mmBound) {
    newSessionBtn.addEventListener("click", async () => {
      const persona = personaSel?.value || "General";
      const model   = modelSel?.value || "gpt-4o-mini";
      const s = await apiStartSession({ persona, model });
      setCurrent(s.id);
      toast(`🆕 ${s.title} created`);
    });
    newSessionBtn.dataset.mmBound = "1";
  }

  if (refreshBtn && !refreshBtn.dataset.mmBound) {
    refreshBtn.addEventListener("click", async () => {
      await syncSessionsFromBackend();
      const s = getCurrent();
      if (s && !s.id.startsWith("local_")) await loadHistoryIntoLocal(s.id);
      renderSessionList();
      renderConversation();
      toast("↻ Sessions refreshed");
    });
    refreshBtn.dataset.mmBound = "1";
  }

  /* ------------------------------------------------------------
     🚦 Init
  ------------------------------------------------------------ */
  (async () => {
    await syncSessionsFromBackend();

    // If stale currentId points to nowhere, clear it
    if (currentId && !getCurrent()) {
      currentId = "";
      localStorage.removeItem(CURR_KEY);
    }

    if (!currentId && sessions.length) setCurrent(sessions[0].id);
    if (!sessions.length) {
      const s = await apiStartSession({ persona: "General", model: "gpt-4o-mini" });
      setCurrent(s.id);
    }

    const cur = getCurrent();
    if (cur && !cur.id.startsWith("local_")) await loadHistoryIntoLocal(cur.id);

    renderSessionList();
    renderConversation();
  })();

  console.log(
    `%c🤖 ${APP} mastermind.js initialized — ${APP_VERSION}`,
    "background:#5bd0ff;color:#00121e;padding:4px 8px;border-radius:4px;font-weight:bold;"
  );
  debug("MASTER MIND PAGE LOADED", {
    app: APP, version: APP_VERSION, apiBase,
    sessions: sessions.length, currentId,
  });
});
