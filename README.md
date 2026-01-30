# 🌌 **ASTRA v2.0.0**

<p align="center">
  <img src="https://github.com/Akash-Kadali/ASTRA/blob/main/data/astra.png" alt="ASTRA Logo" width="700"/>
</p>

### *Autonomous System for Talent & Resume Automation*

**Author:** Sri Akash Kadali

> *“Intelligence that understands your profile, humanizes your story, and aligns every resume to the role.”*

---

## 📘 Overview

**ASTRA** (Autonomous System for Talent & Resume Automation) is a **modular AI ecosystem** built to optimize resumes, generate tailored cover letters, and assist with job applications end-to-end.
It combines **LaTeX-based automation**, **LLM reasoning**, and **humanized writing** into one unified app.

ASTRA runs locally as a **FastAPI + PyWebView desktop application**, providing a native ChatGPT-like experience with **persistent memory**, **LaTeX rendering**, and **analytics dashboards**.

---

## 🪐 ASTRA Submodules

ASTRA is composed of three core intelligent submodules:

| Submodule          | Description                                                                                                                              |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 🧠 **HIREX**       | *High Resume eXpert* — the core engine that builds, optimizes, and compiles ATS-friendly LaTeX resumes and cover letters.                |
| 🗣️ **SuperHuman** | The humanization engine that rewrites and enhances resume or cover letter content to sound natural, confident, and professionally human. |
| 💬 **MasterMind**  | The conversational reasoning assistant — a local ChatGPT-class model with session memory, tone control, and job-awareness.               |

Each of these submodules powers ASTRA’s integrated tools like the **Resume Optimizer**, **Cover Letter Generator**, and **Talk to ASTRA** modules.

---

### 🧩 Core Features

| Module                          | Purpose                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------ |
| 🧠 **MasterMind (Submodule)**   | ChatGPT-style reasoning assistant with persistent memory (RAG-like context).                     |
| 🗣️ **SuperHuman (Submodule)**  | AI text humanizer for resumes, cover letters, and interview answers.                             |
| 🧾 **HIREX (Submodule)**        | Resume optimizer built on LaTeX rendering and GPT-powered JD alignment.                          |
| 💬 **Talk to ASTRA**            | Job-aware Q&A system that answers recruiter/interview questions using saved JD + resume context. |
| ✍️ **CoverLetter Engine**       | Auto-drafts role-specific cover letters integrating SuperHuman rewrites.                         |
| 🧍 **Humanize (AIHumanize.io)** | Enhances LaTeX bullets (`\resumeItem{}`) for clarity and readability.                            |
| 📊 **Dashboard**                | Tracks tone, model usage, fit scores, and historical sessions.                                   |
| ⚙️ **Utils / Models Routers**   | Backend helpers for config, model lists, telemetry, and text utilities.                          |

---

## 🏗️ Project Structure

```
ASTRA/
│
├── backend/
│   ├── api/
│   │   ├── optimize.py          ← Resume optimizer / JD parser (HIREX core)
│   │   ├── coverletter.py       ← Cover letter generator
│   │   ├── talk.py              ← “Talk to ASTRA” Q&A endpoint
│   │   ├── superhuman.py        ← SuperHuman humanizer engine
│   │   ├── humanize.py          ← AIHumanize.io integration
│   │   ├── mastermind.py        ← MasterMind assistant backend
│   │   ├── dashboard.py         ← Analytics + trends
│   │   ├── context_store.py     ← JD + Resume memory store
│   │   ├── models_router.py     ← Model list + pricing
│   │   ├── utils_router.py      ← Helpers (ping, base64, escape)
│   │   └── debug.py             ← Frontend → backend logger
│   │
│   ├── core/
│   │   ├── config.py            ← Global paths, env, and defaults
│   │   ├── compiler.py          ← Secure pdflatex wrapper (HIREX compile)
│   │   ├── security.py          ← File + LaTeX validation
│   │   └── utils.py             ← Logging, hashing, diagnostics
│   │
│   └── data/
│       ├── contexts/            ← Saved JD + resume contexts
│       ├── history/             ← User activity JSONL
│       ├── logs/                ← Event logs for dashboard
│       └── mastermind_sessions/ ← Stored MasterMind chats
│
├── frontend/
│   ├── master.html              ← Main app UI
│   ├── master.js                ← Unified JS controller
│   ├── static/css/              ← Theme + layout
│   └── static/assets/           ← Icons, logos, favicon
│
├── main.py                      ← FastAPI + PyWebView launcher
└── requirements.txt
```

---

## ⚙️ Setup & Environment

### 1️⃣ Install dependencies

```bash
pip install fastapi uvicorn httpx openai python-dotenv pywebview pydantic
```

### 2️⃣ Environment variables (`.env`)

```bash
OPENAI_API_KEY=sk-xxxxxx
HUMANIZE_API_KEY=Bearer xxxxx
DEBUG=true
DEFAULT_MODEL=gpt-4o-mini
API_BASE_URL=http://127.0.0.1:8000
```

### 3️⃣ Run ASTRA

```bash
python main.py
```

**Launch sequence:**

* FastAPI backend starts at **localhost:8000**
* PyWebView opens the desktop UI
* Routers auto-register & mount static files
* Logs + chat sessions persist under `/backend/data/`

Then open:
👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🧠 Backend Modules Summary

### 🧾 `optimize.py` — (HIREX Submodule)

* Extracts **skills**, **courses**, and **roles** from job descriptions.
* Canonicalizes skills (e.g., *PyTorch → Data & ML*).
* Generates LaTeX-safe resume replacements.
* Uses GPT for smart section rebuilding.

### ✍️ `coverletter.py`

* Extracts company & role.
* Generates custom cover letter body.
* Humanizes tone via SuperHuman before PDF compilation.

### 💬 `talk.py` — “Talk to ASTRA”

* Contextual interview Q&A using stored JD + resume.
* Leverages MasterMind for reasoning + SuperHuman for tone.

### 🗣️ `superhuman.py` — (SuperHuman Submodule)

* Refines tone, style, and fluency for different text modes.
* Supports tone presets (`formal`, `conversational`, `academic`, etc.).
* Ensures LaTeX compatibility.

### 🧠 `mastermind.py` — (MasterMind Submodule)

* Persistent AI reasoning assistant.
* Supports personas, tones, and multi-turn sessions.
* Stores sessions in `/data/mastermind_sessions`.

### 🤖 `humanize.py`

* Uses AIHumanize.io to improve bullet clarity and balance.
* Brace-safe LaTeX parsing for `\resumeItem`.

### 🧾 `context_store.py`

* Saves combined JD + resume JSON bundles for reuse.
* Provides dashboard view of past applications.

### 📊 `dashboard.py`

* Aggregates log data for analytics.
* Displays trends and activity metrics.

### ⚙️ `utils_router.py` & `debug.py`

* Diagnostics, telemetry, text encoding, logging.

---

## 💻 Core Framework Files

| File               | Purpose                                 |
| ------------------ | --------------------------------------- |
| `core/config.py`   | Global constants, paths, ENV management |
| `core/compiler.py` | Safe LaTeX → PDF compiler               |
| `core/security.py` | File validation & LaTeX safety          |
| `core/utils.py`    | Logging, hashing, benchmarks            |

---

## 🖥️ Frontend Overview

* **`master.html`** — Single-page dark UI
* **`master.js`** — Event routing + API interaction
* Design inspired by ChatGPT (dark #0a1020 theme)

Main UI Tabs:

* Resume Optimizer (HIREX)
* Cover Letter Generator
* Talk to ASTRA (MasterMind + SuperHuman)
* Dashboard & History

---

## 💾 Data Directories

| Directory                    | Description               |
| ---------------------------- | ------------------------- |
| `data/logs/events.jsonl`     | Event logs                |
| `data/history/history.jsonl` | Usage history             |
| `data/contexts/`             | Saved JD + Resume bundles |
| `data/mastermind_sessions/`  | Chat session storage      |
| `data/cache/latex_builds/`   | Temporary LaTeX builds    |

---

## 🔐 Security

* Strict `.tex` file validation (≤5 MB)
* `pdflatex` runs in sandbox, no shell escape
* No external code execution
* All input goes through `secure_tex_input()`

---

## 📈 Logging & Analytics

Each event calls:

```python
log_event("event_name", {"meta": {...}})
```

→ stored in `/data/logs/events.jsonl`
→ visualized via `dashboard.py`

Common event types:

* `optimize_resume`
* `superhuman_rewrite`
* `talk_answer`
* `coverletter_draft`
* `frontend_debug`

---

## 🧱 Run Modes

| Mode                               | Description             |
| ---------------------------------- | ----------------------- |
| `python main.py`                   | Full desktop app (GUI)  |
| `uvicorn backend.api:app --reload` | API-only developer mode |
| `/api/docs`                        | Swagger API UI          |

---

## 🛠️ Roadmap (v2.2.x → v3.0)

* ✅ Resume Fit Scoring (JD ↔ Resume match %)
* 🌐 Cloud Sync for Contexts
* 💡 RAG Memory Retrieval for ASTRA chat
* 🪶 PDF → LaTeX converter
* 🔄 Live WebSocket MasterMind chat
* 📈 Skill Graph visualization

---

## 🪙 License & Attribution

Copyright © 2025 **Sri Akash Kadali**

Educational & research use permitted.
Trademarks: **ASTRA™, HIREX™, SuperHuman™, MasterMind™** belong to their respective author.

---
