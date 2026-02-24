# 🌌 **ASTRA v1.0.0**

<p align="center">
  <img src="https://github.com/Akash-Kadali/ASTRA-MacOS/blob/main/data/test2.png" alt="ASTRA Logo" width="700"/>
</p>

### *Autonomous System for Talent & Resume Automation*

**Author:** Sri Akash Kadali

> *“Intelligence that understands your profile, humanizes your story, and aligns every resume to the role.”*

---

## 📘 Overview

**ASTRA** (Autonomous System for Talent & Resume Automation) is a **local-first, modular AI ecosystem** designed to:

* Optimize **ATS-friendly LaTeX resumes**
* Generate **role-specific cover letters**
* Provide a **job-aware chat assistant**
* Maintain **persistent sessions + analytics logs** locally

ASTRA runs as a **FastAPI backend + PyWebView desktop app**, delivering a native ChatGPT-like UI with PDF preview, saved history, and safe LaTeX compilation.

---

## 🪐 ASTRA Submodules

| Submodule          | Description                                                                                                    |
| ------------------ | -------------------------------------------------------------------------------------------------------------- |
| 🧠 **HIREX**       | *High Resume eXpert* — core engine for JD parsing, LaTeX resume optimization, and PDF compilation.             |
| 🗣️ **SuperHuman** | Humanization engine that rewrites bullets/sections to sound natural and professional while staying LaTeX-safe. |
| 💬 **MasterMind**  | Job-aware conversational assistant with session memory and tone control.                                       |

---

## 🧩 Core Features

| Module                          | Purpose                                                           |
| ------------------------------- | ----------------------------------------------------------------- |
| 🧠 **MasterMind (Submodule)**   | Chat assistant with persistent memory (session storage).          |
| 🗣️ **SuperHuman (Submodule)**  | Humanizes resume bullets, cover letters, and interview answers.   |
| 🧾 **HIREX (Submodule)**        | JD-aligned ATS resume optimization using LaTeX-safe replacements. |
| 💬 **Talk to ASTRA**            | Contextual Q&A using saved JD + resume bundles.                   |
| ✍️ **CoverLetter Engine**       | Generates role-specific cover letters from templates + context.   |
| 🧍 **Humanize (AIHumanize.io)** | Optional external humanizer for `\resumeItem{}` bullet upgrades.  |
| 📊 **Dashboard**                | Tracks usage, fit score trends, sessions, and output history.     |
| ⚙️ **Utils / Routers**          | Config, model routing, telemetry/logging, helpers.                |

---

## 🏗️ Project Structure

```text
ASTRA/
│
├── backend/
│   ├── api/
│   │   ├── optimize.py
│   │   ├── coverletter.py
│   │   ├── talk.py
│   │   ├── superhuman.py
│   │   ├── humanize.py
│   │   ├── mastermind.py
│   │   ├── dashboard.py
│   │   ├── context_store.py
│   │   ├── models_router.py
│   │   ├── utils_router.py
│   │   └── debug.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── compiler.py
│   │   ├── security.py
│   │   └── utils.py
│   │
│   └── data/
│       ├── contexts/
│       ├── history/
│       ├── logs/
│       └── mastermind_sessions/
│
├── frontend/
│   ├── master.html
│   ├── master.js
│   ├── static/css/
│   └── static/assets/
│
├── main.py
└── requirements.txt
```

---

## ⚙️ Setup & Environment

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

If you don’t have a finalized `requirements.txt` yet:

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

**Launch behavior:**

* FastAPI backend starts on **127.0.0.1:8000**
* PyWebView opens the desktop UI
* Logs + sessions persist under `backend/data/`

Open:

* `http://127.0.0.1:8000`
* `http://127.0.0.1:8000/api/docs` (Swagger)

---

## 🧠 Backend Modules Summary

### 🧾 `optimize.py` — HIREX Core

* Extracts role requirements from JD (skills, tooling, keywords, expectations)
* Produces LaTeX-safe, ATS-friendly edits
* Compiles output with secure LaTeX pipeline

### ✍️ `coverletter.py`

* Extracts company + role context
* Generates role-specific cover letter
* Uses templates + safe LaTeX compile

### 💬 `talk.py` — Talk to ASTRA

* JD + resume context-based interview Q&A
* Uses MasterMind reasoning + SuperHuman tone control

### 🗣️ `superhuman.py`

* Tone presets (formal, conversational, concise, etc.)
* Ensures LaTeX compatibility and avoids brittle formatting breaks

### 🧠 `mastermind.py`

* Persistent chat sessions saved locally
* Supports multi-turn reasoning tied to job context

### 🧾 `context_store.py`

* Saves combined JD + resume bundles for reuse and history tracking

### 📊 `dashboard.py`

* Aggregates logs into analytics signals (activity, usage, trends)

---

## 💾 Data Directories

| Directory                            | Description               |
| ------------------------------------ | ------------------------- |
| `backend/data/logs/events.jsonl`     | Event logs                |
| `backend/data/history/history.jsonl` | Usage history             |
| `backend/data/contexts/`             | Saved JD + Resume bundles |
| `backend/data/mastermind_sessions/`  | Stored chats              |
| `backend/data/cache/latex_builds/`   | Temporary LaTeX builds    |

---

## 🔐 Security

* Strict `.tex` validation (size + extension rules)
* `pdflatex` runs in a sandboxed temp build directory
* No shell escape
* Inputs pass through LaTeX safety checks before compile

---

## 📈 Logging & Analytics

Events use:

```python
log_event("event_name", {"meta": {...}})
```

Stored in:

* `backend/data/logs/events.jsonl`

Example events:

* `optimize_resume`
* `superhuman_rewrite`
* `talk_answer`
* `coverletter_draft`
* `frontend_debug`

---

## 🧱 Run Modes

| Mode                   | Command                            |
| ---------------------- | ---------------------------------- |
| Full desktop app (GUI) | `python main.py`                   |
| API-only dev mode      | `uvicorn backend.api:app --reload` |
| API docs               | `/api/docs`                        |

---

## 🛠️ Roadmap (v1.x → v2.0)

Planned upgrades after v1.0.0:

* Resume Fit Scoring (JD ↔ Resume match %)
* Better memory retrieval (RAG-style) for MasterMind
* PDF → LaTeX converter
* WebSocket streaming chat
* Skill graph visualization + richer dashboard analytics

---

## 🪙 License & Attribution

Copyright © 2025–2026 **Sri Akash Kadali**

Educational & research use permitted.
Trademarks: **ASTRA™, HIREX™, SuperHuman™, MasterMind™** belong to their respective author.

---
