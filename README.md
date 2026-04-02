# 🌌 ASTRA

<p align="center">
  <img src="https://github.com/Akash-Kadali/ASTRA-MacOS/blob/main/data/test2.png" alt="ASTRA Logo" width="700"/>
</p>

<p align="center">
  <b>Autonomous System for Talent & Resume Automation</b>
</p>

<p align="center">
  Local-first AI system for ATS resume optimization, cover letter generation, job-aware chat, and LaTeX-safe document workflows.
</p>

---

## 📘 Overview

**ASTRA** is a **local-first, modular AI ecosystem** built to help candidates generate stronger, more targeted application materials with a desktop-style experience.

It combines:

- **JD-aware resume optimization**
- **Role-specific cover letter generation**
- **Humanized writing support**
- **Persistent job-aware chat**
- **Safe LaTeX compilation with PDF output**
- **Local history, logs, and reusable session context**

ASTRA runs through a **FastAPI backend + PyWebView desktop application**, giving a native app-like workflow with preview, saved sessions, and structured output generation.

---

## 🪐 ASTRA Subsystems

| Subsystem | Description |
|---|---|
| 🧠 **HIREX** | Core resume optimization engine. Parses job descriptions, rewrites LaTeX resumes, improves ATS alignment, and compiles final PDFs. |
| 🗣️ **SuperHuman** | Humanization layer that rewrites content to sound natural, professional, and recruiter-friendly while staying LaTeX-safe. |
| 💬 **MasterMind** | Job-aware conversational assistant with persistent memory, context retention, and multi-turn support. |

---

## ✨ What ASTRA Does

ASTRA is designed to generate better job application outputs, not just generic text.

### Core capabilities

- Optimize **ATS-friendly LaTeX resumes**
- Generate **job-specific cover letters**
- Provide **context-aware Q&A** using saved JD and resume context
- Maintain **persistent local chat sessions**
- Track **history, analytics, and output metadata**
- Compile documents through a **safe LaTeX pipeline**

---

## 🧩 Main Features

| Module | Purpose |
|---|---|
| 🧾 **Resume Optimizer** | Rewrites LaTeX resumes using JD-aware bullet planning, skill alignment, and one-page fit control. |
| ✍️ **Cover Letter Engine** | Produces tailored cover letters using company, role, and resume context. |
| 💬 **Talk to ASTRA** | Lets users ask questions about a saved JD, resume, or generated output. |
| 🗣️ **SuperHuman** | Humanizes bullets, responses, and cover letters into more natural language. |
| 🧠 **MasterMind** | Stores and manages conversation context across multiple sessions. |
| 📊 **Dashboard** | Tracks usage patterns, history, fit trends, and output activity. |
| ⚙️ **Utilities / Routers** | Handles config, model routing, validation, telemetry, and helper functions. |

---

## 🧠 Resume Optimizer Highlights

The current optimizer is not a simple keyword replacer. It performs a multi-stage optimization pipeline designed for stronger final resume quality.

### Resume optimization pipeline

- Extracts **must-have, should-have, and nice-to-have** skills from the JD
- Classifies the role into a **role archetype**
- Decomposes the JD into **day-to-day tasks**
- Extracts **JD key phrases** that should appear naturally in the resume
- Profiles the **ideal candidate**
- Builds a **12-bullet master plan** across experience sections
- Generates **JD-mirrored project entries**
- Runs **post-generation validation and repair**
- Rebuilds the **Skills section** using JD-priority ranking
- Applies **safe one-page trimming**
- Injects **PDF metadata**
- Compiles the final LaTeX safely into PDF

### Output-quality features

- **Task-aware bullet generation**
- **JD phrase mirroring**
- **Metric diversity enforcement**
- **Placeholder cleanup** for bad generations like `XYZ`, `ABC`, `Foo`, `Lorem`, etc.
- **Cross-block bullet deduplication**
- **Bullet quality scoring and improvement**
- **Coverage remediation** for missing must-have JD terms
- **Skills prioritization** based on JD relevance
- **One-page preservation rules** so experience bullets are not over-trimmed
- **Projects section generation** in `\textbf{Title} -- one-liner` format

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
│       ├── mastermind_sessions/
│       └── cache/
│
├── frontend/
│   ├── master.html
│   ├── master.js
│   ├── static/css/
│   └── static/assets/
│
├── main.py
└── requirements.txt
````

---

## ⚙️ Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is still incomplete, install the core packages manually:

```bash
pip install fastapi uvicorn httpx openai python-dotenv pywebview pydantic
```

---

### 2. Configure environment variables

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-xxxxxx
HUMANIZE_API_KEY=Bearer xxxxx
DEBUG=true
DEFAULT_MODEL=gpt-4o-mini
API_BASE_URL=http://127.0.0.1:8000
```

You may also configure default file paths and local storage directories inside `backend/core/config.py`.

---

### 3. Run ASTRA

```bash
python main.py
```

### Launch behavior

* FastAPI backend starts on `127.0.0.1:8000`
* PyWebView opens the desktop UI
* Sessions, logs, and generated artifacts persist locally

Useful endpoints:

* `http://127.0.0.1:8000`
* `http://127.0.0.1:8000/api/docs`

---

## 🧾 Backend Modules

### `optimize.py` — HIREX Core

Main resume optimization engine.

Responsibilities include:

* extracting JD requirements
* identifying role type and tone
* planning and rewriting experience bullets
* generating project entries
* rebuilding the skills section
* safe LaTeX compilation
* one-page fitting logic
* resume coverage and alignment reporting

---

### `coverletter.py`

Generates role-specific cover letters using:

* JD context
* target company and role
* existing resume profile
* safe output formatting

---

### `talk.py`

Supports contextual Q&A tied to:

* the current JD
* saved resume context
* previous ASTRA sessions
* generated outputs

---

### `superhuman.py`

Applies humanization and tone control for:

* resume bullets
* cover letters
* answers
* recruiter-facing writing

Designed to keep outputs natural without breaking LaTeX structure.

---

### `mastermind.py`

Provides:

* persistent local session storage
* multi-turn reasoning
* contextual conversational support
* reusable job-aware memory

---

### `context_store.py`

Stores reusable context bundles such as:

* JD text
* selected resume state
* generated outputs
* prior chat state

---

### `dashboard.py`

Aggregates usage and application-related signals such as:

* generation frequency
* session activity
* optimization history
* fit-score or coverage trends

---

## 💾 Local Data Storage

| Directory                            | Purpose                   |
| ------------------------------------ | ------------------------- |
| `backend/data/logs/events.jsonl`     | Event logs                |
| `backend/data/history/history.jsonl` | Output and usage history  |
| `backend/data/contexts/`             | Saved JD + resume bundles |
| `backend/data/mastermind_sessions/`  | Persistent chat sessions  |
| `backend/data/cache/latex_builds/`   | Temporary LaTeX builds    |

---

## 🔐 Security and Safety

ASTRA uses a defensive document pipeline:

* strict `.tex` validation
* secure input filtering
* temporary sandboxed LaTeX build directory
* no shell escape during compile
* controlled local file handling
* LaTeX-safe escaping before rendering

This helps reduce broken builds and unsafe LaTeX execution patterns.

---

## 📈 Logging and Analytics

ASTRA uses structured event logging for local observability.

Example:

```python
log_event("event_name", {"meta": {...}})
```

Common event types:

* `optimize_resume`
* `coverletter_draft`
* `talk_answer`
* `superhuman_rewrite`
* `frontend_debug`

Logs are stored in:

```text
backend/data/logs/events.jsonl
```

---

## 🧱 Run Modes

| Mode              | Command                            |
| ----------------- | ---------------------------------- |
| Full desktop app  | `python main.py`                   |
| API-only dev mode | `uvicorn backend.api:app --reload` |
| API docs          | `/api/docs`                        |

---

## 🚀 Roadmap

Planned upgrades include:

* richer fit scoring and recruiter-style evaluation
* better session retrieval and memory search
* PDF-to-LaTeX conversion
* streaming generation
* dashboard improvements
* stronger artifact versioning
* more advanced analytics for resume evolution

---

## 🪙 License and Attribution

Copyright © 2025–2026 **Sri Akash Kadali**

Educational and research use permitted.

**ASTRA™, HIREX™, SuperHuman™, MasterMind™** are associated with the author’s project ecosystem.

---

## 👤 Author

**Sri Akash Kadali**

Applied Machine Learning Graduate Student
University of Maryland

> *Intelligence that understands your profile, aligns your story to the role, and improves resume output quality end to end.*
