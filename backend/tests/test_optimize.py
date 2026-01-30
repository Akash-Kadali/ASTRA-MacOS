"""
HIREX • tests/test_optimize.py
Minimal end-to-end tests for API + static frontend.
Run:  pytest -q
Author: Sri Akash Kadali
"""

from io import BytesIO
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


# ============================================================
# 🔧 Sample Minimal LaTeX Resume
# ============================================================

def _sample_tex() -> str:
    """Returns a minimal valid LaTeX resume for test upload."""
    return r"""
\documentclass[letterpaper,10pt]{article}
\usepackage[margin=0.6in]{geometry}
\begin{document}
\section{Education}
\begin{itemize}
\item B.S. in Computer Science, XYZ University
\end{itemize}

\section{Skills}
Languages: Python, SQL
Cloud: AWS, Docker

\section{Experience}
\textbf{Acme Corp} \hfill \textit{Software Intern} \hfill 2024
\begin{itemize}
\item worked on python data pipelines
\item helped with docker deployment
\end{itemize}
\end{document}
""".strip()


# ============================================================
# 🧠 API Health & Frontend Tests
# ============================================================

def test_health_endpoint():
    """Ensure /health endpoint is alive and reports version."""
    r = client.get("/health")
    assert r.status_code == 200, f"Health failed: {r.text}"
    data = r.json()
    assert data.get("status") == "ok"
    assert "version" in data


def test_frontend_index_served():
    """Ensure root / serves the frontend HTML file."""
    r = client.get("/")
    assert r.status_code == 200, f"Frontend index not served: {r.text}"
    assert "text/html" in r.headers.get("content-type", "")


# ============================================================
# 🚀 /api/optimize End-to-End Tests
# ============================================================

def test_optimize_endpoint_happy_path(tmp_path: Path):
    """
    Verify that /api/optimize processes a sample resume and JD text
    end-to-end through OpenAI + Humanize pipeline.
    """
    tex_bytes = _sample_tex().encode("utf-8")
    files = {"base_resume_tex": ("resume.tex", BytesIO(tex_bytes), "text/plain")}
    data = {"jd_text": "We need Python, AWS, FastAPI. Seniority: SWE I or Senior."}

    r = client.post("/api/optimize", files=files, data=data)
    assert r.status_code == 200, f"Unexpected response: {r.text}"
    payload = r.json()

    # Check essential keys in response
    for key in ("tex_string", "jd_summary", "coverage", "pdf_base64"):
        assert key in payload, f"Missing key in response: {key}"

    # Validate JD summary structure
    jd = payload["jd_summary"]
    for key in ("must_haves", "skills", "metrics", "seniority"):
        assert key in jd, f"Missing JD key: {key}"

    # Ensure coverage includes at least Python (case-insensitive)
    items = {c["item"].lower() for c in payload["coverage"]}
    assert "python" in items, f"Coverage missing Python: {items}"

    # PDF presence check
    pdf_b64 = payload.get("pdf_base64")
    assert isinstance(pdf_b64, (str, type(None)))
    if pdf_b64:
        assert len(pdf_b64) > 50, "Encoded PDF too small (possibly truncated)."


def test_optimize_endpoint_missing_fields():
    """Missing JD text should result in a validation or bad request error."""
    tex_bytes = _sample_tex().encode("utf-8")
    files = {"base_resume_tex": ("resume.tex", BytesIO(tex_bytes), "text/plain")}
    r = client.post("/api/optimize", files=files, data={})
    assert r.status_code in (400, 422), f"Expected validation failure, got {r.status_code}"


def test_optimize_endpoint_invalid_file():
    """Invalid extension or empty content should not crash the API."""
    files = {"base_resume_tex": ("resume.txt", BytesIO(b""), "text/plain")}
    r = client.post("/api/optimize", files=files, data={"jd_text": "Python"})
    assert r.status_code in (400, 500), f"Unexpected status: {r.status_code}"
