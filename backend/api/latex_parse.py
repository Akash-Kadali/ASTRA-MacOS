"""
HIREX • api/latex_parse.py
Lightweight Resume Parser (Raw-Preserve Mode)
Extracts sections from LaTeX or text resumes with zero cleaning or normalization.
Also provides a safe (no-escape) cover-letter body injector with BODY anchors.

Purpose: Provide unaltered LaTeX/text blocks for AI-based optimization.
Author: Sri Akash Kadali
"""

from __future__ import annotations

import re
from typing import Dict, List, Any


# ============================================================
# ⚙️ Section Extraction Utilities
# ============================================================
def extract_section(tex: str, section_name: str) -> str:
    """
    Extracts raw LaTeX section content between \\section{<section_name>}
    (or \\section*{<section_name>}) and the next section, or a plaintext
    header line that looks like Title Case.

    No stripping or cleanup is applied.
    """
    # Accept \section or \section*, and a plain text header line
    pattern = (
        rf"(?:\\section\*?\{{{re.escape(section_name)}\}}"      # \section{Name} or \section*{Name}
        rf"|^(?:{re.escape(section_name)})\s*$)"                # or plain line "Name"
        r"(.*?)"                                                # capture content non-greedily
        r"(?=(?:\\section|\n[A-Z][A-Za-z ]+\n|$))"              # until next section/header or EOF
    )
    m = re.search(pattern, tex, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE)
    return m.group(1) if m else ""


# ============================================================
# 🧠 Main Parser (No Cleaning, No Normalization)
# ============================================================
def parse_latex_resume(tex_content: str) -> Dict[str, Any]:
    """
    Parses a LaTeX or plain-text resume into structured JSON form.
    All text is preserved as-is (no trimming, reformatting, or escaping).

    Extracts:
      - Education
      - Skills
      - Experience
      - Projects
      - Achievements
    """
    tex = tex_content.replace("\r", "")

    education_block    = extract_section(tex, "Education")
    skills_block       = extract_section(tex, "Skills")
    experience_block   = extract_section(tex, "Experience")
    projects_block     = extract_section(tex, "Projects")
    achievements_block = extract_section(tex, "Achievements")

    return {
        "education": _extract_bullets(education_block) or _split_lines(education_block),
        "skills": _parse_skills(skills_block),
        "experience": _parse_experience(experience_block),
        "projects": _parse_experience(projects_block),
        "achievements": _extract_bullets(achievements_block) or _split_lines(achievements_block),
    }


# ============================================================
# 🧩 Helper Parsers (Preserve Original Text)
# ============================================================
def _split_lines(block: str) -> List[str]:
    """Split section into lines — keeps all original spacing and symbols."""
    return [ln for ln in (block or "").splitlines() if ln.strip()]


def _extract_bullets(section: str) -> List[str]:
    """Extract bullet lines without reformatting or cleanup."""
    if not section:
        return []
    # \item, \item[] …, and plain bullets -, •
    bullets = re.findall(r"\\item(?:\[[^\]]*\])?\s+(.*)", section)
    if not bullets:
        bullets = re.findall(r"^[\-\u2022]\s+(.*)$", section, flags=re.MULTILINE)  # - or •
    return [b for b in bullets if str(b).strip()]


def _parse_experience(section: str) -> List[Dict[str, Any]]:
    """
    Extract Experience/Projects entries minimally.
    Preserves LaTeX formatting and avoids stripping or normalization.

    Supports common patterns:
      1) \\textbf{Role} \\hfill \\textit{Company} \\hfill Date
      2) \\textbf{Company} \\hfill \\textit{Role} \\hfill Date
      followed by an itemize block.
    """
    entries: List[Dict[str, Any]] = []
    if not section:
        return entries

    # Pattern 1: Role then Company
    pat_role_company = re.compile(
        r"\\textbf\{(?P<title>.*?)\}\s*\\hfill\s*(?:\\textit|\\emph)\{(?P<company>.*?)\}"
        r"(?:\s*\\hfill\s*(?P<date>[^\n]*))?"
        r"(?P<body>.*?)\\end\{itemize\}",
        flags=re.DOTALL,
    )

    # Pattern 2: Company then Role
    pat_company_role = re.compile(
        r"\\textbf\{(?P<company>.*?)\}\s*\\hfill\s*(?:\\textit|\\emph)\{(?P<title>.*?)\}"
        r"(?:\s*\\hfill\s*(?P<date>[^\n]*))?"
        r"(?P<body>.*?)\\end\{itemize\}",
        flags=re.DOTALL,
    )

    # Try role→company first; if none, try company→role
    matches = list(pat_role_company.finditer(section)) or list(pat_company_role.finditer(section))
    for m in matches:
        bullets = _extract_bullets(m.group("body"))
        entries.append({
            "company": m.group("company"),
            "title": m.group("title"),
            "date": (m.group("date") or "").strip(),
            "bullets": bullets,
        })

    if entries:
        return entries

    # Plain text fallback (preserves spacing)
    blocks = re.split(r"\n(?=[A-Z].*\d{4})", section)
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        header = lines[0]
        date_match = re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|[A-Za-z]+)?\.?\s*\d{4}.*\d{4}\b", header)
        date = date_match.group(0) if date_match else ""

        title, company = "", ""
        if " at " in header:
            title, company = header.split(" at ", 1)
        elif " - " in header:
            parts = header.split(" - ", 1)
            title = parts[0]
            company = parts[1]

        bullets = [ln for ln in lines[1:] if not re.match(r"^[A-Z][A-Za-z ]+$", ln)]
        entries.append({
            "company": company or "Unknown",
            "title": title or "Role",
            "date": date,
            "bullets": bullets,
        })
    return entries


def _parse_skills(section: str) -> Dict[str, List[str]]:
    """Extracts Skills lines as-is (no lowercase, trimming, or formatting)."""
    if not section:
        return {}
    lines = [l for l in section.splitlines() if l.strip()]
    skills_dict: Dict[str, List[str]] = {}
    for line in lines:
        if ":" in line:
            key, val = line.split(":", 1)
            values = [v.strip() for v in val.split(",") if v.strip()]
            if values:
                skills_dict[key.strip()] = values
    return skills_dict


# ============================================================
# ✉️ Cover-Letter Body Injector (No Escaping)
#  - Inserts body between BODY anchors if present
#  - Else injects just before \end{document}
#  - Strips accidental preamble/closing from body but does not escape/clean
# ============================================================
def inject_cover_body(base_tex: str, body_tex: str) -> str:
    if base_tex is None:
        base_tex = ""
    if body_tex is None:
        body_tex = ""

    # Strip preamble and closing from the body (keep raw content)
    body = re.sub(r"\\documentclass[\s\S]*?\\begin\{document\}", "", body_tex, flags=re.IGNORECASE)
    body = re.sub(r"\\end\{document\}\s*$", "", body, flags=re.IGNORECASE).strip()

    # Prefer explicit anchors
    anchor_rx = r"(%-+BODY-START-+%)(.*?)(%-+BODY-END-+%)"
    if re.search(anchor_rx, base_tex, flags=re.DOTALL):
        return re.sub(anchor_rx, lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", base_tex, flags=re.DOTALL)

    # Otherwise, inject right before \end{document}
    if re.search(r"\\end\{document\}\s*$", base_tex, flags=re.IGNORECASE):
        return re.sub(r"\\end\{document\}\s*$",
                      f"\n% (Auto-inserted by HIREX)\n{body}\n\\end{{document}}\n",
                      base_tex, flags=re.IGNORECASE)

    # Fallback: append a closing tag
    return base_tex.rstrip() + f"\n\n% (Auto-inserted by HIREX)\n{body}\n\\end{{document}}\n"


# ============================================================
# 🧪 Local Test
# ============================================================
if __name__ == "__main__":
    sample_resume = r"""
    \documentclass{article}
    \begin{document}

    %-----------EDUCATION-----------
    \section{Education}
    University of Maryland, College Park, United States CGPA: 3.55/4
    Master of Science in Applied Machine Learning August 2024 - May 2026
    • Relevant Coursework:

    %-----------EXPERIENCE-----------
    \section{Experience}
    \textbf{Machine Learning Intern} \hfill \textit{IIT Indore} \hfill May 2023 – Dec 2023
    \begin{itemize}
      \item Developed DeBERTa-based architecture for hate-speech detection.
      \item Improved accuracy using contrastive learning.
      \item Enhanced features with emotion embeddings.
    \end{itemize}

    \end{document}
    """

    from pprint import pprint
    pprint(parse_latex_resume(sample_resume))

    # Cover-letter injection quick check
    base = r"""
    \documentclass{article}
    \begin{document}
    Dear Hiring Manager,

    %-----------BODY-START-----------
    %-----------BODY-END-------------

    Sincerely,\\
    Your Name
    \end{document}
    """
    body = r"""\documentclass{article}\begin{document}
    This is my injected body. \LaTeX{} intact.
    \end{document}"""
    print("\n--- Injected Cover Letter ---")
    print(inject_cover_body(base, body))