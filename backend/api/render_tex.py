"""
HIREX • api/render_tex.py
Template renderer for final LaTeX resume output.
Simplified for direct LaTeX input (no JSON parsing).
Author: Sri Akash Kadali
"""

import re
from backend.core.utils import log_event


# ============================================================
# 🧠 Direct LaTeX Renderer
# ============================================================
def render_final_tex(final_tex: str) -> str:
    """
    Returns the LaTeX text exactly as received from GPT/Humanize,
    after performing minimal safety cleanup and normalization.

    Args:
        final_tex (str): Full LaTeX document string (already formatted)
    Returns:
        str: Safe LaTeX text ready for compilation
    """
    if not isinstance(final_tex, str):
        raise ValueError("render_final_tex() expects a LaTeX string input.")

    # --- Trim and clean any code-fence artifacts ---
    cleaned = (
        final_tex.replace("```latex", "")
        .replace("```", "")
        .strip()
    )

    # --- Normalize line endings ---
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    # pdflatex cannot consume several common Unicode symbols emitted by models.
    # Convert them to LaTeX before compilation while preserving visible meaning.
    unicode_tex = {
        "↔": r"$\leftrightarrow$", "→": r"$\rightarrow$", "←": r"$\leftarrow$",
        "≤": r"$\leq$", "≥": r"$\geq$", "×": r"$\times$", "±": r"$\pm$",
        "≈": r"$\approx$", "∞": r"$\infty$", "•": r"$\bullet$",
        "–": "--", "—": "--", "−": "-", "’": "'", "“": "``", "”": "''",
    }
    for symbol, replacement in unicode_tex.items():
        cleaned = cleaned.replace(symbol, replacement)

    # --- Remove stray leading/trailing blank lines ---
    cleaned = re.sub(r"^\s*\n", "", cleaned)
    cleaned = re.sub(r"\n\s*$", "\n", cleaned)

    # --- Basic validation warnings ---
    if not re.search(r"\\documentclass", cleaned):
        log_event("⚠️ [RENDER] Missing \\documentclass header.")
    if "\\begin{document}" not in cleaned:
        log_event("⚠️ [RENDER] Missing \\begin{document} block.")

    # --- Ensure proper closing tag ---
    if not cleaned.strip().endswith("\\end{document}"):
        cleaned += "\n\\end{document}\n"

    # --- Collapse excessive blank lines for cleanliness ---
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    log_event("✅ [RENDER] Final LaTeX render complete and safe for compilation.")
    return cleaned


# ============================================================
# 🧪 Local Test
# ============================================================
if __name__ == "__main__":
    sample_tex = r"""
\documentclass{article}
\begin{document}
Hello World

\section*{Education}
University of Maryland, College Park

\end{document}
"""
    print(render_final_tex(sample_tex))
