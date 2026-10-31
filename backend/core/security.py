"""
HIREX • core/security.py (v1.0.0)
Security and validation utilities for uploaded files and user input.

⚠️ This build preserves full LaTeX content (no sanitization or macro stripping).
It keeps a backward-compatible `secure_tex_input(...)` API:

  • secure_tex_input(text: str) -> str
      Pass-through (returns text unchanged).

  • secure_tex_input(filename: str, content: bytes|bytearray|str) -> str
      Validates file (extension/size/non-empty), decodes to UTF-8, and returns raw LaTeX.

Author: Sri Akash Kadali
"""

from __future__ import annotations

import os
from typing import Any, Union

from backend.core import config
from backend.core.utils import safe_filename, log_event


# ============================================================
# ⚙️ File Validation
# ============================================================
def validate_file(upload_name: str, content: Union[bytes, bytearray, str]) -> str:
    """
    Validate uploaded file before use.
    Returns sanitized filename if safe, else raises ValueError.
    """
    if not upload_name:
        raise ValueError("❌ Missing filename in upload.")

    _, ext = os.path.splitext(upload_name)
    allowed = getattr(config, "ALLOWED_EXTENSIONS", {".tex"})
    if ext.lower() not in allowed:
        raise ValueError(f"❌ Invalid file extension: {ext} (allowed: {', '.join(sorted(allowed))})")

    # normalize to bytes for size & emptiness checks
    raw = content if isinstance(content, (bytes, bytearray)) else str(content).encode("utf-8", "ignore")

    size_mb = len(raw) / (1024 * 1024)
    max_mb = float(getattr(config, "MAX_UPLOAD_MB", 5))
    if size_mb > max_mb:
        raise ValueError(f"❌ File exceeds {max_mb:.0f} MB limit (got {size_mb:.2f} MB).")

    if not raw.strip():
        raise ValueError("❌ Uploaded file is empty.")

    safe_name = safe_filename(upload_name)
    # log_event supports string + optional meta signature across builds
    log_event(f"✅ File validated: {safe_name} ({size_mb:.2f} MB)")
    return safe_name


# ============================================================
# 🧩 Pass-through LaTeX (No Sanitization)
# ============================================================
def secure_tex_input(*args: Any) -> str:
    """
    Backward-compatible pass-through.

    Usage A (strings in code paths — preserve as-is):
        secure_tex_input(text: str) -> str

    Usage B (uploads — validate & decode):
        secure_tex_input(filename: str, content: bytes|bytearray|str) -> str
    """
    # --- Usage A: single arg (plain text) ---
    if len(args) == 1:
        text = args[0]
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        # No escaping/sanitization — preserve full LaTeX/content
        return text

    # --- Usage B: two args (filename + raw content) ---
    if len(args) == 2:
        filename, content = args[0], args[1]
        _ = validate_file(str(filename), content)

        # Decode to UTF-8 lossily; preserve bytes that decode
        if isinstance(content, (bytes, bytearray)):
            tex = content.decode("utf-8", errors="ignore")
        elif isinstance(content, str):
            tex = content
        else:
            raise TypeError("Unsupported content type for LaTeX input. Expected bytes or str.")

        log_event(f"✅ Raw LaTeX preserved and validated for: {filename}")
        return tex

    # --- Invalid usage ---
    raise TypeError(
        "secure_tex_input expects either (text: str) OR (filename: str, content: bytes|bytearray|str)"
    )


# ============================================================
# 🧪 Local Test
# ============================================================
if __name__ == "__main__":
    example_bytes = br"""
    \documentclass{article}
    \begin{document}
    Hello \textbf{World!} This content should remain unchanged.
    \input{my_commands.tex}
    \end{document}
    """

    try:
        # File path usage
        raw_from_file = secure_tex_input("resume.tex", example_bytes)
        print("==== Raw (file) Output ====")
        print(raw_from_file)

        # Plain text usage
        raw_text = secure_tex_input(r"\section*{Projects} \textit{Keep everything intact.}")
        print("\n==== Raw (text) Output ====")
        print(raw_text)
    except Exception as e:
        print("ERROR:", e)
