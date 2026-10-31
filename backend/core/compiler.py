"""
HIREX • core/compiler.py
Secure LaTeX compiler — converts .tex → .pdf in a sandboxed temp directory.
Prevents shell escapes, runs pdflatex with restricted flags.
Author: Sri Akash Kadali
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

from backend.core import config

# Robust logger wrapper (works with both new/old signatures)
try:
    from backend.core.utils import log_event as _core_log_event  # type: ignore

    def _elog(event: str, meta: dict | None = None) -> None:
        try:
            _core_log_event(event, meta or {})
        except TypeError:
            # Back-compat: older log_event(msg: str)
            _core_log_event(f"{event} {meta or ''}")  # type: ignore
except Exception:  # pragma: no cover
    def _elog(event: str, meta: dict | None = None) -> None:  # fallback to print
        print(event, meta or "")


# ============================================================
# 🧩 Safe PDF Compilation Utility
# ============================================================
_PDFLATEX_UNICODE = {
    "↔": r"$\leftrightarrow$", "→": r"$\rightarrow$", "←": r"$\leftarrow$",
    "⇒": r"$\Rightarrow$", "⇐": r"$\Leftarrow$", "⇔": r"$\Leftrightarrow$",
    "≤": r"$\leq$", "≥": r"$\geq$", "≠": r"$\neq$", "≈": r"$\approx$",
    "×": r"$\times$", "÷": r"$\div$", "±": r"$\pm$", "∞": r"$\infty$",
    "∑": r"$\sum$", "∏": r"$\prod$", "√": r"$\sqrt{}$", "∈": r"$\in$",
    "∉": r"$\notin$", "∂": r"$\partial$", "∇": r"$\nabla$", "°": r"$^\circ$",
    "•": r"$\bullet$", "·": r"$\cdot$", "…": r"\ldots{}",
    "–": "--", "—": "--", "−": "-", "‐": "-", "‑": "-",
    "’": "'", "‘": "'", "“": "``", "”": "''", " ": " ",
    "©": r"\textcopyright{}", "®": r"\textregistered{}", "™": "TM",
}


def _pdflatex_safe_unicode(source: str) -> str:
    """Convert model/user Unicode into deterministic pdflatex-safe text."""
    for symbol, replacement in _PDFLATEX_UNICODE.items():
        source = source.replace(symbol, replacement)
    output: list[str] = []
    for char in source:
        if ord(char) < 128 or char in "\n\r\t":
            output.append(char)
            continue
        # inputenc handles ordinary Latin text, but decomposing it removes a
        # common source of font/engine variance in packaged builds.
        transliterated = unicodedata.normalize("NFKD", char).encode("ascii", "ignore").decode()
        if transliterated:
            output.append(transliterated)
        elif unicodedata.category(char).startswith(("C", "S")):
            # Drop emoji, variation selectors, and invisible controls. They do
            # not belong in an ATS resume and often stop pdflatex completely.
            continue
        else:
            output.append(" ")
    return "".join(output)


def compile_latex_safely(tex_string: str, passes: int = 2) -> bytes | None:
    """
    Compiles LaTeX source code into PDF bytes securely.
    Returns PDF bytes (on success) or None (on failure).

    Security & Stability:
    - Uses sandboxed temp directory under config.TEMP_LATEX_DIR
    - Disables shell escape and restricts file I/O (openin/openout "p")
    - Runs pdflatex twice for stable references
    - Cleans up temporary files automatically
    - Compatible with TeX Live / MiKTeX on all OS
    """
    passes = max(1, min(2, int(passes)))
    pdflatex_path = shutil.which("pdflatex")
    if pdflatex_path is None:
        _elog("latex_pdflatex_missing", {"detail": "pdflatex not found in PATH"})
        return None

    # Ensure our build root exists (tempfile will use it as parent)
    try:
        Path(config.TEMP_LATEX_DIR).mkdir(parents=True, exist_ok=True)
    except Exception:
        # If TEMP_LATEX_DIR is missing or invalid, fall back to system tmp
        pass

    # Safer TeX environment
    env = os.environ.copy()
    # Prevent TeX from reading/writing outside the working directory
    env.setdefault("openout_any", "p")   # p = paranoid (write only in cwd)
    env.setdefault("openin_any", "p")    # p = paranoid (read only in cwd)
    # Explicitly disable shell escape
    env.setdefault("shell_escape", "f")  # f = false
    # Keep logs readable but bounded
    env.setdefault("max_print_line", "1000")

    try:
        temp_root = getattr(config, "TEMP_LATEX_DIR", None)
        with tempfile.TemporaryDirectory(dir=str(temp_root) if temp_root else None) as tmpdir:
            tmp = Path(tmpdir)
            tex_path = tmp / "main.tex"
            pdf_path = tmp / "main.pdf"
            log_path = tmp / "compile.log"

            # Write LaTeX source after a compiler-level safety pass. This is a
            # final guard shared by resumes, cover letters, and edited source.
            safe_tex = _pdflatex_safe_unicode(tex_string)
            tex_path.write_text(safe_tex, encoding="utf-8")
            _elog("latex_compile_start", {"workdir": str(tmp)})

            # Safe compile command (explicitly disable shell-escape)
            # IMPORTANT: use relative filename (main.tex), not absolute path,
            # so TeX in paranoid mode (openin_any=p) will read it.
            cmd = [
                pdflatex_path,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                "-no-shell-escape",
                "-synctex=0",
                tex_path.name,  # <--- changed from str(tex_path)
            ]

            # Run pdflatex twice for cross-refs
            for i in range(passes):
                proc = subprocess.run(
                    cmd,
                    cwd=tmp,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=90,
                    encoding="utf-8",
                    errors="ignore",
                    check=False,
                    env=env,
                )
                # Append compiler output
                try:
                    with open(log_path, "a", encoding="utf-8") as lf:
                        lf.write(proc.stdout or "")
                        lf.write("\n")
                except Exception:
                    pass

                _elog("latex_pdflatex_pass", {"pass": i + 1, "returncode": proc.returncode})
                if proc.returncode != 0:
                    # A package can surface a Unicode character only after macro
                    # expansion. Repair every reported code point and retry once.
                    codes = re.findall(r"Unicode character .*?\(U\+([0-9A-Fa-f]{4,6})\)", proc.stdout or "")
                    repaired = False
                    for code in codes:
                        char = chr(int(code, 16))
                        replacement = _PDFLATEX_UNICODE.get(char, "")
                        if char in safe_tex:
                            safe_tex = safe_tex.replace(char, replacement)
                            repaired = True
                    if repaired:
                        tex_path.write_text(safe_tex, encoding="utf-8")
                        retry = subprocess.run(
                            cmd, cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=90, encoding="utf-8", errors="ignore", check=False, env=env,
                        )
                        with open(log_path, "a", encoding="utf-8") as lf:
                            lf.write("\n--- ASTRA automatic Unicode repair retry ---\n")
                            lf.write(retry.stdout or "")
                        _elog("latex_unicode_repair_retry", {"returncode": retry.returncode, "codepoints": codes})
                        proc = retry
                    if proc.returncode != 0:
                        break

            # Check PDF output
            if pdf_path.exists():
                pdf_bytes = pdf_path.read_bytes()
                size_kb = len(pdf_bytes) / 1024
                _elog("latex_pdf_built", {"size_kb": round(size_kb, 1)})
                return pdf_bytes

            # Error fallback — show last ~20 lines of log
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                tail = "\n".join(lines[-20:])
                _elog("latex_no_pdf", {"tail": tail})
                try:
                    persistent_log = Path(config.CACHE_DIR) / "last_latex_compile.log"
                    persistent_log.write_text("\n".join(lines), encoding="utf-8")
                except Exception:
                    pass
            else:
                _elog("latex_no_log", {"detail": "Compilation failed — no log file created"})
            return None

    except subprocess.TimeoutExpired:
        _elog("latex_timeout", {"limit_seconds": 90})
        return None
    except Exception as e:
        _elog("latex_unexpected_error", {"error": str(e)})
        return None


# ============================================================
# 🧪 Local Test
# ============================================================
if __name__ == "__main__":
    sample_tex = r"""
    \documentclass{article}
    \begin{document}
    Hello World! This is a HIREX LaTeX compile test.
    \end{document}
    """
    result = compile_latex_safely(sample_tex)
    print("✅ PDF generated:", bool(result))
