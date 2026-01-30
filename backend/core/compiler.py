"""
HIREX • core/compiler.py
Secure LaTeX compiler — converts .tex → .pdf in a sandboxed temp directory.
Prevents shell escapes, runs pdflatex with restricted flags.
Author: Sri Akash Kadali
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
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
def compile_latex_safely(tex_string: str) -> bytes | None:
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

            # Write LaTeX source
            tex_path.write_text(tex_string, encoding="utf-8")
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
            for i in range(2):
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
