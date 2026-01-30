"""
============================================================
 ASTRA v2.1.2 — main.py (Web Server + Browser Launcher)
 Automated System for Tailored Resume Applications.
 ------------------------------------------------------------
 Runs FastAPI + serves the frontend, then opens the default
 browser to http://<host>:<port> once /health is ready.

 Features:
   • Graceful startup with backend health polling
   • Auto-open browser (disable via ASTRA_NO_BROWSER=1)
   • Auto router discovery (import fallbacks) + logging (quiet by default)
   • Web-friendly CORS + tracing middleware (skips /health)
   • PyInstaller-friendly paths (MEIPASS)
   • .env loader (root + backend) for EXE and dev
   • System shutdown endpoint (/api/system/shutdown)
   • 🔁 Multi-instance cluster via ASTRA_INSTANCES
   • 🧵 Per-instance uvicorn workers via ASTRA_WORKERS

 Author: Sri Akash Kadali
============================================================
"""

# ============================== Path Setup ==============================
import os, sys, time, signal, threading, importlib, webbrowser, urllib.request, socket
import subprocess, tempfile
from contextlib import closing
from typing import Optional, Dict, Any, Tuple, Iterable, Set
from pathlib import Path

MEIPASS = getattr(sys, "_MEIPASS", None)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # ROOT (ASTRA)
ROOT_DIR = MEIPASS or CURRENT_DIR
# Ensure our packages are importable
for p in (ROOT_DIR, os.path.join(ROOT_DIR, "backend")):
    if p not in sys.path:
        sys.path.append(p)

# -------- tiny debug log (always writes to temp file, even --windowed) --
AUTOPEN_LOG = Path(tempfile.gettempdir()) / "astra_autopen.log"
def _dlog(msg: str) -> None:
    try:
        AUTOPEN_LOG.write_text(
            (AUTOPEN_LOG.read_text(encoding="utf-8") if AUTOPEN_LOG.exists() else "")
            + f"{time.strftime('%H:%M:%S')} | {msg}\n",
            encoding="utf-8",
        )
    except Exception:
        pass

# ============================== .env loader =============================
def _load_env_files():
    def _apply_env(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k, v)
        except Exception:
            pass

    # Root .env and backend/.env
    for rel in (".env", os.path.join("backend", ".env")):
        p = os.path.join(ROOT_DIR, rel)
        if os.path.exists(p):
            _apply_env(p)

    # When packaged, also check MEIPASS copies
    if MEIPASS:
        for rel in (".env", os.path.join("backend", ".env")):
            p = os.path.join(MEIPASS, rel)
            if os.path.exists(p):
                _apply_env(p)

_load_env_files()

# Verbosity toggles (quiet by default)
VERBOSE = os.getenv("ASTRA_VERBOSE", "0") == "1"
CORE_LOG = os.getenv("ASTRA_CORE_LOG", "0") == "1"

# ====================== FastAPI Backend + Config ========================
from fastapi import FastAPI, Request, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

try:
    from backend.core import config  # type: ignore
except Exception:
    class _Shim: APP_VERSION = "2.1.2"
    config = _Shim()  # type: ignore

APP_VERSION = getattr(config, "APP_VERSION", "2.1.2")

app = FastAPI(
    title="ASTRA API",
    description="Automated System for Tailored Resume Applications — Backend Service",
    version=APP_VERSION,
)

from backend.api import debug
app.include_router(debug.router)

# Critical: disable Starlette’s automatic /path <-> /path/ redirects
try:
    app.router.redirect_slashes = False  # type: ignore[attr-defined]
except Exception:
    pass

# ======================= Logging Helper (quiet by default) =============
def _fallback_log(event: str, meta: Optional[Dict[str, Any]] = None) -> None:
    if VERBOSE:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {event} {meta or ''}")
    _dlog(f"{event} {meta or ''}")

try:
    from backend.core.utils import log_event as _core_log_event  # type: ignore
    def _elog(event: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if CORE_LOG:
            try:
                _core_log_event(event, meta or {})
            except TypeError:
                _core_log_event(f"{event} {meta or ''}")  # type: ignore
        if VERBOSE:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] {event} {meta or ''}")
        _dlog(f"{event} {meta or ''}")
except Exception:
    _elog = _fallback_log  # type: ignore

# ============== Router Auto-Import (+ static hints for EXE) =============
for _modname in (
    "backend.api.optimize",
    "backend.api.coverletter",
    "backend.api.talk",
    "backend.api.superhuman",
    "backend.api.humanize",
    "backend.api.mastermind",
    "backend.api.dashboard",
    "backend.api.models_router",
    "backend.api.context_store",
    "backend.api.utils_router",
    "backend.api.debug",
):
    try:
        importlib.import_module(_modname)
    except Exception:
        pass

def _safe_import(module: str):
    for mod_path in (f"backend.api.{module}", f"api.{module}", module):
        try:
            mod = importlib.import_module(mod_path)
            if hasattr(mod, "router"):
                _elog("router_loaded", {"module": mod_path})
                return mod
        except Exception as e:
            _elog("router_import_error", {"module": mod_path, "error": str(e)})
            continue
    _elog("router_load_failed", {"module": module})
    return None

ROUTER_NAMES = [
    "optimize", "coverletter", "talk", "superhuman", "humanize", "mastermind",
    "dashboard", "models_router", "context_store",
    "utils_router", "debug",
]
ROUTERS: Dict[str, object] = {name: _safe_import(name) for name in ROUTER_NAMES}

# ============================ CORS ======================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# ======================= Static + Frontend Mount ========================
FRONTEND_DIR = os.path.normpath(os.path.join(ROOT_DIR, "frontend"))
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    _elog("static_mounted", {"dir": STATIC_DIR})
else:
    _elog("static_missing", {"dir": STATIC_DIR})

def _frontend_path(filename: str) -> Optional[str]:
    candidates = [
        os.path.join(FRONTEND_DIR, filename),
        os.path.join(ROOT_DIR, filename),
        os.path.normpath(os.path.join(MEIPASS, filename)) if MEIPASS else None,
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

# ============================= Health ===================================
# Provide BOTH variants to eliminate any slash redirect.
@app.get("/health", include_in_schema=False)
def health():
    return JSONResponse(
        {"ok": True, "service": "ASTRA", "version": APP_VERSION},
        headers={"Cache-Control": "no-store"}
    )

@app.get("/health/", include_in_schema=False)
def health_slash():
    return JSONResponse(
        {"ok": True, "service": "ASTRA", "version": APP_VERSION},
        headers={"Cache-Control": "no-store"}
    )

# ====== Optional: route inspector to quickly verify registered routes ===
@app.get("/__routes__", include_in_schema=False)
def __routes__():
    return sorted(
        f"{','.join(sorted(r.methods))} {getattr(r, 'path', getattr(r, 'path_format',''))}"
        for r in app.router.routes
    )

# ================== Middleware — Request/Response log ===================
@app.middleware("http")
async def trace_requests(request: Request, call_next):
    start = time.time()
    path = request.url.path
    method = request.method

    # Reduce noise: skip chatty paths and preflights
    skip_exact = {"/health", "/health/", "/__routes__", "/favicon.ico", "/api/debug/log"}
    skip_prefixes = ("/static/",)
    log_this = (
        VERBOSE
        and method != "OPTIONS"
        and path not in skip_exact
        and not any(path.startswith(p) for p in skip_prefixes)
    )

    if log_this:
        _elog("http_request", {"method": method, "path": path})
    try:
        response: Response = await call_next(request)
        ms = (time.time() - start) * 1000
        if log_this:
            _elog("http_response", {
                "method": method, "path": path, "status": response.status_code, "ms": round(ms, 1)
            })
        # Cache policy
        if path.startswith("/static/"):
            response.headers["Cache-Control"] = "public, max-age=604800"
        elif path.endswith(".html"):
            response.headers["Cache-Control"] = "no-store"
        return response
    except Exception as e:
        _elog("middleware_error", {"path": path, "error": str(e)})
        return JSONResponse({"error": "internal_middleware_error", "detail": str(e)}, status_code=500)

# ============================ Frontend ==================================
@app.get("/", include_in_schema=False)
def serve_index():
    for fname in ("index.html", "master.html"):
        f = _frontend_path(fname)
        if f:
            return FileResponse(f, headers={"Cache-Control": "no-store"})
    return JSONResponse({"error": "frontend_not_found"}, status_code=404)

@app.get("/{page_name}", include_in_schema=False)
def serve_page(page_name: str):
    """
    Serve known top-level pages. Do NOT redirect unknowns.
    Reserved paths are excluded so they can 404 or be handled by their routes.
    """
    reserved = {
        "", "/", "api", "docs", "redoc", "openapi.json",
        "health", "health/", "__routes__", "favicon.ico", "static"
    }
    if page_name in reserved:
        raise HTTPException(status_code=404)

    if page_name == "master":
        f = _frontend_path("master.html")
        if f:
            return FileResponse(f, headers={"Cache-Control": "no-store"})
        raise HTTPException(status_code=404)

    page = page_name if page_name.endswith(".html") else f"{page_name}.html"
    f = _frontend_path(page)
    if f:
        return FileResponse(f, headers={"Cache-Control": "no-store"})

    # SPA fallback: serve index.html if present; otherwise 404.
    idx = _frontend_path("index.html")
    if idx:
        return FileResponse(idx, headers={"Cache-Control": "no-store"})
    raise HTTPException(status_code=404)

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    for candidate in ("static/assets/favicon.png", "static/assets/favicon.ico"):
        f = _frontend_path(candidate)
        if f:
            return FileResponse(f)
    return Response(status_code=204)

# =========================== Register Routers ===========================
for name, mod in ROUTERS.items():
    if mod and hasattr(mod, "router"):
        app.include_router(mod.router)
        _elog(
            "router_registered",
            {
                "module": name,
                "prefix": getattr(mod.router, "prefix", None),
            },
        )
    else:
        _elog("router_skipped", {"module": name})

# =========================== System Control =============================
def _delayed_shutdown(delay: float = 0.6):
    time.sleep(delay)
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        os._exit(0)

@app.post("/api/system/shutdown", tags=["System"])
def system_shutdown(payload: Optional[Dict[str, Any]] = Body(None)):
    _elog("shutdown_requested", {"by": "ui", "payload": payload or {}})
    threading.Thread(target=_delayed_shutdown, daemon=True).start()
    return {"ok": True, "message": "Server will shut down now."}

# ==================== Web Server + Browser Launcher =====================
# Infinite port-fallback: shared selection for server + browser opener
_SELECTED_HOST_PORT: Tuple[str, int] | None = None

def _env_host_port() -> Tuple[str, int]:
    host = os.getenv("ASTRA_HOST") or os.getenv("HIREX_HOST") or "127.0.0.1"
    try:
        port = int(os.getenv("ASTRA_PORT") or os.getenv("HIREX_PORT") or "8000")
    except Exception:
        port = 8000
    return host, port

def _parse_list_env(name: str) -> list[int]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return []
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            n = int(tok)
            if 0 < n < 65536:
                out.append(n)
        except Exception:
            continue
    return out

def _parse_range_env(name: str) -> Tuple[int, int] | None:
    raw = (os.getenv(name) or "").strip()
    if not raw or "-" not in raw:
        return None
    a, b = raw.split("-", 1)
    try:
        lo = max(1024, int(a.strip()))
        hi = min(65535, int(b.strip()))
        if lo <= hi:
            return (lo, hi)
    except Exception:
        pass
    return None

def _is_port_free(host: str, port: int) -> bool:
    """
    True if we can bind to (host, port) on at least one resolved address family.
    Avoid false negatives by succeeding on any family that binds.
    """
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, socket.AI_PASSIVE)
    except Exception:
        infos = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (host, port))]
    for af, socktype, proto, _, sa in infos:
        try:
            with closing(socket.socket(af, socktype, proto)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(sa)
                return True
        except OSError:
            continue
        except Exception:
            continue
    return False

def _iter_candidates(preferred: int, rng: Tuple[int, int] | None) -> Iterable[int]:
    """
    Yield ports to try in this order:
      1) ASTRA_PORT_LIST (CSV),
      2) preferred (ASTRA_PORT/HIREX_PORT or default 8000),
      3) ASTRA_PORT_RANGE inclusive (e.g., 9000-9500),
      4) full sweep: preferred..65535 then wrap 1024..preferred-1,
      5) if somehow exhausted, loop 1024..65535 forever.
    """
    listed = _parse_list_env("ASTRA_PORT_LIST")
    tried: Set[int] = set()

    # 1) explicit list
    for p in listed:
        if p not in tried:
            tried.add(p); yield p

    # 2) preferred
    if 0 < preferred < 65536 and preferred not in tried:
        tried.add(preferred); yield preferred

    # 3) configured range
    if rng:
        lo, hi = rng
        for p in range(lo, hi + 1):
            if p not in tried:
                tried.add(p); yield p

    # 4) full sweep
    start = max(1024, preferred if preferred >= 1024 else 1024)
    for p in range(start, 65536):
        if p not in tried:
            tried.add(p); yield p
    for p in range(1024, start):
        if p not in tried:
            tried.add(p); yield p

    # 5) infinite fallback
    while True:
        for p in range(1024, 65535):
            yield p

def _select_host_port() -> Tuple[str, int]:
    host, preferred = _env_host_port()
    rng = _parse_range_env("ASTRA_PORT_RANGE")  # e.g. "9000-9500"
    for p in _iter_candidates(preferred, rng):
        if _is_port_free(host, p):
            _elog("port_selected", {"host": host, "port": p})
            # Keep env consistent for children/diagnostics
            os.environ["ASTRA_PORT"] = str(p)
            return host, p
    # Fallback (should never hit due to infinite generator)
    return host, preferred

def _read_host_port() -> Tuple[str, int]:
    global _SELECTED_HOST_PORT
    if _SELECTED_HOST_PORT is None:
        _SELECTED_HOST_PORT = _select_host_port()
    return _SELECTED_HOST_PORT

def _open_host_for_bind(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host

def _open_url(url: str) -> bool:
    if sys.platform.startswith("win"):
        try:
            os.startfile(url)
            return True
        except Exception as e:
            _dlog(f"os.startfile failed: {e}")
            try:
                subprocess.Popen(["cmd", "/c", "start", "", url], shell=True)
                return True
            except Exception as e2:
                _dlog(f"cmd start failed: {e2}")
    try:
        return webbrowser.open_new_tab(url)
    except Exception as e3:
        _dlog(f"webbrowser failed: {e3}")
        return False

def _wait_for_health(base_url: str, timeout_s: float = 200.0, interval_s: float = 0.5) -> bool:
    """Poll /health until 200 OK or timeout."""
    deadline = time.time() + timeout_s
    for _ in range(int(timeout_s / interval_s)):
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=300) as r:
                if 200 <= r.status < 300:
                    return True
        except Exception:
            pass
        time.sleep(interval_s)
    return False

def launch_browser_when_ready():
    if os.getenv("ASTRA_NO_BROWSER", "0") == "1" or os.getenv("HIREX_NO_BROWSER", "0") == "1":
        _elog("browser_auto_open_skipped", {"reason": "NO_BROWSER=1"})
        return
    host, port = _read_host_port()
    base_url = f"http://{_open_host_for_bind(host)}:{port}"
    _dlog(f"OPEN-WAIT {base_url} (bind={host})")
    ready = _wait_for_health(base_url)
    _dlog(f"HEALTH {'OK' if ready else 'TIMEOUT'} {base_url}")
    _open_url(base_url)

# --------- NEW: pick N free ports for multi-instance cluster ------------
def _pick_free_ports(n: int) -> list[Tuple[str, int]]:
    """Return up to n (host,port) pairs that are currently free, using the same selection logic."""
    host, preferred = _env_host_port()
    rng = _parse_range_env("ASTRA_PORT_RANGE")
    pairs, tried = [], set()
    for p in _iter_candidates(preferred, rng):
        if p in tried:
            continue
        tried.add(p)
        if _is_port_free(host, p):
            pairs.append((host, p))
            if len(pairs) >= n:
                break
    return pairs

def start_backend():
    import uvicorn
    host, port = _read_host_port()
    _elog("backend_start", {"host": host, "port": port})
    try:
        workers = max(1, int(os.getenv("ASTRA_WORKERS", "1")))
    except Exception:
        workers = 1
    uvicorn.run(
        app, host=host, port=port,
        log_level="error", timeout_keep_alive=25,
        reload=False, access_log=False, log_config=None,
        workers=workers,
    )

# ================================ Main ==================================
if __name__ == "__main__":
    # Choose and cache the final port once for this process
    host, port = _read_host_port()

    # ----- NEW: simple multi-instance cluster spawner -------------------
    is_child = os.getenv("ASTRA_CHILD", "0") == "1"
    try:
        instances = max(1, int(os.getenv("ASTRA_INSTANCES", "1")))
    except Exception:
        instances = 1

    if instances > 1 and not is_child:
        extra_needed = instances - 1
        extra_pairs = _pick_free_ports(extra_needed)
        if len(extra_pairs) < extra_needed:
            print(f"⚠️ Requested {instances} instances, only found {1 + len(extra_pairs)} free ports.")
        children = []
        py = sys.executable or "python"
        this_file = os.path.abspath(__file__)

        for (h, p) in extra_pairs:
            env = os.environ.copy()
            env["ASTRA_CHILD"] = "1"
            env["ASTRA_HOST"] = h
            env["ASTRA_PORT"] = str(p)
            # Optional: prevent child instances from opening extra tabs
            if os.getenv("ASTRA_OPEN_CHILD_BROWSER", "0") != "1":
                env["ASTRA_NO_BROWSER"] = "1"
                env["HIREX_NO_BROWSER"] = "1"

            try:
                proc = subprocess.Popen([py, this_file], env=env, close_fds=True)
                children.append((proc.pid, h, p))
                print(f"👶 Spawned ASTRA child PID={proc.pid} at http://{_open_host_for_bind(h)}:{p}")
            except Exception as e:
                print(f"❌ Failed to spawn child on {h}:{p}: {e}")

        print(f"🧩 Cluster: parent on http://{_open_host_for_bind(host)}:{port} plus {len(children)} child instance(s).")

    print(f"🚀 Launching ASTRA v{APP_VERSION} — Web Server Mode")
    print(f"RUNNING FILE: {__file__}")
    print(f"🟢 Visit → http://{_open_host_for_bind(host)}:{port}\n")

    def _graceful_exit(signum, _):
        print("\n🛑 Exiting ASTRA…")
        os._exit(0)

    for sig in ("SIGINT", "SIGTERM"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), _graceful_exit)

    # Single opener thread that waits for /health (prevents double-open)
    # Parent opens browser; children only open if ASTRA_OPEN_CHILD_BROWSER=1
    if is_child and os.getenv("ASTRA_OPEN_CHILD_BROWSER", "0") != "1":
        pass
    else:
        threading.Thread(target=launch_browser_when_ready, daemon=True).start()

    start_backend()
