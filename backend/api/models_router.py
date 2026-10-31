# ============================================================
#  ASTRA v1.0.0 — models_router.py
#  Exposes model catalogs + pricing for the frontend picker.
#  Reads from backend.core.config (AVAILABLE_MODELS, MODEL_PRICING, etc.)
# ============================================================

from __future__ import annotations

from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException

from backend.core import config

router = APIRouter(prefix="/api/models", tags=["models"])


def _providers() -> List[str]:
    """Stable, sorted provider list for deterministic UI rendering."""
    return sorted(getattr(config, "AVAILABLE_MODELS", {}).keys())


def _available() -> Dict[str, Any]:
    """Provider -> list of models or modes."""
    return getattr(config, "AVAILABLE_MODELS", {})


def _pricing() -> Dict[str, Any]:
    """Provider -> pricing dict."""
    return getattr(config, "MODEL_PRICING", {})


def _aliases() -> Dict[str, str]:
    """Optional alias map (human label -> model id)."""
    return getattr(config, "MODEL_ALIASES", {})


@router.get("")
async def list_models():
    """
    Aggregate endpoint consumed by the frontend to render model pickers.
    """
    return {
        "default_model": getattr(config, "DEFAULT_MODEL", ""),
        "providers": _providers(),
        "available": _available(),
        "pricing": _pricing(),
        "aliases": _aliases(),  # safe even if empty
        "version": config.APP_VERSION,
    }


@router.get("/openai")
async def list_openai():
    """
    Return only OpenAI models and their pricing.
    """
    available = _available().get("openai", [])
    pricing = _pricing().get("openai", {})
    return {
        "provider": "openai",
        "models": available,
        "pricing": pricing,
        "default": getattr(config, "DEFAULT_MODEL", ""),
        "aliases": _aliases(),
        "version": config.APP_VERSION,
    }


@router.get("/aihumanize")
async def list_aihumanize():
    """
    Return AIHumanize modes (styles/modes, not token-metered models)
    and the display pricing/plans info if present.
    """
    available = _available().get("aihumanize", [])
    pricing = _pricing().get("aihumanize", {})
    return {
        "provider": "aihumanize",
        "modes": available,
        "pricing": pricing,  # e.g. {"modes":[...], "plans": {...}, "unit": "subscription"}
        "version": config.APP_VERSION,
    }


@router.get("/provider/{name}")
async def list_by_provider(name: str):
    """
    Generic provider fetch. Helpful for future providers.
    """
    key = name.lower().strip()
    available = _available()
    pricing = _pricing()

    if key not in available:
        raise HTTPException(status_code=404, detail=f"Provider '{key}' not found")

    return {
        "provider": key,
        "available": available.get(key, []),
        "pricing": pricing.get(key, {}),
        "version": config.APP_VERSION,
    }


@router.get("/pricing")
async def pricing_only():
    """
    Raw pricing object for UI tables.
    """
    return {
        "pricing": _pricing(),
        "version": config.APP_VERSION,
    }


@router.get("/aliases")
async def aliases_only():
    """
    Optional alias map (human label -> model id).
    """
    return {
        "aliases": _aliases(),
        "version": config.APP_VERSION,
    }
