"""Streamlit presentation layer. Nothing here contains business logic."""

from __future__ import annotations

__all__ = ["render_app"]


def __getattr__(name: str) -> object:
    """Lazily expose ``render_app`` so importing the package needs no Streamlit."""
    if name == "render_app":
        from credit_risk.ui.app import render_app

        return render_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
