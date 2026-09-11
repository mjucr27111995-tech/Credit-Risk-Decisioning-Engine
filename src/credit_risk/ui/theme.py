"""Stylesheet loading.

The stylesheet lives in ``assets/interface.css`` rather than in a Python string
so that editors lint and highlight it as CSS. Design tokens are declared here
from :mod:`credit_risk.ui.palette` and injected as custom properties, so the
Python constants remain the single source of truth for colour.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Final

import streamlit as st

from credit_risk.ui import palette

_STYLESHEET: Final[Path] = Path(__file__).parent / "assets" / "interface.css"

_TOKENS: Final[dict[str, str]] = {
    "--crm-ink": palette.INK,
    "--crm-ink-secondary": palette.INK_SECONDARY,
    "--crm-muted": palette.MUTED,
    "--crm-faint": palette.FAINT,
    "--crm-hairline": palette.HAIRLINE,
    "--crm-hairline-strong": palette.HAIRLINE_STRONG,
    "--crm-canvas": palette.CANVAS,
    "--crm-surface": palette.SURFACE,
    "--crm-accent": palette.ACCENT,
    "--crm-accent-soft": palette.ACCENT_SOFT,
    "--crm-brass": palette.BRASS,
    "--crm-brass-rule": palette.BRASS_RULE,
    "--crm-brass-soft": palette.BRASS_SOFT,
    "--crm-amber": palette.AMBER,
    "--crm-serif": palette.SERIF,
    "--crm-sans": palette.SANS,
    "--crm-mono": palette.MONO,
}


@lru_cache(maxsize=1)
def _stylesheet() -> str:
    """Read the stylesheet once and prepend the token declarations."""
    declarations = "\n".join(f"  {name}: {value};" for name, value in _TOKENS.items())
    return f"<style>\n:root {{\n{declarations}\n}}\n{_STYLESHEET.read_text()}</style>"


def apply_theme() -> None:
    """Inject the stylesheet. Safe to call on every Streamlit rerun."""
    st.markdown(_stylesheet(), unsafe_allow_html=True)
