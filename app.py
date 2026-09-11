"""Streamlit entry point.

Run with::

    streamlit run app.py

Keeps ``src/`` importable when the project is used from a checkout without an
editable install, then delegates all rendering to ``credit_risk.ui``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from credit_risk.ui.app import render_app  # noqa: E402

render_app()
