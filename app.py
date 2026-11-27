"""Top-level Streamlit entrypoint for ForgeAI — Autonomous Agent Factory."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure src/ is importable on cloud hosts (e.g., Streamlit Community Cloud) without editable install
_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from forgeai.ui.dashboard import render_dashboard  # noqa: E402

load_dotenv()
render_dashboard()
