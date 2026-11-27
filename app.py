"""Top-level Streamlit entrypoint for ForgeAI — Autonomous Agent Factory."""

from dotenv import load_dotenv

from forgeai.ui.dashboard import render_dashboard

load_dotenv()
render_dashboard()

