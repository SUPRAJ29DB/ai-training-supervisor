"""
Jobs dashboard page – standalone for multi-page Streamlit apps.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000/api/v1"
st.set_page_config(page_title="Training Jobs", page_icon="⚙️", layout="wide")
st.title("⚙️ Training Jobs")
st.info("Use the sidebar navigation in the main dashboard (home.py).")
